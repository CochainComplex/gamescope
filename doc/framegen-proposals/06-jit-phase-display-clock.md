# Proposal 06 — JIT Phase: Display-Clock Pacing for Generated Frames

Status: **prototype implemented** (opt-in via `GAMESCOPE_FRAMEGEN_JIT=1`, requires the
dedicated framegen queue)
Scope: temporal placement of generated frames on fixed-refresh outputs
Owner-area files: `src/rendervulkan.cpp`, `src/rendervulkan.hpp`, `src/steamcompmgr.cpp`

---

## Motivation / problem it solves

The hardest unsolved problem in compositor-side frame generation is **temporal
placement, not pixel synthesis**: on a fixed-refresh display, a game whose frametime
sits between ~1.0x and ~1.5x the vblank interval (40–59 fps at 60 Hz — the regime
where users most want framegen) cannot have both content-time and display-time
correct, because the batch path stamps them at different moments using different
clocks:

- **Pixels are stamped before the schedule is known.** `framegen_submit_batch`
  bakes `phase = k/nGapVblanks` into the image at *generation* time. The frame's
  content-time is frozen there.
- **The gap is guessed from one noisy sample.** `nMeasuredGapVblanks` is a rounding
  of the single previous real-frame interval. Frametime jitter maps directly into
  phase error; displayed position error = velocity × timing error, worst on fast
  pans (the "temporal wobble").
- **Display time is quantized by a third clock.** The slot presents at whatever
  vblank it wins in the steamcompmgr present decision, FIFO from the pending list,
  with no notion of "which vblank was I made for".

Concretely, 45 fps at 60 Hz, x2, dedicated queue (batch path): real arrivals
quantize to alternating 1- and 2-vblank gaps; the speculative batch plans one slot
at phase 0.5 from a gap guess of 2; the displayed position-vs-time curve is a
sawtooth (generated frames undershoot their vblank by several ms of scene motion,
roughly half the speculative batches are discarded). This is 3:2-pulldown judder
with extra steps, and no amount of shader quality fixes it — it is a *pacing*
defect.

The underlying trilemma: pick two of {zero added latency, regular display cadence,
real-pixel fidelity}. DLSS-FG/FSR3 pick cadence+fidelity by *delaying the real
frame* (interpolation). Gamescope's invariant #1 (real frames are never delayed)
forbids that. This proposal keeps latency+fidelity and recovers as much cadence
correctness as is possible without re-timing real frames: **make the generated
frames' phase an observation instead of a prediction.**

## Design overview

Replace batch planning with **just-in-time single-slot planning against the
display clock**:

```
phase = (t_targetVblank − t_realFrameVblank) / predictedSourceInterval
```

- `t_targetVblank` and `t_realFrameVblank` both come from the vblank timer
  (`GetVBlankTimer().GetNextVBlank(0)`), whose clock is fed by the backend's real
  flip feedback — **KMS pageflip timestamps** on DRM (`MarkVBlank` from the
  pageflip handler). This is vendor-agnostic ground truth for when frames actually
  scan out; no Vulkan extension, no new timer machinery.
- `predictedSourceInterval` comes from a small deterministic **online alpha-beta
  cadence filter**. Its observations are acquire-fence completion timestamps,
  copied from the selected base-layer commit before fixed-refresh quantization.
  The filter learns period and bounded first-order drift. A one-sided leaky peak
  learns recent late prediction error without treating early frames as risk.
  Backends without a source timestamp fall back to composite time for phase
  cadence and retain the conservative legacy 110%-of-refresh admission gate;
  they do not pretend a display-quantized sample is a source-arrival deadline.
  A model is reset rather than mixing those two timestamp provenances.
- Each slot targets **the vblank after the one the current wake is deciding**, so
  the dispatch has a full refresh interval of GPU budget and the present path's
  completion check almost never sees an unfinished slot. The existing early vblank
  wake *is* the JIT timer — zero new scheduling primitives.

Before spending GPU work, real-frame record performs deadline admission:

```
predictedReady = latestSourceReady + predictedSourceInterval
protectedReady = predictedReady + lateErrorEnvelope + arrivalGuard
skip backup only when now < protectedReady < nextCompositorWake
```

The arrival guard is `max(250 us, refreshInterval/32)`. The first four cadence
samples always request a backup. An estimate already in the past is considered
missed: absence of a newer selected source buffer is causal evidence, so the
backup is generated even if the display deadline is still farther ahead.

Three trigger sites, all converging on `framegen_jit_submit()`:

1. **Real-frame record** (`framegen_record_real_frame` tail): supersede any stale
   prediction, then request one backup only when the learned source arrival is
   untrained, overdue, or at/after the next compositor wake deadline. A game
   that is confidently keeping up therefore stops burning the presentation GPU
   on always-discarded predictions, while a marginal or jittering game remains
   covered.
2. **Consume drain** (`vulkan_framegen_consume_generated_frame`): after a slot
   presents and the pending list empties, plan the next slot one vblank ahead
   (replaces `framegen_refill_idle`'s k/gap slot ladder in JIT mode).
3. **Repeat-slot tick** (`vulkan_framegen_jit_tick` from the present decision):
   when a vblank goes to a hardware repeat while framegen is active (a stall, a
   too-slow discard, or a mispredicted keep-up), request the earliest slot that
   can still be prepared. A keep-up miss necessarily leaves the current slot as
   one repeat; unfinished GPU work or the forward-prediction cap can require
   further honest repeats rather than stalling presentation.

This is the variable multiplier: there is no pre-baked x2/x3/x4 presentation
batch. The compositor fills one display slot at a time until a real frame wins
or the bounded forward-prediction cap is reached. The CLI multiplier still
sizes existing resources and degradation policy, but it does not force a fixed
generated-to-real ratio in JIT mode.

What this fixes and what it doesn't:

- **Fixed**: each generated frame is stamped for a specific measured display
  slot instead of a guessed batch position. Source-ready cadence removes the
  fixed-vblank quantization from the interval estimate, so fractional-rate
  phase error no longer directly follows the alternating display-gap pattern.
  Prediction error is still possible, but it is bounded and evolves smoothly.
  At stable integer rates (30 fps at 60 Hz) the computed phases converge to the
  classic values (0.5, then a 1.0 insurance slot).
- **Not fixed**: real frames still present at quantized vblanks with a varying
  content-to-display offset (bounded by ±half a vblank, identical to framegen-off
  vsync). Removing *that* requires re-timing real frames too (present only
  synthesized frames, VR-compositor style) — deliberately out of scope until the
  motion pipeline is trustworthy everywhere.

## Mechanisms (all vendor-agnostic)

- **Display clock**: KMS pageflip timestamps via `CVBlankTimer::GetLastVBlank()` /
  `GetNextVBlank()`. Nested backends feed the same API; a backend that never marks
  vblanks degrades to an epoch-aligned grid whose offset **cancels** in the phase
  subtraction (anchor and target sit on the same grid) — constant phase bias at
  worst, never a sawtooth.
- **Source clock**: `commit_t::Signal()` records `CLOCK_MONOTONIC` time when the
  acquire fence is ready. `FrameInfo_t::Layer_t::acquireReadyTimeNs` carries the
  selected base layer's observation to framegen. This is read-only scheduling
  metadata: it is not client presentation feedback and generated layers leave
  it zero.
- **GPU budget**: the dedicated framegen queue + one-batch-in-flight guard
  (`hasCompletedFramegen`) — unchanged. JIT batches are always one slot, so the
  #04 ladder's rung costs are keyed by count 1 and only the mode rung
  (motion→extrapolate) sheds work; the "does the step actually help" check
  correctly never takes a multiplier notch.
- **Safety net**: the existing supersede/discard present logic is untouched. A
  real frame still discards any pending prediction; a too-slow slot is still
  dropped in favor of a hardware repeat.

## Integration points (as implemented)

- `FramegenHistory_t::cadence`, `ulCurrentCadenceTimeNs`,
  `bCadenceUsesSourceTime`, and `ulCurrentRealVblankNs` — predictor state,
  timestamp provenance, and display-clock anchor. All reset in
  `vulkan_framegen_invalidate_history`.
- `framegen/scheduling.hpp` — pure, constexpr cadence update and fixed-deadline
  admission. It owns no clock, queue, mutable global, or submission decision;
  boundary behavior is covered by `tests/test_framegen.cpp`.
- `framegen_record_real_frame` — anchors the real frame
  (`ulCurrentRealVblankNs = GetNextVBlank(0)`), updates the source-cadence model
  after frame-gap invalidation, obtains the exact next wake deadline through
  non-mutating `CalcNextWakeupTime(true)`, and runs deadline admission instead
  of batch planning.
- `framegen_submit_planned` — the dispatch core, now taking explicit
  `FramegenSlotRequest_t{phase, strength, slotIndex}` entries;
  `framegen_submit_batch` (k/gap planning) and `framegen_jit_submit`
  (display-clock planning) are thin planners on top. Everything downstream
  (single command buffer, timestamp bracketing, pending bookkeeping, cross-queue
  read pins) is shared. Completed read pins retire before real-output
  acquisition; this is lifetime maintenance, not a scheduling-policy change.
- Real-output acquisition remains ownership-driven. If every slot is busy, the
  exceptional path performs one non-blocking backend event-progress pass before
  sacrificing speculative history; it never waits for framegen or overwrites a
  KMS/nested-Wayland-owned DMA-BUF.
- `framegen_jit_submit` — the JIT planner: guards (history valid, pending empty,
  one-batch-in-flight, idle-gap invalidation), target = `GetNextVBlank(0) +
  interval`, phase/strength computation, forward cap (`flStrengthRaw >
  k_flFramegenMaxForwardStrength` → stop generating; a stall quiesces at the cap
  exactly like the old idle refill converged to it).
- `vulkan_framegen_jit_tick` — exported reactive hook; called from
  `steamcompmgr_main` when `vblank && !bShouldPaint` while framegen is active.
- Gate: `framegen_jit_enabled()` = `GAMESCOPE_FRAMEGEN_JIT=1` **and**
  `g_device.hasFramegenQueue()`. The shared-queue fallback keeps today's
  leaky-bucket batch path untouched.

## Latency & throughput analysis

- **Real-frame latency: unchanged (zero added).** JIT never touches the composite
  or present of real frames; a generated slot targets a vblank one interval out
  and is superseded for free if real content arrives.
- **Generation budget**: one slot has a full refresh interval to complete
  (16.7 ms at 60 Hz vs 0.3–2 ms measured slot cost) — the completion-check discard
  ("generation_too_slow") becomes a cold path.
- **Bandwidth**: at 45 fps/60 Hz the batch path discarded ~half its speculative
  slots; JIT generates one slot per empty-or-possibly-empty vblank and trained
  deadline admission eliminates the confidently unnecessary case. At integer rates the
  profile matches the old path (fill + one insurance slot per real frame). Motion
  mode pays one motion-estimation pass per slot instead of per batch; at the
  target regime (≤2 fills per real frame) this is comparable, and the motion field
  could later be cached per real-frame pair if x4-style stall fills matter.
- **CPU**: one acquire-ready timestamp copy, one `CalcNextWakeupTime(true)`, and
  bounded integer arithmetic per real frame; no new thread, timer, allocation,
  blocking call, or wait. Image reconstruction remains Vulkan work on the
  presentation GPU. All scheduling state stays on the steamcompmgr thread.

## Interactions

- **#04 ladder**: fully compatible; JIT rung costs are cleaner (always count=1).
- **Proposal 01 (VRR hybrid)**: orthogonal and complementary — 01 dissolves the
  problem on VRR panels by presenting real frames at their natural time; JIT fixes
  fixed-refresh outputs, which is where the marginal-cadence regime lives.
- **Nested mode**: paces against `g_nNestedRefresh` exactly like the batch path
  (same interval derivation), while host `wp_presentation` feedback re-anchors
  the grid after presented commits. The requested nested refresh must match the
  host display mode; the parent compositor remains the final scanout authority.
  An unmarked vblank clock degrades to a constant phase bias rather than a
  fractional-rate sawtooth (see above).
- **Shared queue / no dedicated queue**: JIT disabled; classic path unchanged.

## Risks & mitigations

| # | Risk | Mitigation |
|---|------|------------|
| 1 | Predictor mis-converges on bimodal frametimes (e.g. 60/30 alternation) | sample clamp and bounded alpha/beta updates prevent one hitch from taking over; the one-sided late envelope biases uncertainty toward generating, not repeating; re-seeds on discontinuity |
| 2 | A sudden first long frame was not predictable | an overdue estimate generates immediately, and the repeat-slot tick fills from the following vblank; arbitrary future stalls cannot be predicted without delaying real frames |
| 3 | `GetLastVBlank` staleness after missed flips skews the target by one vblank | `GetNextVBlank`'s catch-up loop self-corrects next tick; supersede/discard bound the damage to one misplaced slot |
| 4 | Per-slot motion estimation cost at deep stall fills | forward cap bounds fills per stall (~2 at default strength); future: cache the motion field per (previousReal, currentReal) pair |
| 5 | Double-submit races between trigger sites | all three sites run on the steamcompmgr thread and `framegen_jit_submit` refuses when `pending` is non-empty or the previous batch is in flight |

## Testing & validation plan

- `--framegen-debug` logs `jit slot phase=… strength=… target=+…ms
  cadence=…ms`, `jit deadline backup reason=warmup|overdue|deadline …`, and
  `jit deadline skip … headroom=…`; correlate with the
  existing `vblank slot=real|generated|repeat` classification.
- **Cadence test**: vkmark or a game capped to 45 fps on a 60 Hz output
  (`run-framegen-native.sh`); count `slot=repeat` holes and discarded predictions
  with JIT off vs on — discards should drop sharply and generated phases should
  track 0.75/1.5 instead of 0.5.
- **Keep-up test**: content at refresh rate; after four observations, generation
  should go fully quiescent (`jit deadline skip` in logs, zero generated slots) — the batch
  path generates-and-discards every frame.
- **Integer-rate regression**: 30 fps at 60 Hz should reproduce classic x2
  phases (0.5) bit-for-bit through the same shaders.
- **Stall behavior**: freeze the game (SIGSTOP); fills should walk 0.75 → 1.5 and
  quiesce at the cap, then invalidate at the 250 ms idle gap as before.

## Open questions

1. Should the late-error envelope evolve into a bounded online quantile estimate
   after native-DRM traces establish a workload-independent target percentile?
2. Same-vblank JIT: with measured rung cost ≪ the wake's redzone offset, a slot
   could target the *current* vblank and rely on implicit sync (the atomic
   commit's in-fence) to make the flip safe — worth it, or does a missed fence
   (flip lands a vblank late) outweigh the one-vblank-earlier fill?
3. Should the predicted cadence (not the single-sample rounding) also feed the batch path's
   `nMeasuredGapVblanks` when JIT is off?
4. Promote `GAMESCOPE_FRAMEGEN_JIT` to a CLI flag / default-on for dedicated-queue
   configs once validated on the AMD dual-GPU target?
