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
phase = (t_targetVblank − t_realFrameVblank) / frametimeEMA
```

- `t_targetVblank` and `t_realFrameVblank` both come from the vblank timer
  (`GetVBlankTimer().GetNextVBlank(0)`), whose clock is fed by the backend's real
  flip feedback — **KMS pageflip timestamps** on DRM (`MarkVBlank` from the
  pageflip handler). This is vendor-agnostic ground truth for when frames actually
  scan out; no Vulkan extension, no new timer machinery.
- `frametimeEMA` is a **slew-limited EMA** of the real-frame interval (alpha 1/8,
  sample clamped to [EMA/2, EMA*2]). Composite times are quantized by the vblank
  wakes, so a fractional-rate game yields alternating 1- and 2-vblank samples —
  the true frametime exists only as their average, never in any single sample.
- Each slot targets **the vblank after the one the current wake is deciding**, so
  the dispatch has a full refresh interval of GPU budget and the present path's
  completion check almost never sees an unfinished slot. The existing early vblank
  wake *is* the JIT timer — zero new scheduling primitives.

Three trigger sites, all converging on `framegen_jit_submit()`:

1. **Real-frame record** (`framegen_record_real_frame` tail): supersede any stale
   prediction, then speculate one slot for the next vblank — but only when the
   measured cadence says gaps are coming (**keep-up guard**: skip when
   EMA < 1.10x the vblank interval). This is also the previously missing
   "skip when keeping up" guard for the speculative path: a game that holds
   refresh no longer burns the compositing GPU's bandwidth on always-discarded
   predictions.
2. **Consume drain** (`vulkan_framegen_consume_generated_frame`): after a slot
   presents and the pending list empties, plan the next slot one vblank ahead
   (replaces `framegen_refill_idle`'s k/gap slot ladder in JIT mode).
3. **Repeat-slot tick** (`vulkan_framegen_jit_tick` from the present decision):
   when a vblank goes to a hardware repeat while framegen is active (a stall, a
   too-slow discard, or a mispredicted keep-up), fill from the next vblank — a
   hole never exceeds one vblank.

What this fixes and what it doesn't:

- **Fixed**: generated frames' content-time exactly matches their display-time
  relative to the last real frame's scanout. Within one real frame's "reign",
  displayed motion velocity is exactly correct; the sawtooth on generated frames
  is gone. At integer rates (30 fps at 60 Hz) the computed phases degenerate to
  the classic values (0.5, then a 1.0 insurance slot) — proven behavior preserved.
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
- **GPU budget**: the dedicated framegen queue + one-batch-in-flight guard
  (`hasCompletedFramegen`) — unchanged. JIT batches are always one slot, so the
  #04 ladder's rung costs are keyed by count 1 and only the mode rung
  (motion→extrapolate) sheds work; the "does the step actually help" check
  correctly never takes a multiplier notch.
- **Safety net**: the existing supersede/discard present logic is untouched. A
  real frame still discards any pending prediction; a too-slow slot is still
  dropped in favor of a hardware repeat.

## Integration points (as implemented)

- `FramegenHistory_t::ulFrametimeEmaNs`, `ulCurrentRealVblankNs` — estimator state
  and display-clock anchor; reset in `vulkan_framegen_invalidate_history` (the
  estimator re-seeds from the first post-prime interval).
- `framegen_record_real_frame` — anchors the real frame
  (`ulCurrentRealVblankNs = GetNextVBlank(0)`), folds the interval into the EMA
  (after the frame-gap invalidation, so stall intervals never poison it), and runs
  the JIT branch (supersede + keep-up guard + speculative slot) instead of batch
  planning.
- `framegen_submit_planned` — the dispatch core, now taking explicit
  `FramegenSlotRequest_t{phase, strength, slotIndex}` entries;
  `framegen_submit_batch` (k/gap planning) and `framegen_jit_submit`
  (display-clock planning) are thin planners on top. Everything downstream
  (single command buffer, timestamp bracketing, pending bookkeeping, cross-queue
  read pins) is shared and unchanged.
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
  slots; JIT generates one slot per empty-or-possibly-empty vblank and the keep-up
  guard eliminates the always-discarded case entirely. At integer rates the
  profile matches the old path (fill + one insurance slot per real frame). Motion
  mode pays one motion-estimation pass per slot instead of per batch; at the
  target regime (≤2 fills per real frame) this is comparable, and the motion field
  could later be cached per real-frame pair if x4-style stall fills matter.
- **CPU**: one `GetNextVBlank` call and a few integer ops per trigger; no new
  threads, timers, or waits. All framegen state stays on the steamcompmgr thread.

## Interactions

- **#04 ladder**: fully compatible; JIT rung costs are cleaner (always count=1).
- **Proposal 01 (VRR hybrid)**: orthogonal and complementary — 01 dissolves the
  problem on VRR panels by presenting real frames at their natural time; JIT fixes
  fixed-refresh outputs, which is where the marginal-cadence regime lives.
- **Nested mode**: paces against `g_nNestedRefresh` exactly like the batch path
  (same interval derivation), and tolerates an unmarked vblank clock (constant
  bias only, see above).
- **Shared queue / no dedicated queue**: JIT disabled; classic path unchanged.

## Risks & mitigations

| # | Risk | Mitigation |
|---|------|------------|
| 1 | EMA mis-converges on bimodal frametimes (e.g. 60/30 alternation) | slew clamp bounds each step to 12.5%; worst case phase error equals today's single-sample error; re-seeds on scene change |
| 2 | Keep-up guard misjudges a jittery ~58 fps game and leaves single-vblank holes | repeat-slot tick fills from the second vblank; threshold (`k_uJitKeepUpPercent = 110` in `framegen/scheduling.hpp`) is a named constant to tune with `--framegen-debug` slot logs |
| 3 | `GetLastVBlank` staleness after missed flips skews the target by one vblank | `GetNextVBlank`'s catch-up loop self-corrects next tick; supersede/discard bound the damage to one misplaced slot |
| 4 | Per-slot motion estimation cost at deep stall fills | forward cap bounds fills per stall (~2 at default strength); future: cache the motion field per (previousReal, currentReal) pair |
| 5 | Double-submit races between trigger sites | all three sites run on the steamcompmgr thread and `framegen_jit_submit` refuses when `pending` is non-empty or the previous batch is in flight |

## Testing & validation plan

- `--framegen-debug` now logs `jit slot phase=… strength=… target=+…ms
  ema=…ms` and `jit keep-up skip ema=…ms interval=…ms`; correlate with the
  existing `vblank slot=real|generated|repeat` classification.
- **Cadence test**: vkmark or a game capped to 45 fps on a 60 Hz output
  (`run-framegen-native.sh`); count `slot=repeat` holes and discarded predictions
  with JIT off vs on — discards should drop sharply and generated phases should
  track 0.75/1.5 instead of 0.5.
- **Keep-up test**: content at refresh rate; with JIT on, generation should go
  fully quiescent (`jit keep-up skip` in logs, zero generated slots) — the batch
  path generates-and-discards every frame.
- **Integer-rate regression**: 30 fps at 60 Hz should reproduce classic x2
  phases (0.5) bit-for-bit through the same shaders.
- **Stall behavior**: freeze the game (SIGSTOP); fills should walk 0.75 → 1.5 and
  quiesce at the cap, then invalidate at the 250 ms idle gap as before.

## Open questions

1. Should the keep-up guard hysteresis differ from 1.10x (e.g. engage/disengage
   thresholds) to avoid flapping right at the boundary?
2. Same-vblank JIT: with measured rung cost ≪ the wake's redzone offset, a slot
   could target the *current* vblank and rely on implicit sync (the atomic
   commit's in-fence) to make the flip safe — worth it, or does a missed fence
   (flip lands a vblank late) outweigh the one-vblank-earlier fill?
3. Should the EMA (not the single-sample rounding) also feed the batch path's
   `nMeasuredGapVblanks` when JIT is off?
4. Promote `GAMESCOPE_FRAMEGEN_JIT` to a CLI flag / default-on for dedicated-queue
   configs once validated on the AMD dual-GPU target?
