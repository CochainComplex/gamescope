# Proposal 01 — VRR Hybrid Mode: Timed Mid-Interval Generated Flips

Status: design draft
Scope: compositor-side x2 frame generation on Variable-Refresh-Rate (VRR / adaptive-sync) panels
Owner-area files: `src/steamcompmgr.cpp`, `src/vblankmanager.cpp/.hpp`, `src/Backends/DRMBackend.cpp`, `src/waitable.h`, `src/rendervulkan.cpp`

---

## Motivation / problem it solves

Frame generation and VRR are currently **mutually exclusive** in gamescope. The framegen path
forces a fixed-cadence, non-adaptive presentation model: it slots a generated frame into an
otherwise-empty vblank and presents the real frame on the next vblank, at a fixed refresh rate.
Two places encode this exclusion today:

- `src/steamcompmgr.cpp:2570` —
  `frameInfo.allowVRR = cv_adaptive_sync && !vulkan_framegen_is_enabled();`
  Adaptive sync is disabled outright whenever framegen is on. The inline comment
  (`:2566-2569`) explains why: a generated flip is emitted with `allowVRR=false`, so on a VRR
  panel `VRR_ENABLED` would toggle off on the generated flip and back on for the real flip,
  which forces a modeset every alternate frame (`drm_prepare`, `src/Backends/DRMBackend.cpp:2933-2934`).
- `src/Backends/DRMBackend.cpp:3619` — the generated present hard-codes
  `generatedFrameInfo.allowVRR = false`.

The cost of that exclusion is real. On a VRR panel a game running at, say, 70 fps normally gets:
(a) **low latency** — each real frame is scanned out the instant it is composited, with no wait
for a fixed vblank grid; and (b) **no judder** — the panel tracks the render cadence. Turning
framegen on today throws (a) away: we clamp to a fixed refresh and re-introduce up to one refresh
interval of scan-out latency on every real frame, in exchange for doubled motion clarity. On a
Deck-class OLED/LCD VRR panel that trade is often a net loss for the player, and it also risks
dropping the panel below its **LFC (Low Framerate Compensation)** floor at low fps, where the
panel must self-refresh or multiply frames to stay in its supported range.

**The goal of this proposal:** keep the VRR latency win on the *real* frame (present it
immediately, VRR-style, exactly as if framegen were off, with zero artifacts on real content),
and additionally schedule the *generated* frame as a **timer-armed atomic flip at the midpoint of
the measured real-frame interval**. This doubles perceived motion clarity while preserving the
adaptive-sync latency benefit, and — because the generated flip is itself a present — it keeps the
panel refreshing above its LFC floor at low fps (a generated flip at T+interval/2 halves the
maximum gap between scan-outs).

---

## Design overview

Instead of *suppressing* VRR, we run VRR normally and treat framegen as an **opportunistic
in-between flip**:

1. Real frame is composited and presented immediately via the existing VRR flip path
   (`allowVRR=true`, `FlipType::VRR`). Nothing about the real-frame critical path changes.
2. After the real composite, the generated frame is produced off the critical path exactly as
   today (`framegen_record_real_frame` → async compute submit, `src/rendervulkan.cpp:4416`,
   `:4565`), populating `g_framegenHistory.pendingGenerated`.
3. On presenting the real frame, we **arm a one-shot `CLOCK_MONOTONIC` absolute timer** for
   `T_realflip + I_est/2`, where `I_est` is the estimated current real-frame interval. This is a
   new `gamescope::CTimerFunction` (`g_FramegenMidTimer`) registered on `g_SteamCompMgrWaiter`,
   modeled on the existing `g_FPSLimitVRRTimer` (`src/steamcompmgr.cpp:8311`, `:8461`).
4. When the timer fires, and *iff* (a) a generated frame is ready, (b) no real frame has been
   latched since, and (c) no present is already in flight, we drive one repaint that presents the
   generated frame as a full-screen scanout **with `allowVRR=true`** — so `VRR_ENABLED` does not
   toggle and no modeset is triggered.
5. **Real content always wins.** The moment a new real frame is latched, the pending generated
   frame is discarded and the mid-interval timer is disarmed. A generated flip can never delay a
   real frame; it only fills time the game left idle.

The mid-interval timer is the only genuinely new mechanism. Everything else is a re-wiring of
existing VRR present, framegen consume, and vblank-estimator code.

```
real frame ready ──► composite ──► VRR flip NOW (allowVRR=true, low latency)
                                     │
                                     ├─► async framegen compute (off critical path)
                                     └─► arm g_FramegenMidTimer @ T_flip + I_est/2
                                                     │
      (new real frame arrives?) ──yes──► discard generated + disarm timer ──► (loop)
                                     │ no
                                     ▼
                        timer fires @ midpoint
                                     │
             generated ready && no present in flight && no new real frame?
                        ├─ yes ─► VRR flip generated frame (allowVRR=true)
                        └─ no  ─► SKIP (leave VRR self-refresh; no flip)
```

---

## Vulkan mechanisms & extensions

This proposal is **KMS/DRM-side**; it does not add Vulkan extensions beyond what the framegen core
already uses. For completeness and correctness of the surrounding system:

- **Generation** stays on the existing async-compute submit in `src/rendervulkan.cpp` (`g_device.submit`,
  `:4565`), synchronized via the internal timeline (`g_device.hasCompleted(generatedSeqNo)`,
  `:4310`/`:4494`). No `VkPresent*` is involved — generated frames never touch a Wayland swapchain,
  preserving the critical rule that they emit no `frame_done` / presentation feedback.
- The generated image is a DRM-scanout-capable `CVulkanTexture` from the framegen output ring
  (`g_output.framegenOutputImages`, `:4401`). It is exported as a DMA-BUF / DRM framebuffer and
  presented through the same atomic path as any composited buffer.
- **Presentation** is `drmModeAtomicCommit` (`src/Backends/DRMBackend.cpp:4091`) with flags
  `DRM_MODE_ATOMIC_NONBLOCK | DRM_MODE_PAGE_FLIP_EVENT` (`:2989`, `:3007`). The generated flip
  uses the **same non-async** path as a VRR real flip — it must *not* set
  `DRM_MODE_PAGE_FLIP_ASYNC`, because async/immediate flips are tearing flips and framegen is
  incompatible with tearing (see below).
- Fence discipline: `vulkan_framegen_consume_generated_frame` (`:4299`) already gates on
  `hasCompleted` and, on the miss path, `vulkan_wait`s the generation seqno (`:4323`) so the
  DMA-BUF is guaranteed idle before the KMS commit references it. The DRM `OUT_FENCE_PTR` /
  in-fence plumbing on the CRTC (`drm_prepare`) remains the scan-out completion signal.

The **only new kernel-facing primitive** is the userspace one-shot timer:
`timerfd_create(CLOCK_MONOTONIC, ...)` armed with `TFD_TIMER_ABSTIME`, which gamescope already
wraps in `gamescope::ITimerWaitable::ArmTimer(absTimeNanos)` (`src/waitable.h:167-178`) and
`DisarmTimer()` (`:180-183`). KMS has **no** "flip at absolute time T" primitive — this timer *is*
the scheduling mechanism, and the atomic commit is issued from the timer callback.

---

## Integration points in gamescope

Concrete hooks, by file and current line region:

**A. Stop suppressing VRR when framegen is on (opt-in to hybrid).**
`src/steamcompmgr.cpp:2570`. Change from
`frameInfo.allowVRR = cv_adaptive_sync && !vulkan_framegen_is_enabled();`
to allow VRR while framegen is enabled *when* the new hybrid mode is selected:
`frameInfo.allowVRR = cv_adaptive_sync && (!vulkan_framegen_is_enabled() || cv_framegen_vrr_hybrid);`
The existing "fixed-slot" behavior remains the fallback when `cv_framegen_vrr_hybrid` is off or
VRR is unavailable.

**B. Generated present must not toggle VRR.**
`src/Backends/DRMBackend.cpp:3619`. In the generated-frame present block (`:3613-3649`), set
`generatedFrameInfo.allowVRR = pFrameInfo->allowVRR;` (i.e. inherit the real frame's VRR state)
instead of the hard-coded `false`. This is what prevents the per-frame `VRR_ENABLED` toggle and
the modeset at `drm_prepare` (`:2933-2934`, `:3097-3098`). Under hybrid mode the real and
generated flips both carry `allowVRR=true`, so `bVRREnabled` (`:2930`) is stable and
`needs_modeset` is not raised.

**C. New mid-interval timer object.**
`src/steamcompmgr.cpp` near `g_FPSLimitVRRTimer` (`:8311`). Add:
```cpp
static gamescope::CTimerFunction g_FramegenMidTimer{ []
{
    g_FramegenMidTimer.DisarmTimer();
    g_bFramegenMidDeadline = true;   // consumed by the main loop this iteration
    nudge_steamcompmgr();            // ensure the loop wakes even if idle
} };
```
Register it next to the others: `src/steamcompmgr.cpp:8461`
(`g_SteamCompMgrWaiter.AddWaitable( &g_FramegenMidTimer );`).

**D. Arm the timer right after a real VRR flip.**
`src/steamcompmgr.cpp`, in the present block after `paint_all(...)` / `OnEndFrame()`
(`:9159`, `:9176`). When hybrid mode is active, a real frame was just presented, and a generated
frame is (or will be) pending, compute
`T_mid = T_realflip + I_est/2` and `g_FramegenMidTimer.ArmTimer(T_mid)`.
`I_est` comes from the vblank estimator (`GetVBlankTimer()`), see Data & control flow.

**E. Fire path — present the generated frame at the midpoint.**
`src/steamcompmgr.cpp:9131-9144`. Today this block only presents a pending generated frame when
`vblank` is true. Extend the condition: also present when `g_bFramegenMidDeadline` is set by the
timer callback (and clear it). The existing "real frame supersedes" logic at `:9140-9141`
(`vulkan_framegen_discard_generated_frame("superseded_by_real_frame")`) is reused verbatim, plus a
new `g_FramegenMidTimer.DisarmTimer()` alongside it so the pending flip is *cancelled* the instant
real content latches.

**F. In-flight guard.**
`src/steamcompmgr.cpp:9117-9120` already refuses to paint under VRR when
`PresentationFeedback().CurrentPresentsInFlight() != 0`. The generated flip must obey the same
guard so two flips never race into one scan-out (see Data & control flow, step 6).

**G. Tearing stays mutually exclusive.**
`src/steamcompmgr.cpp:9037-9040`. `bTearing` is already gated on `!vulkan_framegen_is_enabled()`.
No change; the generated flip is always a page-flip-event (non-async) flip.

**H. Discard/disarm on invalidation.**
`vulkan_framegen_invalidate_history` / `vulkan_framegen_discard_generated_frame`
(`src/rendervulkan.cpp:4268`, `:4333`) are called on focus change, resize, etc. Add a
`g_FramegenMidTimer.DisarmTimer()` at those sites (or have the main loop disarm when
`!vulkan_framegen_has_pending_generated_frame()`), so a stale timer never fires against a
discarded frame.

---

## Data & control flow

Per real-frame cycle, in the `steamcompmgr_main` loop (`src/steamcompmgr.cpp:8514` onward):

1. **Wake.** `g_SteamCompMgrWaiter.PollEvents()` (`:8526`) returns because one of: an X/Wayland
   event, the vblank timer, the FPS-limit timer, **or the new `g_FramegenMidTimer`** became ready.
2. **Classify wake.** `GetVBlankTimer().ProcessVBlank()` (`:8529`) tells us if this is a vblank
   tick. Under VRR, `vblank` is forced true (`:8546-8548`). The new
   `g_bFramegenMidDeadline` flag (set in the timer callback, step C) distinguishes a
   *mid-interval* wake from a normal one.
3. **Real frame present (unchanged critical path).** If a real frame is ready, we composite and
   present it via `FlipType::VRR` (`:9053`, `:9092-9123`). This is byte-for-byte the pre-existing
   VRR path; framegen adds nothing here, so **real-frame latency is identical to framegen-off VRR**.
4. **Framegen production (off critical path).** After the real composite,
   `framegen_record_real_frame` (`src/rendervulkan.cpp:4876` call site) submits the generation on
   async compute; `g_framegenHistory.pendingGenerated` becomes non-null once the submit is
   recorded (fence completion tracked separately).
5. **Arm midpoint.** Immediately after the real flip is committed (hook D), compute
   `I_est`. Under VRR, the vblank estimator's `CalcNextWakeupTime` VRR branch
   (`src/vblankmanager.cpp:157-177`) does *not* track a real cadence (it assumes "always can
   vblank"), so `I_est` must come from the **measured interval between the last two real
   `page_flip_handler` vblank timestamps** (`src/Backends/DRMBackend.cpp:764-775`,
   `MarkVBlank`). We keep a short rolling estimate `g_ulRealFrameInterval` updated in
   `page_flip_handler`/`MarkVBlank`. Arm `g_FramegenMidTimer` for
   `T_realflip + g_ulRealFrameInterval/2`, clamped (step 8).
6. **Cancel-on-real.** On any later loop iteration where a new real frame or overlay update is
   latched (`hasRepaint || hasRepaintNonBasePlane`, `:9140`), discard the pending generated frame
   and `DisarmTimer()`. Real content preempts the not-yet-fired midpoint flip with zero added
   latency.
7. **Fire.** When `g_FramegenMidTimer` fires, the callback sets `g_bFramegenMidDeadline` and nudges
   the loop. On the next iteration the extended block at `:9131` presents the generated frame *iff*:
   - `vulkan_framegen_has_pending_generated_frame()` **and** the generation fence has completed
     (`hasCompleted`, enforced inside `vulkan_framegen_consume_generated_frame`, `:4310`); if the
     generation missed the deadline, `consume` discards it (`generation_too_slow`, `:4312`) and we
     simply **skip** — the panel VRR-self-refreshes the real frame, no harm;
   - `CurrentPresentsInFlight() == 0` (`:9119`) — otherwise a real flip is still scanning out;
     skip to avoid two flips racing into one scan-out;
   - no new real frame latched this iteration (else step 6 already won).
   The present goes through `drm_prepare(allowVRR=true)` + `Commit` (`DRMBackend.cpp:3639-3648`).
8. **Miss / clamp handling.** If `now >= T_mid` by the time we would arm (generation or scheduling
   slipped), or if `g_ulRealFrameInterval/2` is below a floor (e.g. < 2 ms, faster than we can
   composite+flip a generated frame), **do not arm** — skip generation for that interval. If the
   real interval is very long (low fps, near LFC), clamp `T_mid` so the generated flip still lands
   inside the panel's max scan-out interval (keeps us above LFC; see VRR interaction).

Why this stays off the real-frame critical path: the only work added *before* a real flip is
arming a timerfd (a single `timerfd_settime`, sub-microsecond). All generation and the generated
flip happen strictly *after* the real flip has been committed, on a timer wake that is preempted by
any real content. The real flip's code path is unchanged from framegen-off VRR.

---

## Latency & throughput analysis

- **Real-frame added latency: ~0.** The real frame takes the unmodified VRR flip path. The only
  addition is one `timerfd_settime` (`ArmTimer`, `src/waitable.h:176`) after the commit — tens to
  hundreds of nanoseconds, and *after* the flip is already queued.
- **Generated-frame budget.** The generated flip must be composited-and-ready and committed before
  `T_mid`. Budget = `I_est/2 − (generation compute time + KMS commit latency)`. At 70 fps,
  `I_est ≈ 14.3 ms`, midpoint budget ≈ 7.1 ms; framegen extrapolate compute on an iGPU is the
  dominant term. If the budget is negative we skip (step 8) — graceful degradation, never a stall.
- **Cross-PCIe traffic.** Unchanged from the existing dual-GPU framegen design: the real composited
  frame already lands in the compositing GPU's memory; generation reads history from the
  output-image ring (zero-copy per Tier-1). The generated flip adds **one extra scan-out per real
  frame** — i.e. up to 2× the KMS commit rate and 2× panel scan-out bandwidth, which the panel
  already sustains (we present within its refresh range, never above `g_nOutputRefresh`).
- **Extra passes.** One extra `drmModeAtomicCommit` + one extra `page_flip_handler` event per real
  frame (`DRMBackend.cpp:751`). Negligible CPU (a few µs) and it happens on the flip-handler
  thread, off the compositor hot path.
- **Throughput ceiling.** Presented frames/s ≤ 2× real fps, capped at `g_nOutputRefresh`. The
  midpoint schedule guarantees the two flips are ~`I_est/2` apart, so we never exceed the panel's
  minimum scan-out interval and never queue a flip the panel cannot absorb.

---

## Interaction with VRR / HDR / anti-lag / tearing

- **VRR:** This is the whole point — VRR stays *active* (`allowVRR=true` on both flips, hook A/B).
  `VRR_ENABLED` is stable across real+generated flips, so no per-frame modeset. The real frame keeps
  its adaptive-sync latency; the generated frame is an extra in-range flip.
- **LFC:** At low fps the interval between real flips can approach or exceed the panel's max
  supported scan-out interval, forcing the panel/driver into LFC (self-refresh or frame-doubling).
  A generated flip at the midpoint *halves* the max gap between presents, which keeps the panel
  comfortably inside its VRR window and reduces reliance on LFC — the generated frame is exactly
  the "extra refresh" LFC would otherwise synthesize, but with real motion-extrapolated content.
  We clamp `T_mid` (step 8) so the generated flip lands before the LFC threshold when the real
  interval is very long.
- **Anti-lag / Reflex / DXVK / Wine:** Untouched. Generated frames create **no Wayland commit and
  emit no `frame_done` / presentation feedback** (they are pure KMS flips of an internal DMA-BUF),
  so the game's frame-pacing and low-latency logic never sees them — the critical rule holds.
  The real frame's presentation feedback is delivered exactly as in framegen-off VRR.
- **HDR:** The generated present already inherits `outputEncodingEOTF` and sets colorspace to
  HDR10_PQ vs SRGB accordingly (`DRMBackend.cpp:3620`, `:3633`). It sets
  `applyOutputColorMgmt=false` because the composited output is already in output encoding. The
  `HDR_OUTPUT_METADATA` / `Colorspace` connector props are unchanged between real and generated
  flips (both non-modeset), so no HDR re-negotiation occurs mid-stream.
- **Tearing:** Strictly exclusive (`src/steamcompmgr.cpp:9039-9040`). The generated flip is always a
  synchronous page-flip-event flip; it never sets `DRM_MODE_PAGE_FLIP_ASYNC`. Tearing and framegen
  cannot be co-enabled.

---

## Risks & mitigations

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|------------|
| 1 | Two flips race into one scan-out (real + generated queued together) | Med | `CurrentPresentsInFlight()==0` guard (`:9119`) reused for generated flip; `uPendingFlipCount` (`DRMBackend.cpp:4072`) enforced; skip generated flip if a present is in flight. |
| 2 | Stale generated frame flips after real content arrived | Med | Cancel-on-real: discard + `DisarmTimer()` on any latched real frame (`:9140`, hook E/H). |
| 3 | `VRR_ENABLED` toggles → modeset every other frame (current bug this replaces) | High if mis-implemented | Generated flip inherits `allowVRR` from real frame (hook B); assert `bVRREnabled` is stable; log a warning if `needs_modeset` is raised while framegen is active. |
| 4 | Generation misses the midpoint deadline | Med | `consume` gates on fence `hasCompleted` (`:4310`); on miss, discard (`generation_too_slow`) and **skip** — panel self-refreshes; never stall. |
| 5 | Bad interval estimate → generated flip too close to a real flip (visible double-image / stutter) | Med | Estimate `I_est` from measured real vblank deltas in `page_flip_handler`; enforce a min-gap floor (skip if `I_est/2` < composite+flip budget). |
| 6 | Timer jitter (scheduler latency) makes the midpoint imprecise | Low | `CLOCK_MONOTONIC` abs-time timerfd is ±sub-ms on a lightly loaded RT-niced compositor; midpoint precision is not safety-critical (worst case slightly uneven cadence, not a stall). |
| 7 | Low-priority framegen compute queue starves under GPU pressure | Med | Existing degrade: if `hasCompleted` is false at deadline, skip (risk 4). Feature naturally sheds generated frames first. |
| 8 | Extra flips increase power draw on battery | Low | Gate hybrid behind CLI flag; allow disabling on battery via existing power-state hooks. |
| 9 | Panel drops out of VRR range because generated flip pushes effective rate too high | Low | Cap presented rate at `g_nOutputRefresh`; never arm midpoint if real fps already near cap (I_est small → floor check skips). |
| 10 | Generated flip presented during a modeset / mode change | Low | Disarm timer + discard on resize/focus/mode change via `vulkan_framegen_invalidate_history` sites (`:4268`, hook H). |

---

## Testing & validation plan

**Counters / logs to add** (gate behind `g_bFramegenDebug`, cf. existing framegen logging at
`src/rendervulkan.cpp:4327`, `src/steamcompmgr.cpp:9146-9154`):

- `framegen.mid.armed` / `mid.fired` / `mid.skipped_no_frame` / `mid.skipped_in_flight` /
  `mid.skipped_too_slow` / `mid.cancelled_by_real` — one counter per outcome of the midpoint timer.
- `framegen.interval_est_ms` — rolling `I_est` and the actual measured real-flip delta, to validate
  the estimator.
- `framegen.gap_ms` — measured time between consecutive presents (real→generated, generated→real);
  the histogram should cluster near `I_est/2`.
- Per-flip slot tag extended: current log emits `slot=real|generated|repeat` (`:9151-9154`); add
  `sched=vblank|midtimer` to distinguish scheduling source.

**What to measure:**

1. **Real-frame latency parity.** Compare click-to-photon / present latency of framegen-off VRR vs
   hybrid VRR at matched fps. Requirement: statistically indistinguishable (this is the core claim).
   Use the existing MangoApp output-update timestamps (`mangoapp_output_update`,
   `DRMBackend.cpp:787`) and `gpuvis` traces (`flip commit` / `page_flip_handler`, `:4088`, `:773`).
2. **No spurious modesets.** Assert `needs_modeset` is never raised by an `allowVRR` mismatch while
   hybrid is active. Fail the test if `VRR_ENABLED` value changes between consecutive commits.
3. **Cadence.** Present-gap histogram bimodal→unimodal check: gaps should be ~`I_est/2`, not
   alternating long/short (would indicate a scheduling bug).
4. **Preemption correctness.** Fuzz real-frame arrival phase relative to `T_mid`; verify a real
   frame arriving just before the timer always wins and the generated flip is dropped, never
   delaying the real flip.
5. **LFC boundary.** Drive fps down toward the panel's VRR floor; confirm generated flips keep the
   effective present interval inside the VRR window (read back panel min/max via EDID / connector
   props) and that we clamp before crossing the LFC threshold.
6. **Miss-under-load.** Artificially throttle the compute queue; verify graceful skip (counter
   `mid.skipped_too_slow` rises, no stalls, real frames unaffected).
7. **Correctness of pixels:** with gpuvis + a high-speed capture, confirm generated flips carry the
   extrapolated buffer and never a torn/partial buffer (fence discipline, `:4323`).

---

## Incremental rollout / flags

- **`--framegen-vrr-hybrid` / `cv_framegen_vrr_hybrid`** (default **off**): master switch for this
  mode. When off, framegen keeps today's fixed-slot behavior (VRR suppressed at
  `src/steamcompmgr.cpp:2570`).
- Activation predicate (all must hold): `vulkan_framegen_is_enabled()` **and**
  `cv_framegen_vrr_hybrid` **and** `cv_adaptive_sync` **and** connector `IsVRRActive()`
  (`DRMBackend.cpp:462`). If VRR is not actually active, transparently fall back to fixed-slot
  framegen (or plain VRR if framegen also unsupported).
- **`cv_framegen_mid_fraction`** (default `0.5`): the temporal fraction for the midpoint; exposed
  for tuning and forward-compat with dynamic per-slot fractions (Tier-2). x3/x4 multipliers arm
  multiple timers at `k·I_est/N`.
- **`cv_framegen_mid_min_budget_ms`** (default `2.0`): floor below which we skip generation for the
  interval (risk 5/8).
- Fallback ladder: hybrid → fixed-slot framegen → plain VRR → fixed-refresh. Each downgrade is a
  runtime decision made from the activation predicate + deadline-miss counters, with no restart.
- Roll out behind the existing experimental framegen gating; keep `g_bFramegenDebug` logging on in
  the experimental builds so the counters above are visible in the field.

---

## Open questions

1. **Interval source under VRR.** The vblank estimator's VRR branch
   (`src/vblankmanager.cpp:157-177`) intentionally does not model a real cadence. Is a short rolling
   average of real-flip deltas in `page_flip_handler` accurate enough, or do we need a dedicated
   real-frame-interval estimator (EMA + spike clamp like the non-VRR redzone logic at `:127-145`)?
2. **Should the midpoint be dynamic per phase?** If the game's frame time is bimodal (e.g. 60/72
   alternating), a fixed 0.5 fraction can place the generated flip poorly. Worth a phase-aware
   fraction?
3. **Multi-flip (x3/x4) under VRR:** arming N−1 timers per interval multiplies the in-flight-guard
   and cancel bookkeeping. Is a single mid-flip the sane ceiling for VRR, with x3/x4 reserved for
   fixed-refresh mode?
4. **Interaction with `DRM_MODE_PAGE_FLIP_ASYNC` force-async paths** (`g_bForceAsyncFlips`,
   `:3009`): must be hard-disabled under hybrid; confirm no other code path re-enables async while
   framegen is active.
5. **Panel-specific LFC thresholds:** do we have reliable min/max VRR range from EDID for all
   target panels, or do we need a conservative hard-coded floor?
6. **Cursor plane vs generated flip:** if the hardware cursor moves between the real flip and the
   midpoint, the generated flip re-commits the cursor plane too — is there any risk of cursor
   flicker, and should the generated flip leave the cursor plane untouched?
