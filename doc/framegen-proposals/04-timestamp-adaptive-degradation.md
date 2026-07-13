# Proposal 4 — Timestamp-Driven Adaptive Degradation Ladder

Status: **implemented with deliberate control-policy divergences**. The original design below is
retained as rationale. Production uses a monotonic down-only ladder, 85% vblank budget,
per-`(rung, generated-count)` 7/8 EMA, a three-completed-sample warm-up, and a four-frame post-step
hold. The dedicated-queue semaphore wait covers `ALL_COMMANDS`, keeping the opening timestamp after
the composite dependency; deltas are modular in `timestampValidBits`, so sub-64-bit counter wrap is
valid. Single-queue devices key the same non-blocking query readback to the scratch timeline, so the
ladder remains active there; only generation admission is deliberately more conservative.
Scope: `src/rendervulkan.cpp` framegen core, `CVulkanDevice`/`CVulkanCmdBuffer`,
`src/steamcompmgr.cpp` slot scheduler, `src/vblankmanager.cpp` timing source.
Depends on: the existing experimental framegen path (`--experimental-framegen`)
and, for the upper ladder rungs, the Tier-1/2 work (motion mode, x3/x4
multiplier, dedicated low-priority compute queue).

---

## Motivation / problem it solves

Framegen degrades **reactively** today. Two mechanisms exist, both *after the
fact*:

1. **Oversubscription peek** — `framegen_record_real_frame`
   (`src/rendervulkan.cpp:4494`) calls `g_device.hasCompleted(
   g_framegenHistory.generatedSeqNo )`. If the *previous* generation has not
   finished by the time the *next* real frame arrives, it invalidates history
   and re-enters the stabilization window (`k_uFramegenStableFramesRequired = 8`,
   line 4244). This is coarse: it only fires when a generation overruns an
   entire inter-real-frame gap, and its only response is the nuclear one —
   go dormant for 8 frames.

2. **Late-skip at present** — `vulkan_framegen_consume_generated_frame`
   (`src/rendervulkan.cpp:4310`) peeks completion again and, if the generated
   frame is not ready, discards it (`"generation_too_slow"`). The vblank shows a
   hardware repeat. The GPU time was already spent; the slot is already lost.

Neither path *knows how long generation actually takes*. They infer overrun
from a boolean "did it finish yet". Consequences:

- **No graceful middle.** The system is binary: full-quality generation, or
  dormant. There is no "the compositing GPU is 80% loaded, drop from motion
  mode to extrapolate" step. On an iGPU sharing memory bandwidth with the
  compositor, or a second GPU that is itself lightly used for other work, this
  bang-bang control produces visible flapping: 8 good generated frames, one
  missed vblank, 8 dormant frames, repeat — read as periodic microstutter (the
  exact failure the hysteresis comment at line 4240 acknowledges but only
  partially tames).

- **Wasted work.** A generation that will miss is still dispatched, still
  consumes bandwidth and queue occupancy, and is then discarded. Under the
  critical rule (real frames never delayed), that wasted compute still competes
  for the compositing GPU and can push the *next* composite's start.

- **No headroom signal.** We cannot pick a richer algorithm (motion-compensated
  warp, deeper pyramid, x4 multiplier) when there is slack, because we have no
  measurement of how much slack there is.

The upgrade: **measure actual generation GPU time with timestamp queries, and
preemptively step quality/rate down (or up) before deadlines are missed.**
Replace bang-bang with a ladder + a control loop driven by measured cost vs.
the known vblank budget.

---

## Design overview

Add a **cost estimator** and a **degradation ladder** in front of the existing
record/dispatch path.

- Every generation command buffer brackets its dispatch(es) with
  `vkCmdWriteTimestamp`. The delta × `timestampPeriod` gives GPU-time-on-device
  for that generation.
- We **never read the timestamp of the generation we just submitted** (that
  would stall). We read the *prior* frame's result from a double-buffered query
  pool — by the time the next real frame arrives the prior generation has
  essentially always completed (the oversubscription gate at line 4494
  guarantees it, or drops us out).
- Measured GPU times feed an **EWMA**. Each real frame, the control loop
  compares `EWMA(cost)` against the **available budget** (a safety fraction of
  the vblank interval derived from `g_nOutputRefresh`) and selects a **ladder
  rung**.
- The ladder, highest cost → lowest:

  | Rung | Description | Approx cost driver |
  |------|-------------|--------------------|
  | 5 | Motion-compensated mode, full pyramid depth, x4 | pyramid build + block match + warp, 3 extra generated frames |
  | 4 | Motion mode, full pyramid, x3 | as above, 2 generated frames |
  | 3 | Motion mode, reduced pyramid depth, x2 | shallower pyramid + warp, 1 generated frame |
  | 2 | Extrapolate mode (`cs_framegen_extrapolate.comp`), x2 | single dispatch, neighborhood rectification |
  | 1 | Extrapolate mode, x2, low strength / no rectification refinement | cheapest generation |
  | 0 | **Dormant** — no history copy, no dispatch | zero |

  (Rungs 3–5 are gated on the Tier-1/2 motion mode and multiplier work landing;
  until then the ladder collapses to {2, 1, 0} and is still useful for the
  flap problem.)

- **Hysteresis** is asymmetric and matches the existing philosophy (line 4240):
  stepping **down** is fast (protect the real frame / avoid the miss), stepping
  **up** is slow (require sustained headroom over several frames) so we do not
  oscillate around a rung boundary.

- The ladder **coordinates with**, not replaces, the existing dormancy/skip
  logic. Rung 0 *is* the existing dormancy. The oversubscription peek and the
  late-skip stay as hard safety nets — the ladder's job is to make them fire
  rarely.

The estimator adds one query-pool write pair per generation and one non-blocking
`vkGetQueryPoolResults` per real frame. No new synchronization on the real-frame
critical path.

---

## Vulkan mechanisms & extensions

### Timestamp queries

- **`VkQueryPool` with `VK_QUERY_TYPE_TIMESTAMP`.** Created once (per output
  resource lifetime) in the framegen resource path
  (`framegen_ensure_resources`, `src/rendervulkan.cpp:4373`), destroyed in
  `vulkan_framegen_reset` (line 4282).
  ```c
  VkQueryPoolCreateInfo qpci = {
      .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType  = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = k_uFramegenQuerySlots * k_uTimestampsPerGen, // 2 slots × 2..N
  };
  vk.CreateQueryPool(device(), &qpci, nullptr, &m_framegenTsPool);
  ```
  Two pool *slots* (double-buffered), each holding a begin/end pair (and, in
  motion mode, per-stage intermediate stamps: pyramid, match, warp — so the
  debug view can attribute cost). `k_uTimestampsPerGen` = 2 in extrapolate, up
  to 4 in motion mode.

- **`vkCmdResetQueryPool`** must run for a slot's queries **before** they are
  written, and cannot be inside a render pass (framegen is compute-only, so this
  is naturally satisfied). Reset the slot at the top of the generation command
  buffer *for the slot we are about to write*.

- **`vkCmdWriteTimestamp`** (core 1.0) or **`vkCmdWriteTimestamp2`** (core 1.3 /
  `VK_KHR_synchronization2`) around the dispatch(es) in
  `framegen_record_real_frame` (the dispatch is at line 4560):
  ```c
  vk.CmdResetQueryPool(cmd, pool, base, count);
  vk.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,          pool, base + 0);
  // ... bindPipeline / bindTarget / bindTexture / dispatch ...
  vk.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,       pool, base + 1);
  ```
  Use `BOTTOM_OF_PIPE`/`COMPUTE_SHADER_BIT` for the *end* stamp so it is written
  after the dispatch's shader stage completes, and `TOP_OF_PIPE` for the start.
  The delta measures device-side execution of the generation, *including* the
  history `copyImage` if we bracket before it — we deliberately place the start
  stamp **after** the `copyImage` (line 4518) so the measured quantity is the
  *generation* cost, not the copy (the copy is unavoidable history upkeep and is
  accounted separately if needed).

- **`VkPhysicalDeviceLimits::timestampPeriod`** (nanoseconds per timestamp tick)
  converts ticks → ns. Gamescope already reads `VkPhysicalDeviceProperties`
  during device selection (`src/rendervulkan.cpp:412`, `:502`); cache
  `props.limits.timestampPeriod` and `props.limits.timestampComputeAndGraphics`
  (or the per-queue-family `timestampValidBits`) into `CVulkanDevice` at init.
  **Guard:** if the chosen compute queue family has `timestampValidBits == 0`,
  disable the estimator and fall back to the existing reactive logic (the ladder
  degenerates to today's behavior). This satisfies constraint (3): no hard
  dependency.

- **`vkGetQueryPoolResults`** with `VK_QUERY_RESULT_64_BIT`. Read the **other**
  (prior) slot, **without** `VK_QUERY_RESULT_WAIT_BIT`, so the call returns
  `VK_NOT_READY` instead of blocking if — against expectation — the prior
  generation has not completed:
  ```c
  uint64_t stamps[2];
  VkResult r = vk.GetQueryPoolResults(device(), pool, priorBase, 2,
      sizeof(stamps), stamps, sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT); // NO WAIT bit
  if (r == VK_SUCCESS) {
      uint64_t ticks = stamps[1] - stamps[0];
      double genNs = (double)ticks * m_timestampPeriod;
      // mask to timestampValidBits before subtract if < 64
  }
  ```
  If `VK_NOT_READY`, we simply do not update the EWMA this frame and (optionally)
  treat it as a strong down-signal, because a not-ready prior generation is
  itself evidence of overrun.

### GPU↔CPU clock correlation

The vblank estimator and all gamescope timing use **`CLOCK_MONOTONIC`**
(`get_time_in_nanos`, `src/steamcompmgr.cpp:1291`; the kernel reports page flips
on `CLOCK_MONOTONIC`, and `CVBlankTimer::GetNextVBlank`,
`src/vblankmanager.cpp:83`, is built on it). The timestamp query gives a *device
timeline* value, not a `CLOCK_MONOTONIC` value.

For the **duration** measurement (end − start within one command buffer) no
correlation is needed — both stamps are on the same device timeline and
`timestampPeriod` converts the delta directly to nanoseconds. This is the
primary signal the ladder uses, and it needs no extension beyond core
timestamps.

Correlation is only needed if we want to answer "did generation *finish before
the target vblank wall-clock*", i.e. to measure **slack in monotonic time**
rather than pure duration. For that use:

- **`VK_EXT_calibrated_timestamps`**, `vkGetCalibratedTimestampsEXT` with
  `VK_TIME_DOMAIN_DEVICE_EXT` and `VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT`
  (query availability via `vkGetPhysicalDeviceCalibrateableTimeDomainsEXT`).
  A single calibrated read yields a `(device_ticks, monotonic_ns)` pair plus a
  `maxDeviation`. Cache this pair; convert any device timestamp `d` to monotonic:
  `mono ≈ mono0 + (d − device0) * timestampPeriod`. Re-calibrate periodically
  (e.g. once per second) to absorb clock drift; `maxDeviation` bounds the error
  we must add to the safety margin.
- This is an **opt-in accelerator**, not a requirement (constraint 3). Absent
  the extension, the ladder runs on duration-vs-budget alone, which is
  sufficient: the budget is derived from the vblank interval and we compare a
  *duration* against a *fraction of the interval*, no absolute-clock mapping
  required.

### Sync primitives touched

None added on the real-frame path. The generation command buffer is still
submitted via `CVulkanDevice::submit` / `submitInternal`
(`src/rendervulkan.cpp:1400`), signaling the existing timeline semaphore
`m_scratchTimelineSemaphore`. Completion peeks continue to use
`GetSemaphoreCounterValue` (`hasCompleted`, line 1623). The query pool is read
only after that semaphore has advanced past the prior generation's seq no, so
the results are guaranteed available with the non-waiting read in the common
case.

---

## Integration points in gamescope

All line numbers are current-tree anchors, not exact after edits.

1. **`CVulkanDevice` (`src/rendervulkan.hpp:778`)** — add members:
   `VkQueryPool m_framegenTsPool`, `double m_timestampPeriod`,
   `uint32_t m_timestampValidBits`, `bool m_bTimestampsUsable`. Populate
   `m_timestampPeriod`/valid-bits from the `VkPhysicalDeviceProperties` already
   fetched at `src/rendervulkan.cpp:412`/`:502`. Add thin methods:
   `framegenTimestampPool()`, `readFramegenGpuNanos(slot, out)` wrapping the
   non-waiting `vkGetQueryPoolResults`.

2. **`CVulkanCmdBuffer` (`src/rendervulkan.hpp:935`)** — add
   `resetTimestamps(pool, base, count)` and
   `writeTimestamp(pool, index, stage)` thin wrappers over
   `CmdResetQueryPool`/`CmdWriteTimestamp`, mirroring the style of existing
   `dispatch`/`copyImage`.

3. **Cost/ladder state** — extend `FramegenHistory_t`
   (`src/rendervulkan.cpp:4210`) with:
   `uint32_t nQuerySlot` (0/1 double-buffer index),
   `uint64_t generatedGpuNanosEwma` (fixed-point or double),
   `int nLadderRung`, `uint32_t nRungUpCredits` (hysteresis accumulator),
   `uint64_t nLastMeasuredGpuNanos`, and per-rung/per-stage debug counters.

4. **Control loop + measurement** — inside `framegen_record_real_frame`
   (`src/rendervulkan.cpp:4416`):
   - *Before* deciding to dispatch: read the **prior** slot's result
     (`readFramegenGpuNanos`), update the EWMA, run the control law, pick
     `nLadderRung` (see control law below). Rung 0 ⇒ take the existing dormant
     early-returns.
   - The existing gates stay: dormancy on fast frames (line 4479),
     oversubscription peek (line 4494), stabilization window (line 4504). The
     ladder decision slots **between** the oversubscription peek and the actual
     dispatch — i.e. after we know the GPU kept up, choose *how much* to do.
   - At dispatch time (line 4548–4560): choose shader/multiplier/pyramid params
     from the rung; bracket the dispatch(es) with reset + write-timestamp into
     the **current** slot; flip `nQuerySlot ^= 1` after submit (line 4565).

5. **Slot scheduler (`src/steamcompmgr.cpp:9131`)** — unchanged in mechanism.
   The x3/x4 rungs (once Tier-1/2 lands) mean multiple pending generated frames
   per real frame; the "generated vs real vs repeat" classification at line 9146
   gains a rung field for the `--framegen-debug` log.

6. **Late-skip (`vulkan_framegen_consume_generated_frame`, line 4299)** —
   unchanged as a safety net; add a counter increment on the
   `"generation_too_slow"` path so the validation plan can prove the ladder
   drives that counter toward zero.

7. **Budget source** — the vblank interval already computed at
   `src/rendervulkan.cpp:4478`
   (`1e12 / g_nOutputRefresh`, mHz→ns) is reused verbatim as the budget
   denominator. `g_nOutputRefresh` is the authoritative output refresh
   (`src/main.hpp:21`, set per-backend).

8. **CLI flags (`src/main.cpp:139`, `:815`)** — add ladder-related flags (see
   Rollout section) alongside the existing `--framegen-*` options.

---

## Data & control flow

Per **real frame** (new base-layer commit reaching
`framegen_record_real_frame`):

1. Enter `framegen_record_real_frame` (line 4416). Overlay-only / dormancy /
   gap / oversubscription gates run exactly as today (lines 4434–4499). If any
   fires, the frame is dormant — the ladder observes this as rung 0 and does not
   update the EWMA (there was no generation to measure).

2. **Measure (prior frame):** read query slot `nQuerySlot ^ 1` with a
   non-waiting `vkGetQueryPoolResults`. On `VK_SUCCESS`, compute
   `genNs = (end−start) * timestampPeriod`; update EWMA. On `VK_NOT_READY`
   (rare — implies the prior generation overran), skip the EWMA update and apply
   a one-step down bias.

3. **Decide rung:** compute `budgetNs = k_flFramegenBudgetFraction *
   vblankIntervalNs` (default fraction ≈ 0.6, see analysis). Compare
   `EWMA` to the current rung's cost and adjacent rungs' expected costs;
   apply the hysteresis law. Result: `nLadderRung`.

4. **Dispatch (this frame):** if rung > 0, select shader
   (`SHADER_TYPE_FRAMEGEN_EXTRAPOLATE`/`..._MOTION`), multiplier, pyramid depth,
   strength from the rung. Record into command buffer:
   `resetQueryPool(slot)` → `writeTimestamp(start, TOP_OF_PIPE)` → generation
   dispatch(es) → `writeTimestamp(end, COMPUTE_SHADER_BIT)`. Submit via
   `g_device.submit` (line 4565); store `generatedSeqNo`. Flip `nQuerySlot`.

5. **Present (later, on a vblank):** `steamcompmgr` slot scheduler (line 9131)
   presents the pending generated frame if the vblank is empty;
   `vulkan_framegen_consume_generated_frame` (line 4299) does the final
   non-blocking completion peek and either presents or late-skips.

**Why it stays off the real-frame critical path:**

- The measurement read is **non-blocking** and reads a **prior** slot; it never
  waits on the just-submitted generation.
- The generation itself already lives in its own command buffer, submitted
  *after* the composite (comment at line 4512). The added `CmdResetQueryPool` +
  two `CmdWriteTimestamp` calls execute inside that same trailing command
  buffer — they cost the real frame nothing.
- The control-law arithmetic is a handful of scalar ops on the CPU inside a
  function that already runs after composite.
- Picking a *cheaper* rung strictly *reduces* the compute the compositing GPU
  runs behind each composite, which can only *help* the next composite start on
  time. The ladder's whole purpose is to keep generation from ever queueing in
  front of a composite (the one way this design could delay a real frame,
  line 4491).

---

## Latency & throughput analysis

**Added GPU cost per generation:** `CmdResetQueryPool` + 2×`CmdWriteTimestamp`.
Timestamp writes are effectively free relative to a full-frame compute dispatch
(a pipeline marker, no bandwidth). Reset is a trivial control command. At
4K, one extrapolate dispatch already reads two RGBA16F history textures
(~2×32 MB = 64 MB read) and writes one (~32 MB) — the timestamp overhead is
sub-microsecond and unmeasurable against that.

**Added CPU cost per real frame:** one `vkGetQueryPoolResults` (a memcpy of two
`uint64_t` from a mapped/host-visible query result region on most drivers) plus
~10 scalar ops for EWMA + rung selection. Sub-microsecond. Runs post-composite.

**Cross-PCIe traffic:** none added. Timestamp results are device-local query
data read by the host over the existing control channel, kilobytes/sec. The
dual-GPU history copy is unchanged.

**Budget math (why 0.6 default):** at 60 Hz the vblank interval is 16.67 ms; the
generated frame must be *complete before its target vblank*. Because generation
starts only after the composite finishes and must land by the *next* vblank, the
usable window is less than a full interval. Empirically the composite itself and
present scheduling consume part of the interval, so a safety fraction of ~0.6
(≈10 ms at 60 Hz, ≈4.2 ms at 144 Hz) leaves margin for scheduler jitter and the
`calibrated_timestamps maxDeviation`. This fraction is a CLI-tunable
(`--framegen-budget-fraction`). Rungs are chosen so that
`EWMA(rung_cost) ≤ budget` with a hysteresis band; if even rung 1 exceeds
budget, the loop drops to rung 0 (dormant) — the same outcome as today's
oversubscription drop, but reached *before* a miss rather than after.

**Throughput of x3/x4 rungs:** each additional generated frame is another full
dispatch. The estimator measures *per-generation* cost; for x3/x4 the loop must
verify `N × EWMA(per_gen) ≤ (N) × slot_budget` where each generated frame has
its own vblank slot. Practically: only promote to x4 when a single generation
costs `< budget/1.5` so all three fit their slots with margin. The ladder
encodes this as the rung thresholds.

---

## Interaction with VRR / HDR / anti-lag / tearing

- **VRR:** framegen already forces adaptive sync off while active —
  `frameInfo.allowVRR = cv_adaptive_sync && !vulkan_framegen_is_enabled()`
  (`src/steamcompmgr.cpp:2570`), and generated presents set
  `allowVRR = false` (`src/Backends/DRMBackend.cpp:3619`). The ladder does **not**
  re-enable VRR; the budget is always computed against the fixed
  `g_nOutputRefresh` interval. Rung transitions do not change present timing, so
  they cannot perturb the fixed-refresh cadence framegen relies on.

- **HDR:** the budget/estimator are colorspace-agnostic — they measure
  wall-clock-equivalent GPU duration. HDR (fp16/PQ) history textures are larger
  and thus *more expensive* to generate over, which the EWMA captures
  automatically: on HDR content the loop will naturally sit a rung lower for the
  same GPU. The existing EOTF-change history invalidation (line 4450) still
  fires; a rung is preserved across it (cost characteristics do not reset just
  because history did).

- **Anti-lag / Reflex:** untouched. Generated frames still emit **no** Wayland
  commits and **no** client `frame_done`/presentation feedback (constraint 2);
  the ladder only changes *which* generation shader runs and *whether* it runs.
  It never introduces a client-visible event. Measurement is passive.

- **Tearing:** framegen suppresses async/tearing flips while active (log at
  line 4425). Generated presents go through the normal (non-async) commit. The
  ladder does not alter this.

---

## Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Driver reports `timestampValidBits == 0` on the compute queue | Estimator unusable | Detect at init; disable ladder, fall back to today's reactive logic. Log once. |
| Reading a query for a generation that hasn't completed | Stall if `WAIT_BIT` set | Never set `VK_QUERY_RESULT_WAIT_BIT`; read the **prior** slot; treat `VK_NOT_READY` as a down-signal. |
| Timestamp measures duration, not slack-to-vblank | Rung too optimistic near deadline | Conservative budget fraction (0.6) + optional `calibrated_timestamps` for true monotonic slack. |
| Query pool not reset before reuse | UB / garbage results | `CmdResetQueryPool` at top of every generation cmdbuf for the slot about to be written; double-buffer so we never reset a slot still being read. |
| EWMA lag after a scene-complexity spike | One overrun before adapting | Asymmetric hysteresis: instant single-step *down* on `VK_NOT_READY` or on `EWMA > budget`; the late-skip net (line 4310) still catches the stray miss. |
| Rung flapping around a boundary | Microstutter | Up-transitions require `nRungUpCredits` sustained-headroom frames; down is immediate. Mirrors existing `k_uFramegenStableFramesRequired` philosophy. |
| Query pool leaked across resize/format change | VkQueryPool leak | Destroy in `vulkan_framegen_reset` (line 4282); recreate in `framegen_ensure_resources` (line 4373) alongside textures. |
| Motion-mode per-stage stamps inflate query count | Pool sizing | Size pool for worst case (`k_uTimestampsPerGen` = 4) × 2 slots; extrapolate only writes 2, leaves rest reset-but-unwritten (skipped in read). |
| `calibrated_timestamps` drift between recalibrations | Slack miscomputed | Recalibrate ~1 Hz; add `maxDeviation` to safety margin; feature is opt-in anyway. |
| Ladder promotes to x4 on a GPU that then gets a background load spike | Multiple wasted generations | Per-slot budget check for x3/x4 (`N×per_gen ≤ N×slot_budget`) with margin; down-step drops the multiplier first (rungs 5→4→3). |

---

## Testing & validation plan

**Counters to add (exposed under `--framegen-debug`):**

- `genGpuNanosLast`, `genGpuNanosEwma`, `budgetNanos`, `budgetFraction`.
- `ladderRung` (current), and a histogram of time-spent-per-rung.
- `rungStepDowns`, `rungStepUps`, and the reason for the last down-step
  (`ewma_over_budget` / `query_not_ready` / `oversubscribed`).
- `lateSkips` (increment at `vulkan_framegen_consume_generated_frame`
  `"generation_too_slow"`, line 4312) and `oversubscribeDrops` (increment at
  line 4497). **Success criterion: with the ladder on, both trend toward zero**
  while `framesGenerated` stays high — i.e. we preempt instead of miss.
- Per-stage motion-mode stamps (`pyramidNanos`, `matchNanos`, `warpNanos`) when
  rung ≥ 3, to attribute cost.
- Optional: `calibratedDeviationNanos` from
  `vkGetCalibratedTimestampsEXT`.

**Log lines (under `--framegen-debug`):**
- On every rung change:
  `framegen: ladder rung N->M ewma=%.2fms budget=%.2fms reason=...`.
- Extend the existing slot classification log (line 9146) with the rung:
  `framegen: vblank slot=generated rung=N`.
- Periodic (once/sec) summary: rung occupancy %, late-skip rate,
  measured-vs-budget ratio.

**What to measure / how to prove correctness:**

1. **Estimator accuracy.** Cross-check `genGpuNanosLast` against an external
   profiler (RGP / GpuVis / `VK_EXT_calibrated_timestamps` correlated to a
   CPU-side `CLOCK_MONOTONIC` bracket around submit→complete). They should agree
   within timestamp granularity.
2. **Preemption vs. reaction.** Synthetic load: run a game/benchmark, then apply
   an escalating GPU load on the compositing device (a second compute stress).
   With the ladder **off**, expect a burst of `lateSkips`/`oversubscribeDrops`
   and 8-frame dormant re-entries (flapping). With it **on**, expect the rung to
   step down *before* misses; `lateSkips ≈ 0`; smooth rung occupancy shift.
3. **No real-frame regression.** Measure real-frame present latency (existing
   present timing / GpuVis) with framegen+ladder vs. framegen-only vs. off. The
   ladder must not increase real-frame latency (it only changes trailing
   generation cost). Assert composite start time is unaffected.
4. **Hysteresis.** Drive the GPU to sit exactly on a rung boundary; confirm
   `rungStepUps + rungStepDowns` stays low (no flapping) via the counter.
5. **Fallback.** Force `m_bTimestampsUsable = false`; confirm the ladder
   degrades to today's behavior and nothing regresses.
6. **Correctness of double-buffering.** Validation-layers-clean run: no
   "query reset" / "results not available" warnings; confirm we never read the
   in-flight slot.

---

## Incremental rollout / flags

Gate everything behind the existing `--experimental-framegen`. New flags
(add to `src/main.cpp:139` option table and `:815` parser, defaults in
`src/main.cpp`):

- `--framegen-adaptive` (bool, **default on** when framegen is enabled) —
  master switch for the ladder + estimator. Off ⇒ exactly today's reactive
  behavior.
- `--framegen-budget-fraction 0.6` (float 0.3–0.95) — safety fraction of the
  vblank interval used as the generation budget.
- `--framegen-max-rung N` — cap the ladder (e.g. pin to extrapolate-only until
  motion mode is trusted; `N=2` reproduces the pre-Tier-2 world).
- `--framegen-min-rung N` — floor (debug: force a rung to profile its cost).
- `--framegen-calibrated-timestamps` (bool, default auto) — enable
  `VK_EXT_calibrated_timestamps` slack correlation when available; auto = use if
  the extension and both time domains are present.

**Rollout stages:**

1. Land estimator + counters **observation-only** (`--framegen-adaptive` present
   but ladder pinned to current mode). Ship the timestamp plumbing and validate
   accuracy (test 1) with zero behavior change.
2. Enable the {2,1,0} ladder (extrapolate strength / dormant) — solves the
   flapping problem with no dependency on Tier-2. Validate tests 2–4.
3. Once motion mode + x3/x4 + dedicated compute queue land, extend the ladder to
   rungs 3–5, raise the default `--framegen-max-rung`.

**Fallback path (constraint 4 — degrade gracefully):** the ladder *is* the
graceful-degradation mechanism. Its terminal rung (0) is the existing dormancy,
and the existing oversubscription drop (line 4494) + late-skip (line 4310)
remain as hard nets underneath it. If timestamps are unavailable, or any query
read fails unexpectedly, the estimator disables itself and the system reverts to
the current reactive behavior with no loss of the safety guarantees.

---

## Open questions

1. **Which stage flag for the end timestamp** across vendors — `BOTTOM_OF_PIPE`
   vs `COMPUTE_SHADER_BIT`. `BOTTOM_OF_PIPE` is the safe "everything before is
   done" choice; verify it does not over-count trailing barrier work on the
   dedicated compute queue.
2. **Budget denominator for x3/x4:** should each generated slot get an equal
   `1/N` share of the inter-real-frame gap, or should we budget against the
   fixed vblank interval regardless of `N`? The fixed-interval view is simpler
   and matches the "generated frames fill fixed vblank slots" model
   (line 2566–2570), but under-utilizes headroom when the game runs at exactly
   `refresh/N`.
3. **EWMA time constant.** Too fast → chases transient spikes (flap); too slow →
   late to adapt to a genuine complexity change. Candidate: α ≈ 0.2 per real
   frame, but this should be tuned per-refresh (fewer samples/sec at 60 Hz than
   144 Hz argues for a time-based rather than frame-based constant).
4. **Should a `VK_NOT_READY` prior-slot read hard-drop to rung 0** (like the
   current oversubscription behavior) or just step down one rung? Stepping down
   one rung preserves *some* generation; hard-drop is safer for the real frame.
   Likely: step down one rung *and* let the existing oversubscription gate
   (line 4494) still hard-drop on its own criterion.
5. **Cross-GPU timestamp domains.** In the dual-GPU design the compositing GPU
   owns generation; its timestamps are self-consistent. But if calibrated
   timestamps are only exposed on one device, is the monotonic correlation still
   valid for the compositing device we actually run generation on? (It should
   be, since generation and its queries are on the *same* device — confirm.)
6. **Interaction with the history `copyImage` cost.** We measure generation
   excluding the copy, but the copy competes for the same GPU. Should the budget
   subtract an estimate of the copy, or should we bracket the copy separately and
   include it in the oversubscription reasoning?
