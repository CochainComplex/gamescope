# Compositor-Side Frame Generation: Architecture

The engineer's map of gamescope's compositor-side frame generation: entry points,
end-to-end dataflow, the algorithmic branches, the decision state machine, the
invariants and the resource design. Source locations are identified by symbol and
file, never by line number: the hot path moves frequently, while the named
contracts remain reviewable. The numbered documents in `doc/framegen-proposals/`
are historical design records; this document and the implementation are
authoritative when a proposal describes an older shape.

Audience: developers changing or reviewing the frame-generation code.
Related: [Documentation index](README.md) · [User guide](framegen-howto.md) · [Configuration reference](framegen-reference.md) · **Architecture** (this page) · [Maintenance](framegen-maintenance.md) · [Testing](framegen-testing.md) · [Proposals](framegen-proposals/README.md) · [Research notes](research-framegen.md) · [Scripts](../scripts/README.md)

## Reading order: the 30-second model

> Statements marked **(measured)** are dated observations from one rig, not guarantees. The numbers are in [Appendix A](#appendix-a-measurements-dated-non-contractual). Known problems and unvalidated features are listed in [Section 10](#10-sharp-edges--wip--open-bugs); read it before you start debugging.

### What the code does, in one paragraph

A game renders **real frames** at its own pace. The display refreshes at a fixed rate, for example 120 Hz. Whenever a refresh would otherwise repeat the old image, gamescope can show a **generated frame** instead. It builds that frame from the real frames it already has. In the default mode it predicts forward from the last two real frames, so a real frame is never held back. In the opt-in **bidir** mode it interpolates between two real frames, which looks better but shows every real frame late.

### The picture

```text
 render GPU (the game)                         present GPU (display-connected)
 ---------------------                         ------------------------------------------------
 game --Vulkan--> [FROG WSI layer] --dma-buf--> import
                  (forces MAILBOX)                |
                                                  v
                                   vulkan_composite  -> real frame N (composited, full screen)
                                                  |
                                                  v
                          history: previousReal <- currentReal <- frame N     (ring slots, no copy)
                                                  |
                                                  v   framegen queue (second queue if the GPU has one)
                          luma pyramid -> block match -> FB check -> (B4 stats / learned net) -> warp
                                                  |
                                                  v
                                       generated frame (one "slot")
                                                  |
                                                  v
              arbiter (steamcompmgr), once per vblank:  real  |  generated  |  hardware repeat
                                                  |
                                                  v
                                   DRM atomic commit  /  nested Wayland present
```

The block-match stage and everything after it are the **motion** mode. The simpler `extrapolate` mode skips the matching and runs one cheap shader instead (3.1).

### Vocabulary

| Term | Meaning | Details |
|---|---|---|
| WSI layer (FROG) | A Vulkan layer inside the game's process. It makes the game hand finished frames to gamescope as dma-bufs. Source: `layer/VkLayer_FROG_gamescope_wsi.cpp`. | 2 |
| MAILBOX | The Vulkan present mode the layer forces: the newest finished frame replaces an older one that was not yet shown, so the game never blocks. | 2 |
| dma-buf | A GPU buffer that can be shared between processes and between GPUs. | 2, 2.1 |
| Render GPU | The GPU the game renders on. The present GPU is the one wired to the display. | 2.1 |
| Composite | Drawing the game picture plus overlays into the final output image (`vulkan_composite`). | 2 |
| Arbiter | The code in `src/steamcompmgr.cpp` (steamcompmgr is gamescope's compositor manager) that picks real, generated or repeat for each vblank. | 2, 4.1 |
| vblank | One display refresh tick. | 4.1.1 |
| KMS | Linux kernel mode setting: the DRM interface the native backend uses to put an image on the screen (atomic commit). | 2 |
| Real frame | A frame the game rendered. It takes priority over any generated frame, and is held back only under bidir. | 2 |
| Generated frame | A frame gamescope made for a refresh in which no new real frame arrived. | 3 |
| Slot | One display refresh opportunity. Each slot shows exactly one of: a real frame, a generated frame, or a hardware repeat of the last image. | 2, 4.1.1 |
| Causal | Uses only frames that already exist. It predicts (extrapolates) forward from the last two real frames and adds no hold-back latency (only the forced composite, about 2 ms, see 2 step 3). This is the default. | 3.1, 4.1.1 |
| Bidir | Bidirectional: interpolates between two real frames, so every real frame is shown late by a target delay D. Opt-in (`GAMESCOPE_FRAMEGEN_BIDIR=1`). | 3.3 |
| History | `previousReal` and `currentReal`: the two newest real frames, held as slots of the composite output ring without copying them. | 6 |
| Motion field | A grid of motion vectors, one per 8x8 pixel block (the "/8" field), giving each block's displacement over one real-frame interval. The opt-in finer field is /4. | 3.2 |
| SAD | Sum of absolute differences, the cost the block matcher minimizes to find where a block moved. | 3.2 |
| FB check (B2) | Forward-backward check. The match is repeated in reverse; a vector that does not round-trip is unreliable. | 3.2 |
| Killed | A motion vector whose confidence was set to zero (rejected). The warp then falls back to holding the real frame. | 3.2 |
| B3 | The bidir interpolation mode. | 3.3 |
| B4 | Self-supervised adaptation: each real frame grades the field that was predicted for it, and the thresholds follow. | 3.4 |
| Stage C | The learned refiner: a tiny net that corrects the motion field and its confidence. | 3.5 |
| C2 | In-situ training: the net keeps learning on the framegen GPU while you play (`GAMESCOPE_FRAMEGEN_NET_ONLINE=1`). | 3.5 |
| E2 | Held-out colour capture used to grade image quality offline (`GAMESCOPE_FRAMEGEN_RECORD_COLOR`). | 8 |
| GSFD, GSCF, GSFR | Magic numbers of the file formats: GSFD = dataset capture for training, GSCF = E2 colour capture, GSFR = learned-net weights and per-game profile. | 3.5, 8 |
| TAA clamp | Clamping a predicted pixel to the range of its neighbours in the source frame, as temporal anti-aliasing does. | 3.1, 3.2 |
| LDS | Local data share: the fast per-workgroup memory of a GPU. | 3.1 |
| VFI | Video frame interpolation, the research family bidir belongs to. | 3.6 |
| Scratch timeline | The timeline semaphore of the normal composite queue. The generation work waits on it. | 2 |
| Dormant, prime | Dormant: nothing is generated because the game already fills every refresh. Prime: the first real frame after a scene change only fills the history. | 4.1, 4.5 |
| Rung, ladder | The degradation ladder: cheaper generation steps taken when the GPU cost would miss the deadline. | 4.4 |

The [configuration reference](framegen-reference.md#glossary) has a shorter glossary for players.

### What runs if you change nothing

| Setting | Default | Opt-in | Experimental or debug |
|---|---|---|---|
| Mode | `extrapolate` | `motion` | `blend` (debug only) |
| Pipeline (motion mode only) | `warp` | `checked`, `learned`, `predict`, `guided` | none |
| Scheduler | causal deadline scheduler (dedicated queue) | none | `GAMESCOPE_FRAMEGEN_CLASSIC=1` and `GAMESCOPE_FRAMEGEN_SINGLE_QUEUE=1` select the older fallbacks (4.2, 4.3) |
| Interpolation | off (causal) | `GAMESCOPE_FRAMEGEN_BIDIR=1` | none |
| Base-layer generation | off | `GAMESCOPE_FRAMEGEN_BASE=1` | prototype, validated nested on the dual-GPU split only |
| VRR hybrid | off | `GAMESCOPE_FRAMEGEN_VRR_HYBRID=1` | prototype, **not validated** on a VRR panel |
| Learned net | off | `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` (learns in game) or `GAMESCOPE_FRAMEGEN_NET=<blob>` | none |
| Finer /4 field | off | `GAMESCOPE_FRAMEGEN_FIELD_FINE=1` | none |
| Sharp fallback | on | `=0` restores the crossfade | none |
| B4 adaptation, FB check, agreement test | on | `=0` disables | none |
| Ladder recovery | on | `GAMESCOPE_FRAMEGEN_RECOVER=0` disables | no effect with bidir or VRR hybrid |

The command-line default pipeline is `warp`. The player-facing recommendation (`learned` with `GAMESCOPE_FRAMEGEN_NET_ONLINE=1`) is in [Choosing settings](../README.md#choosing-settings); it rests on one viewer and is not yet backed by blind tests on every content class (Section 10).

### Reading the code

`src/rendervulkan.cpp` is very large. Search it by symbol, not by scrolling. The path of one real frame is:

1. `vulkan_composite` composites the real frame.
2. `framegen_record_real_frame` runs the scheduler (called from the tail of `vulkan_composite`).
3. `framegen_submit_planned` records the generation work; `framegen_prepare_motion` runs once per real pair and `framegen_warp_slot` once per slot.
4. The arbiter in `src/steamcompmgr.cpp` picks real, generated or repeat for each vblank.
5. `vulkan_framegen_consume_generated_frame` hands the generated image to `CDRMBackend::Present` or `CWaylandConnector::Present`.

The headers in `src/framegen/` hold the stateless policy. One source map is kept in [framegen-maintenance.md](framegen-maintenance.md); Section 0 below summarizes it.

### Where to go next

- **Which control regime am I in?** Read [4.1](#41-the-structural-fact-two-control-regimes-selected-by-hasframegenqueue) first. It decides which scheduler runs.
- **Opt-in interpolation:** `GAMESCOPE_FRAMEGEN_BIDIR=1` is the bidirectional variant (3.3). It presents each real frame late by a target delay and so relaxes the NEVER-QUEUED, REAL-DISCARDS-PENDING and FORWARD-ONLY invariants of Section 0.
- **Build and run:** build and launch recipes are in the [user guide](framegen-howto.md); test entry points are in [framegen-testing.md](framegen-testing.md); every variable is in the [configuration reference](framegen-reference.md).
- **Something is wrong:** the symptom table at the end of [Section 8](#8-observability) maps log lines and metrics to mechanisms.

## Contents

- [0. Scope and the one organizing fact](#0-scope-and-the-one-organizing-fact)
- [1. Entry points](#1-entry-points-the-doors-in)
- [2. End-to-end dataflow](#2-end-to-end-dataflow-one-real-frame): [2.1 dual-GPU path](#21-dual-gpu-path)
- [3. Generation modes and pipelines](#3-generation-modes-and-pipelines-the-core): [3.1 matrix](#31-comparison-matrix), [3.2 motion mode](#32-motion-mode-detail), [3.3 bidirectional](#33-bidirectional-interpolation-b3-gamescope_framegen_bidir1-opt-in) (with [delay controller](#bidirectional-delay-controller)), [3.4 adaptation](#34-self-supervised-online-adaptation-b4-default-on-in-motion-mode), [3.5 learned refinement](#35-learned-field-refinement-stage-c-opt-in-via-gamescope_framegen_netblob), [3.6 prior art](#36-prior-art--how-the-algorithm-maps-to-the-literature)
- [4. The decision state machine](#4-the-decision-state-machine): [4.1.1 default causal fixed-cadence behavior](#411-default-causal-fixed-cadence-behavior)
- [5. Latency and correctness invariants](#5-latency--correctness-invariants-with-enforcement-points)
- [6. Resource architecture](#6-resource-architecture)
- [7. Vendor and performance model](#7-vendor--perf-model)
- [8. Observability](#8-observability): [debugging by symptom](#debugging-by-symptom)
- [9. Roadmap](#9-roadmap-docframegen-proposals)
- [10. Sharp edges and open bugs](#10-sharp-edges--wip--open-bugs)
- [Appendix A. Measurements](#appendix-a-measurements-dated-non-contractual)

---

## 0. Scope and the one organizing fact

`src/rendervulkan.cpp` owns scheduling state, resource lifetime, motion preparation,
generation, and online learning. The headers under `src/framegen/` hold stateless policy
(cadence confidence and deadline constants in `scheduling.hpp`, the causal slot planner, the bidir
planner and delay controller in `deadline.hpp`, B4 counter decoding in `adaptation.hpp`); they are inlined into that owner and hold
no Vulkan resources. The module map below is a summary; the maintained source map is [Source map in framegen-maintenance.md](framegen-maintenance.md#source-map).

| File in `src/framegen/` | Responsibility |
|---|---|
| `scheduling.hpp` | cadence confidence, deadline constants, ladder thresholds and recovery constants, VRR-hybrid keep-up policy |
| `deadline.hpp` | causal slot planner (`plan_next_causal_slot`, `CausalSlotPlan_t`), bidir delay controller (`bidir_target_delay`, `BidirWindow_t`, `BidirRingBound_t`) and `plan_bidir_step` |
| `policy.hpp` | stateless ladder rung transformation (`max_degrade_steps`, effective config) and the motion-field scale (`motion_field_scale`, `motion_coarse_levels`, `motion_reference_texels`) |
| `adaptation.hpp` | B4 probe counter decoding, threshold policy and the quality-meter counts (`MotionQualityCounts`) |
| `temporal.hpp` | interval, phase, strength and VRR equivalent-gap arithmetic |
| `settings.hpp` | strict env/flag parsing (`parse_bool_setting`, numeric parsers) |
| `dispatch_policy.hpp` | vendor/capability selection among bit-identical shader variants |
| `net_layout.hpp`, `net_profile.hpp` | learned-net parameter layout and the persisted profile format |
| `push_constants.hpp` | push-constant structs mirrored against the GLSL contract |
| `metrics.hpp` | `GAMESCOPE_FRAMEGEN_METRICS` accounting |
| `hud.hpp` | built-in status HUD text and layout |
| `query_ring.hpp` | GPU timestamp query ring bookkeeping |
| `numeric.hpp`, `types.hpp` | shared numeric helpers and enums/value types |
| `device.cpp` | `CVulkanDevice` framegen queue, completion, command-buffer retirement, timestamps |
| `atomic_file.cpp/.hpp` | unique staging-file, checked close and atomic rename for captures and profiles |

`src/framegen/device.cpp` contains the `CVulkanDevice`
framegen queue, completion, command-buffer retirement, and GPU timestamp methods
without owning separate state. `atomic_file.cpp` owns only the staging mechanism; the
lifetime and worker synchronization of captures and learned profiles remain in the renderer.
The `cs_framegen_*.comp` shaders are the
algorithmic surface. `src/steamcompmgr.cpp` is the present arbiter; the
DRM/Wayland backends and FROG WSI layer carry dual-GPU and present-timing
plumbing. `doc/framegen-proposals/` records the roadmap and the rationale behind
implemented or rejected prototypes.

**Target topology (drives every design choice, see [2.1](#21-dual-gpu-path)):** the game renders on a strong GPU,
the completed frame crosses through dma-buf, and the display-connected GPU
composites, generates, and presents. Generation bandwidth therefore comes from
the present GPU. The load-bearing performance truth is that these passes are
mostly memory-bandwidth-bound; nearly every optimization below reduces history
traffic or bounds work on that device. The implementation also retains a
single/shared-queue fallback. Correctness does not depend on GPU vendor; measured
dispatch selection may choose a bit-identical implementation variant by vendor.

**Four invariants the whole design upholds.** They describe the default forward-extrapolation
path. The opt-in **bidir** mode (`GAMESCOPE_FRAMEGEN_BIDIR=1`, §3.3) deliberately relaxes NEVER-QUEUED
and FORWARD-ONLY. Other sections cite them by name. Section 5 shows where each one is enforced;
[framegen-maintenance.md](framegen-maintenance.md) lists the rules a change must keep.

1. **NEVER-QUEUED.** Real frames are **never queued behind** generated work. *(On native KMS a real frame that arrives
   after a generated atomic commit was submitted, but before its page-flip lands, is presented at the
   next vblank. Bidir is the larger exception: it presents each real
   frame late by its target delay D (see [Bidirectional delay controller](#bidirectional-delay-controller)) — the intrinsic cost of interpolating
   between two reals, hence opt-in.)*
2. **NEVER-LATE-GENERATED.** Generated frames are **never** waited on or presented late.
3. **REAL-DISCARDS-PENDING.** A real frame latching **always** discards any pending generated frame (latency safety).
   *(Bidir preserves queued real frames instead — its pending queue is a presentation timeline, §3.3.)*
4. **FORWARD-ONLY.** Generation is **forward extrapolation** (predict the future beyond the current real frame,
   no hold-back latency, only the forced composite of about 2 ms, see 2 step 3) — *not* interpolation. (The debug-only `blend` mode and the opt-in **bidir**
   mode are the exceptions — bidir interpolates *between* the two reals and so adds latency, bounded by its target delay D.)

---

## 1. Entry points (the doors in)

### CLI flags (`src/main.cpp`)
| Flag | Global set | Notes |
|---|---|---|
| `--experimental-framegen` | `g_bExperimentalFramegen` | master enable; forces the full-composite path |
| `--framegen-mode {extrapolate\|motion\|blend}` | `g_eFramegenMode` | default `extrapolate`; `blend` is debug-only |
| `--framegen-pipeline {warp\|checked\|learned\|predict\|guided}` | `g_eFramegenPipeline` | additive generation pipeline; default `warp`; in real-game tests `learned` with `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` felt best ([Choosing settings](../README.md#choosing-settings)) |
| `--framegen-multiplier {2,3,4}` | `g_nFramegenMultiplier` | validated, `exit(1)` on bad value; also *sizes the output pool* (forward causal mode: only sizes buffers, generated pool 2N and ring 4+2N; bidir: caps interpolations at N-1 per pair; see 4.1.1 and 3.3) |
| `--framegen-strength <0.0-1.0>` | `g_flFramegenStrength` | validated finite float forward step; malformed or out-of-range values fail startup |
| `--framegen-debug` | `g_bFramegenDebug`, `g_uFramegenDebugEvery` | see §8 observability |

Parsing and benchmark early-exit are in `src/main.cpp`.

### Environment variables
[framegen-reference.md](framegen-reference.md) is the authoritative list of values, defaults and ranges. The CLI flags are parsed in `src/main.cpp`; the environment switches are read through `framegen_bool_setting` and the related numeric helpers in `src/rendervulkan.cpp`.

Switches by group, each linked to its reference section. Everyday player-facing variables are `_BIDIR`, `_BASE`, `_NET_ONLINE`, `_NET_PROFILE`, `_NET_EVERY`, `_HUD`, `_METRICS` and `_RECOVER`; the Scheduler group and `_RECORD*` are for validation:

| Group | Variables | Reference |
|---|---|---|
| Scheduler | `_CLASSIC`, `_SINGLE_QUEUE`, `_COMMIT_LEAD_MS`, `_RECOVER`, `_BATCH_REPAINT` | [Developer and validation](framegen-reference.md#developer-and-validation) |
| Regimes | `_BIDIR`, `_BASE`, `_VRR_HYBRID` | [Environment variables](framegen-reference.md#environment-variables) |
| Quality | `_FB`, `_FB_TOL`, `_AGREE`, `_SHARP_FALLBACK`, `_FIELD_FINE`, `_ADAPT`, `_RESERVOIR`, `_SHADING` | [Environment variables](framegen-reference.md#environment-variables) |
| Learning | `_NET`, `_NET_ONLINE`, `_NET_LR`, `_NET_EVERY`, `_NET_PROFILE`, `_RECORD*` | [Environment variables](framegen-reference.md#environment-variables) |
| Diagnostics | `_DEBUG_EVERY`, `_METRICS`, `_HUD*`, `_BENCHMARK` | [HUD rows](framegen-reference.md#hud-rows), [Metrics log fields](framegen-reference.md#metrics-log-fields) |
| Legacy | `_JIT` (no effect, logs a notice) | none |

(All names start with `GAMESCOPE_FRAMEGEN`.) The table below maps each switch to the code it gates and the section that describes the mechanism. On/off switches accept `1/0`, `true/false`, `yes/no` and `on/off` (`parse_bool_setting`); an invalid value keeps the default and logs once.

| Variable | Code symbol / gate | Mechanism |
|---|---|---|
| `GAMESCOPE_FRAMEGEN_CLASSIC` | classic-planner selector near `framegen_setting` helpers | selects gap-rounded batches on a dedicated queue instead of the causal scheduler (4.1) |
| `GAMESCOPE_FRAMEGEN_SINGLE_QUEUE` | `hasFramegenQueue()` | forces the shared-queue regime (4.1, 4.2) |
| `GAMESCOPE_FRAMEGEN_RECOVER` | `framegen_recovery_enabled`, `framegen_recovery_active_for_path` | enables ladder recovery probes (4.4); no effect with `_BIDIR` or `_VRR_HYBRID` |
| `GAMESCOPE_FRAMEGEN_JIT` | none | legacy; no effect, logs a notice (the causal deadline scheduler is the default) |
| `GAMESCOPE_FRAMEGEN_VRR_HYBRID` | `vulkan_framegen_vrr_hybrid_requested`, `framegen_vrr_hybrid_submit` | #01 mid-interval generated flip; dedicated queue and active VRR only (Section 9) |
| `GAMESCOPE_FRAMEGEN_BASE` | `framegen_base_layer_enabled`, `framegen_base_layer_usable`, `framegen_base_present_composite` | #02 generation on the pre-upscale base layer with late overlay composite (Section 9); no dedicated queue needed |
| `GAMESCOPE_FRAMEGEN_BIDIR` | `vulkan_framegen_bidir_active` | B3 interpolation between two reals; requires motion mode, ignored with a log line when `_VRR_HYBRID` is requested, and combinable with `_BASE`, which is decided per scene: base-layer scenes run causal, scenes that fall back to output-space generation run bidir (3.3) |
| `GAMESCOPE_FRAMEGEN_BIDIR_OCCLUSION`, `_BIDIR_TRACE` | bidir warp specialization | experimental bidir A/B knobs (3.3) |
| `GAMESCOPE_FRAMEGEN_FB`, `_FB_TOL`, `_AGREE` | `framegen_fbcheck_enabled` and the warp agreement test | Stage-B forward-backward and two-source checks (3.2) |
| `GAMESCOPE_FRAMEGEN_SHARP_FALLBACK` | `framegen_sharp_fallback`, `FramegenMotionWarpPush_t::sharpFallback` | default on: where the motion evidence is weak the plain warp scales its displacement by the final confidence (one sample) instead of crossfading with extrapolation; `=0` restores the crossfade (3.2) |
| `GAMESCOPE_FRAMEGEN_FIELD_FINE` | `framegen_motion_scale`, `motion_field_scale` in `policy.hpp` | opt-in /4 motion field instead of /8; `warp` and `checked` pipelines only (3.2) |
| `GAMESCOPE_FRAMEGEN_ADAPT` | `framegen_adapt_enabled` | B4 self-supervised adaptation (3.4) |
| `GAMESCOPE_FRAMEGEN_RESERVOIR`, `_SHADING` | Guided pipeline resources | three-frame disocclusion resolver and learned shading focus (3.2.1, 3.5) |
| `GAMESCOPE_FRAMEGEN_NET`, `_NET_ONLINE`, `_NET_LR`, `_NET_EVERY`, `_NET_PROFILE` | `framegen_net_weights_path` and the C2 trainer | Stage C learned refiner and in-situ learning (3.5) |
| `GAMESCOPE_FRAMEGEN_RECORD`, `_RECORD_MAX`, `GAMESCOPE_FRAMEGEN_RECORD_COLOR` and its `_SWEEP/_MAX/_SKIP/_SPAN/_OFFSET/_PHASE_TOLERANCE` | dataset and held-out capture paths | training capture and E2 validation (3.5, 8) |
| `GAMESCOPE_FRAMEGEN_BENCHMARK` | early-exit in `src/main.cpp` | runs the shader microbenchmark then `exit(0)` before output creation (7) |
| `GAMESCOPE_FRAMEGEN_DEBUG_EVERY`, `_METRICS` | debug log rate, `metrics.hpp` | observability (8) |
| `GAMESCOPE_FRAMEGEN_HUD`, `_HUD_SCALE`, `_HUD_CORNER` | `hud.hpp` | built-in present-only status HUD (8) |
| `GAMESCOPE_FRAMEGEN_CURSOR` | cursor handling for generated frames | cursor composite on generated frames (Section 6) |
| `GAMESCOPE_FRAMEGEN_COMMIT_LEAD_MS` | commit-lead override in `rendervulkan.cpp` | overrides the commit lead used by deadline planning (4.1) |
| `GAMESCOPE_FRAMEGEN_BATCH_REPAINT` | `framegen_batch_repaint` | opt-in composite request (`force_repaint()`) after each generated batch instead of only waking the compositor; causal mode only (5) |

### Runtime hot path
- **`framegen_record_real_frame()`** in `src/rendervulkan.cpp` — the scheduler, called from the tail of `vulkan_composite` for eligible real commits.
- **Present arbiter** in `src/steamcompmgr.cpp` — chooses real, generated, or hardware repeat for each display opportunity.
- **`vulkan_framegen_consume_generated_frame(pFrameInfo)`** — returns a scanout-ready generated frame to `CDRMBackend::Present` or `CWaylandConnector::Present`. In base-layer mode it substitutes the generated base into `layers[0]` and calls `framegen_base_present_composite`; output-space mode returns the pre-composited generated image.
- **`IBackend::SupportsFramegen()`** in `src/backend.h` — defaults false; DRM, nested Wayland and SDL return true, and `CDeferredBackend` delegates to its child. On headless and OpenVR, `--experimental-framegen` is ignored with a one-time log. It contains the forced-composite/present-timing cost to capable backends.

### Shader build wiring (`src/meson.build`)
The `cs_framegen_*.comp` shaders (see the `SHADER()` table for the current set) are compiled to SPIR-V C-arrays (`meson.build`) and registered unconditionally by the `SHADER()` macro table; `MOTION_WARP_ACCEL` serves Predict acceleration and Guided reconstruction. **Load-bearing:** when `!supportsShaderFloat16`, the FP16 enum slots are **aliased to the fp32 SPIR-V arrays**, so the dispatcher can name an fp16 `ShaderType` unconditionally with no null-pipeline branch at dispatch time.

---

## 2. End-to-end dataflow (one real frame)

This is what happens between the game finishing a frame and the display showing something. Steps 1 to 5 run once per real frame. Steps 6 to 9 run once per generated frame or per display refresh. The GPU split is summarized in [2.1](#21-dual-gpu-path).

1. **Client render (render GPU) + WSI handoff.** The game renders on the strong GPU. The FROG WSI
   layer (it makes the game hand its frames to gamescope) forces `VK_PRESENT_MODE_MAILBOX_KHR` unconditionally (`layer/VkLayer_FROG_gamescope_wsi.cpp`)
   and only chains `VK_EXT_swapchain_maintenance1` structs when the driver supports **both** the
   extension and the feature bit (`deviceSupportsSwapchainMaintenance1()`, the `CreateDevice`
   gate, and the per-swapchain capability memo). The gate exists because the unconditional force once failed
   device creation on the NVIDIA proprietary driver with `VK_ERROR_INITIALIZATION_FAILED`, which
   blocked any NVIDIA-render / AMD-present setup (history in Section 10).
2. **dma-buf import to the compositor GPU.** DRM `drm_fbid_from_dmabuf` / Wayland
   `ImportDmabufToBackend`, in `DRMBackend.cpp` and `WaylandBackend.cpp`. The per-frame
   `--debug-dual-gpu-route` log traces this path (Section 8).
3. **Force full composite.** `bNeedsFullComposite |= vulkan_framegen_is_enabled()`
   in both present backends means every real frame goes through `vulkan_composite`, so
   a full-screen image lands in the output ring to serve as zero-copy
   history. This forced-composite tax applies **even when generation is dormant**; it is the cost
   the `SupportsFramegen()` gate contains, and it measures about +2 ms of source-to-flip latency (§5).
4. **Composite on the present card.** `vulkan_composite` records into the
   composite command buffer and returns a scratch-timeline `sequence`. The real frame's present
   waits **only** on that composite `sequence`.
5. **Framegen decision.** `framegen_record_real_frame` runs the overlay-only filter,
   history-invalidation checks, the generate/dormant gate, multiplier adaptation (classic and shared-queue only, 4.3), and the
   degradation-ladder step, then shifts zero-copy history `previousReal ← currentReal ←
   composited image`.
6. **Generation on the framegen queue.** The default causal scheduler submits one exact target at a
   time; bidir submits the candidates for its absolute epoch, while the classic A/B fallback retains
   gap-rounded batches. Each submission uses one `markFramegen()` command buffer, brackets it with GPU timestamps,
   and submits via `submitFramegen` (`src/framegen/device.cpp`). With queue index 1 it waits the
   composite scratch-timeline value, `QueueSubmit`s to `m_framegenQueue`, and signals `m_framegenTimeline`;
   single-queue devices submit in order on the normal scratch timeline. The submit path
   appends `PendingGenerated_t` entries and pins `genReadA/B = previousReal/currentReal`,
   `genReadSeqNo = batch seqNo`. Then `nudge_steamcompmgr()` in `src/steamcompmgr.cpp` wakes the
   compositor thread to run the empty-vblank iteration at all; only the opt-in
   `GAMESCOPE_FRAMEGEN_BATCH_REPAINT` turns that wake into a composite request (`force_repaint()`, §5).
7. **Ring acquisition.** Before every real composite, the output ring is scanned from `nOutImage`
   for a slot whose `CVulkanTexture::IsInUse()` is false. This covers logical history, unfinished
   cross-queue reads, KMS requests, and nested-Wayland attaches with one ownership rule. Completed
   `genReadA/B` pins retire at their framegen timeline point. If backend pressure fills the ring,
   the exceptional path runs one non-blocking backend event-progress pass and retries. It then
   invalidates speculative history and retries once more; persistent pressure skips the composite
   instead of overwriting a live DMA-BUF. The steady-state path performs no additional poll.
   `nLastOutImage` records the actual target.
8. **Present decision (per deadline).** The arbiter in `src/steamcompmgr.cpp` gives a due real endpoint
   or new causal real content priority, then a ready generated deadline, else HW repeat. Bidir keeps
   its timestamp-ordered delayed-real endpoints instead of applying causal supersede semantics.
9. **Present in an available slot.** `vulkan_framegen_consume_generated_frame()` presents an
   output-space result as a single full-screen base layer; base-layer mode first runs its late
   overlay/cursor composite. Classic fixed-refresh mode disables VRR for the generated present;
   the explicit VRR-hybrid path carries the real present's VRR state. Consume drops the front
   slot if `!hasCompletedFramegen` (never stalls). A drained default causal slot replans from the
   absolute display timeline; only the retained classic path calls `framegen_refill_idle`.

**Two clocks.** Framegen paces on `nFramegenRefreshMhz = g_nNestedRefresh ? g_nNestedRefresh :
g_nOutputRefresh` → `ulVblankIntervalNs = 1e12 / mHz`, which must equal
`CVBlankTimer::GetRefresh()`. In nested mode `g_nOutputRefresh` is later overwritten with the
*parent monitor's* refresh while slots are still placed on `g_nNestedRefresh`'s cadence — deriving
the interval from `g_nOutputRefresh` there desyncs phase/strength and was the temporal wobble. On
DRM the two are equal (no-op).
Pure interval, phase, strength, and VRR equivalent-gap arithmetic
lives in `src/framegen/temporal.hpp`. The deterministic online cadence state and
fixed-deadline decision live in `src/framegen/scheduling.hpp`; clock acquisition,
timestamp provenance, mutable history, and the decision to submit remain in
`rendervulkan.cpp`.

### 2.1 Dual-GPU path

The design target is a hybrid or dual-GPU system. The game renders on the strong GPU. A different GPU, the one the display is connected to, composites, generates and presents. The generation passes are memory-bandwidth-bound, so their cost lands on the present GPU. Validation uses this topology; single-GPU runs do not exercise the import path.

| Stage | GPU | File or symbol |
|---|---|---|
| Game renders | render GPU | the game; picked with `MESA_VK_DEVICE_SELECT` |
| Swapchain handoff | render GPU | WSI layer forces `VK_PRESENT_MODE_MAILBOX_KHR` (`layer/VkLayer_FROG_gamescope_wsi.cpp`) |
| Frame leaves the render GPU | both | dma-buf export and import |
| Import on the present GPU | present GPU | DRM: `drm_fbid_from_dmabuf` in `src/Backends/DRMBackend.cpp`; nested: `ImportDmabufToBackend` in `src/Backends/WaylandBackend.cpp` |
| Composite, framegen, present | present GPU | `vulkan_composite`, then the framegen queue, then the backend `Present` |

Selectors: `--prefer-vk-device <vendor:device>` picks the present GPU (gamescope's compositing device). `MESA_VK_DEVICE_SELECT` picks the render GPU for the game. The HUD `present` row shows whether client buffers arrive `local` (same GPU, zero-copy) or `xGPU` (copied once into present-GPU memory). Setup steps are in the [user guide](framegen-howto.md); the one open cross-GPU bug, bug-179, is in Section 10.

---

## 3. Generation modes and pipelines (the core)

Two axes select the work. The **mode** (`extrapolate` / `motion` / `blend`) picks the shader family.
The **pipeline** (`warp` / `checked` / `learned` / `predict` / `guided`) is meaningful only in motion
mode: it picks the pass set and is also the rung the degradation ladder walks (4.4). The **bidir/causal**
split (3.3) is a third, independent axis: causal extrapolates past the newest real, bidir interpolates
between two reals. The matrix in 3.1 enumerates the resulting variants.

`framegen_submit_planned` picks Motion / Blend / Extrapolate per `eff.mode`. Extrapolate fuses
slots in **pairs** (`extrapolatePair(i, i+1)`, odd trailing slot → single `extrapolate`). All
shaders are `local_size 8×8`. `framegen_bind_history` binds:
`s_samplers[0]=previousReal`, `[1]=currentReal`, both `setTextureSrgb(true)`, which selects
`srgbView()`. Despite the name that is the UNORM view: shaders read raw output code values, with no
hardware linearization (see "sRGB-view naming" in §10).

### 3.1 Comparison matrix

The first five rows are the **non-motion shaders**: four extrapolate variants (`--framegen-mode extrapolate`, the default) and the debug-only blend (`--framegen-mode blend`). Each is one cheap shader with no motion search. The last five are the **motion pipelines** (`--framegen-mode motion`): a block-matching estimator plus a warp. Status wording is repeated from the [Pipelines table in the reference](framegen-reference.md#pipelines), which is where it is kept up to date; the verdicts come from one viewer in real games (Section 10).

| Variant | Math | Memory strategy | Dispatched when | Tradeoff / cost | Status |
|---|---|---|---|---|---|
| **Extrapolate LDS fp32** (`cs_framegen_extrapolate.comp`) | `delta = cur−prev`; `strength = u_strength·(1−smoothstep(0.08,0.40,motion))`, `motion = max(\|Δr\|,\|Δg\|,\|Δb\|)`; `predicted = cur + delta·strength`; full 3×3 neighbor clamp; **no [0,1] clamp**; alpha = `cur.a` | 10×10 LDS apron `s_cur[100]` of CURRENT only, staged once/group; diagonals add LDS reads but no image traffic | non-NVIDIA integer targets | preserves diagonal/thin motion better; halos bounded by suppression + neighbor clamp | Default (chosen by GPU) |
| **Extrapolate direct** (`_extrapolate_direct.comp`) | **bit-identical** to LDS fp32 | no LDS / no barrier; neighbors via `texelFetch` straight from texture cache | current measured NVIDIA override (`vendorID==0x10DE`) | identical output; **~30–37% faster** on the measured NVIDIA path (measured) (large L2 makes the apron pure overhead); also beats fp16 there | Default on NVIDIA |
| **Extrapolate fp16** (`_extrapolate_fp16.comp`) | same math, `f16vec3` ALU; push uploaded fp32 then cast; alpha never demoted | **LDS apron stays vec4 fp32** — only ALU is fp16, bandwidth unchanged, so upside is capped | `supportsShaderFloat16 && !floatTarget`, non-NVIDIA | fp16 range/precision bands scRGB highlights → **gated off for float targets** | Default where eligible |
| **Extrapolate pair (+ pair_fp16)** (`_extrapolate_pair.comp`) | writes TWO slots from one dispatch; suppress term + neighbor min/max computed **once** and shared | halves the two full-res history reads across the pair | a submission carries two or more slots (classic and bidir batches); not used by the one-slot causal scheduler | same quality; halves history bandwidth across the pair | Used whenever a batch has two or more slots |
| **Blend** — DEBUG ONLY (`cs_framegen_blend.comp`) | `mix(prev, cur, phase)` full vec4 **including alpha**; **INTERPOLATION**; no suppression/rectification | normalized-coord **bilinear** `texture()` (the only framegen shader using sampler filtering) | `eff.mode==Blend` only | structurally violates FORWARD-ONLY (§0) → judder; reference lever | Debug only |
| **Motion warp** | forward luma pyramid + hierarchical match + constant-velocity warp | no reverse field, stats, or ML dispatches | `eff.pipeline==Warp` | cheapest motion-compensated pipeline; default | Opt-in mode, default pipeline |
| **Motion checked** | warp + reverse match / FB check + full-res agreement | reverse chain once per real | `eff.pipeline==Checked` | rejects disocclusions and boundary bleed; in a real game its rejections felt choppy | Opt-in; felt choppy |
| **Motion learned** | checked + B4 adaptation; optional learned refiner/training | probe once per real; with the net, one refiner pass per real | `eff.pipeline==Learned` | self-tuning, and with `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` the in-game net that rescues rejected vectors that beat holding; the best real-game start so far | Opt-in; best real-game start so far (one viewer) |
| **Motion predict** | learned + retained-field acceleration warp | one field copy per estimated interval + one low-res field sample per output pixel | `eff.pipeline==Predict` | bounded second-order forward prediction; felt choppy in a real game (acceleration looks net negative so far) | Opt-in; felt choppy, not recommended |
| **Motion guided** | predict + color-guided reconstruction + three-frame disocclusion resolver; optional learned shading focus | bounded boundary hypotheses; two 1/8-res luma history images; with net, one low-res focus write and a bounded color-trend term in the existing warp | `eff.pipeline==Guided` | most passes and cost; a close second to `learned` in real games | Opt-in; costliest, close second |

The Predict/Guided acceleration, guided reconstruction, reservoir and shading entries are causal
only. Bidir uses its symmetric checked-field warp at every pipeline; Learned+ adds B4 and optional
confidence-only ML, but does not schedule those causal extensions.

**Invariant across all extrapolate variants:** alpha pinned to `current`; the *only* range bound is
the 9-tap neighbor clamp — no `[0,1]` clamp, so UNORM stays in range while scRGB keeps HDR>1.0 and
wide-gamut negatives; previous frame sampled at center only.

### 3.2 Motion mode detail

The passes of the motion pipelines, in order. The first four run once per real-frame pair; only the warp runs per generated slot. Every `cs_framegen_motion_*` shader is in `src/shaders`.

| Pass | Shader | Resolution | Runs per | Output | Pipelines |
|---|---|---|---|---|---|
| Luma and pyramid | `_luma_pair.comp`, `_pyramid.comp` (`_rgba` twins) | /8, then /16 and /32 | real pair | luma pyramids of both frames | all |
| Block match | `_match.comp`, `_match_refine.comp` | coarse (/32) to fine | real pair | forward motion field (vector + confidence) | all |
| FB check | `_fbcheck.comp` | field resolution | real pair | reverse field, confidence killed where vectors do not round-trip | `checked` and above (always on under bidir) |
| Stats and trust (B4) | `_stats.comp`, `_stats_apply.comp` | field resolution | real pair | adaptation thresholds, field trust, scene-cut flag | `learned` and above |
| Learned refiner | `_net.comp` (training: `_net_train.comp`, `_net_opt.comp`) | field resolution | real pair, opt-in | refined field | `learned` and above, with `GAMESCOPE_FRAMEGEN_NET` or `_NET_ONLINE` |
| Warp | `_warp.comp` (`_warp_accel.comp` for `predict` and `guided`, `_bidir.comp` for bidir) | full resolution | each slot | the generated frame | all |

Stages 1+2, learned refinement, B4 probing and online training run **once per real-frame pair**
through `framegen_prepare_motion`; stage 3 runs **per slot** through `framegen_warp_slot`. The
final field is cached by real-frame ID, pipeline and causal/bidir mode, so later deadline/classic-refill slots for
the same pair issue only their warps and cannot overweight a slow interval in online training.
Falls back to extrapolate for the whole batch if intermediates can't be allocated.

- **Stage 1 — luma pyramid** (`cs_framegen_motion_luma_pair.comp` +
  `cs_framegen_motion_pyramid.comp`): **4-tap bilinear box** to 1/8-res luma, both frames in one
  dispatch (`dstPrev=binding1`, `dstCur=binding2`), then two further 2× box steps (one bilinear tap
  each, both frames per dispatch) to 1/16 and 1/32 res. The box filter is a **wobble root-cause
  fix** — a single centered tap covers ~4/64 px, so sub-block translation aliases the low-res luma
  and the integer matcher toggles its winner frame-to-frame. Applies the Rec.709 weights to the
  reals' **output code values**: `setTextureSrgb(true)` selects the UNORM `srgbView()` (see
  "sRGB-view naming" in §10), so for 8/10-bit outputs the luma, and with it the matching, runs on
  gamma-coded values, and for float outputs on linear ones. Luma targets are R16F when
  `framegen_format_supports_sampled_storage(R16F)` (¼ bandwidth, no sRGB view to mis-decode), else
  ABGR16161616F (the `_rgba` shader twins). The scales in this and the next stages are the default
  /8 field; the opt-in /4 field is described after stage 3.
- **Stage 2 — coarse-to-fine SAD block match** (`cs_framegen_motion_match.comp` at the coarsest
  level, `cs_framegen_motion_match_refine.comp` below it): the full `[−R,R]²` search
  (R = `clamp(4,1,MAX_RADIUS=4)`, LDS-staged 20×20 prev window + 10×10 cur) runs **only at the
  1/32-res level**, where it spans ≈ **±128 full-res px** of motion (4× the old single-level range)
  and each 9-tap SAD integrates 4× the content — the context that disambiguates self-similar detail
  (particle fields, tiled textures) the fine level alone confidently mismatches. Each finer level
  takes the **vector median** (min L1-distance sum) of its 3×3 coarse-field neighbourhood as the
  seed — an isolated wrong coarse match would otherwise seed a whole coarse-texel-sized band with a
  confident large displacement whose boundary shears like a tear line — doubles it, and searches a
  **±1 window around the rounded seed** (9 candidates instead of 81 at the finest, largest level;
  no LDS — seeds diverge per invocation and the per-level luma fits in texture cache), regularized
  `0.02·|mv−seedF|₁` toward the *fractional* seed so flat-content ties follow the better-informed
  coarse motion. A **zero-MV
  static anchor** is always evaluated with the same seed-distance penalty, so a hallucinated coarse
  vector over genuinely static detail can always snap back. The finest pass keeps the **sub-pixel
  parabolic refinement** (*un-regularized*, guarded by `denom>1e-5`) that de-quantizes the
  ~8·strength px staircase, and the **two-factor confidence**
  `clamp(1−residual·6,0,1)·clamp((zeroSad−bestSad)·4,0,1)`. Reads luma **raw** (`setTextureSrgb
  false`; R16F has no sRGB view, so the `texelFetch` can't mis-decode). Writes `{mv.x, mv.y, conf,
  0}` to an RGBA16F field (MV is prev→cur flow, in 1/8-res texels), sampled **bilinearly** by the
  warp (the whole sub-pixel scheme depends on this).
- **Stage 2.5 — forward-backward consistency** (`cs_framegen_motion_fbcheck.comp`, default on,
  `GAMESCOPE_FRAMEGEN_FB=0` disables): the same coarse-to-fine chain runs a second time with the
  two luma bindings swapped, estimating the **reverse flow anchored at the previous frame**
  (reusing the coarse fields as scratch — only two extra full-res field images). A correct forward
  vector round-trips: `R(p − F(p)) ≈ −F(p)`. Lookalike mislocks and disocclusions don't, and get
  their confidence killed so the warp falls back there (to the held real frame by default, to
  bounded extrapolation in the crossfade variant of stage 3). The check tests
  the **four covering reverse texels individually** (a bilinear blend across a motion boundary
  fails spuriously), forgives `tolBase + tolSlope·(|F|+|R|)` texels of error
  (`GAMESCOPE_FRAMEGEN_FB_TOL` overrides tolBase, default 0.75), and **feathers the resulting kill
  mask with a 3×3 tent** — a raw per-texel kill is an 8×8-px tile snapping to the static fallback
  while its neighbours warp, which reads as "tearing tiles" and flickers at the real-frame rate.
  `.a` carries the raw round-trip error for future per-tile quality feedback.
- **Stage 3 — warp** (`cs_framegen_motion_warp.comp`): `predictedMotion = texture(currentReal,
  uv − mvUv·strength)` (backward gather along the flow); alpha = `cur.a`. The warp adds a
  **per-pixel two-source agreement test** (`GAMESCOPE_FRAMEGEN_AGREE=0` disables): the same flow is
  also projected from the previous frame (`uv − mvUv·(strength+1)`, one interval further along);
  where the two projections read different content the flow is wrong *for that pixel* and `conf` is
  killed at full resolution — the 1/8-res confidence field cannot make this call, and a
  mid-confidence mix of two positions is a double exposure (ghosted edges, doubled text). The
  distance is normalized by local magnitude so the threshold stays meaningful for scRGB values
  above 1.0. What the final `conf` does to the output is the **output stage**, selected by the
  `sharpFallback` push constant (`framegen_sharp_fallback()` fills it into the 28-byte
  `FramegenMotionWarpPush_t`):
  - **Sharp fallback (default; `GAMESCOPE_FRAMEGEN_SHARP_FALLBACK`).** Where the motion evidence is
    weak the real frame is moved only as far as the confidence allows: the displacement is scaled
    by the final confidence and the shader takes one final sample,
    `texture(currentReal, uv − mvUv·strength·conf)`. Low confidence therefore degrades toward
    holding the real frame instead of toward a second, differently placed image. Only real-frame
    samples are shown, so HDR is unaffected. Per generated frame it costs the same GPU time at
    1080p and about 10% less at 1440p and 2160p than the crossfade (Radeon 890M, Appendix A). It is
    the default because in blind A/Bs (one viewer, `checked`, 120 Hz) it beat framegen off on
    Xonotic and on GravityMark, while the crossfade had lost to off on Xonotic.
  - **Crossfade (`GAMESCOPE_FRAMEGEN_SHARP_FALLBACK=0`).** `predictedExtra = cur + delta·strength`
    with suppression; `predicted = mix(predictedExtra, predictedMotion, conf)`. **Key subtlety:** the
    TAA 4-neighbor clamp is applied **only to `predictedExtra`** — clamping the motion term would
    cap displacement at ~1 texel/frame and defeat the warp. *Confidence, not clamping, is the motion
    term's safety*: a poor match lowers `conf` and blends back toward the bounded extrapolation.
    Crossfading two differently placed images shows as double edges and blur wherever confidence
    changes.
  - **Which pipelines.** This shader serves `warp`, `checked` and `learned`, and `predict` whenever
    its acceleration history is not valid (3.2.1). `predict` with history and `guided` use
    `cs_framegen_motion_warp_accel.comp`, which keeps the crossfade (its push block has no
    `sharpFallback`); bidir has its own shader (3.3), whose fallback is a phase-correct crossfade.
  - **Known limit: swarm fraying.** On dense particle swarms the sharp fallback frays edges
    (GravityMark's 1M asteroids: "clearly smoother, frayed edges" against framegen off). Dropping
    the per-pixel agreement test from the sharp path, so that only the field confidence counts,
    removed the fraying but made Xonotic choppier in a blind test, so the test was kept.

In the crossfade variant `predicted = mix(predictedExtra, predictedMotion, conf)`, so **motion is a
strict superset of extrapolate** — `conf=0` reproduces the bounded extrapolate output exactly. In
the default sharp variant `conf=0` shows the current real frame unchanged, a hold. Either way the
failure floor is bounded (a clamped extrapolation or a hold), so it can never be much worse even
where matching fails.

**Finer field (opt-in, `GAMESCOPE_FRAMEGEN_FIELD_FINE=1`).** The stages above describe the default
/8 field (`k_uMotionReferenceScale`); the opt-in makes it /4 (`k_uMotionFineScale`): 4× the field
texels and one more pyramid level, hence one more pyramid dispatch and one more refinement per
direction (the pyramid is a level loop over `motion_coarse_levels(scale)`). The full ±R search
still runs only at /32, so the motion range (≈ ±128 px) is unchanged, and the constants tuned in
/8 field texels (the FB tolerance, the static-motion bar) are converted with
`motion_reference_texels` to keep their meaning in pixels. **Scope:** `warp` and `checked` only.
`motion_field_scale` takes the *configured* pipeline, never the effective one, so a ladder step can
never switch the finer, costlier field on; the learned net is trained on /8 fields, so `learned`,
`predict` and `guided` ignore the variable. `GAMESCOPE_FRAMEGEN_RECORD` is ignored while it is on
(GSFD samples are /8). It stays opt-in because the eye and the offline gate disagree: in blind
A/Bs it looked crisper on Xonotic, tied on the vkmark desktop and blocky on GravityMark's swarm,
although `scripts/framegen-temporal-gate.py` scored /4 better than /8 on flicker in every content
class; blind A/Bs outrank that metric.

#### 3.2.1 Predict causal acceleration and Guided reconstruction

`cs_framegen_motion_warp_accel.comp` uses a third causal interval without requiring application
motion vectors, a vendor optical-flow block, or a future frame. The final
checked/refined/trust-scaled forward field remains resident as the current pair's cached field. At
the start of the next consecutive, same-pipeline causal pair it is copied to `mvFieldHistory`
**before** preparation overwrites the working images; same-pair deadline/classic refills therefore retain both
the finalized current field and the true preceding field. The shader samples that history at
`uv-currentFlow`, so the old and current vectors describe the same moving content. The fields are interval
**displacements**, not velocities, so cadence jitter must not become false acceleration. For the
current real-frame interval `h0` and preceding interval `h1`, the shader evaluates
`A = F_current - F_old·h0/h1`; the future displacement is
`s·F_current + h0/(h0+h1)·s·(s+1)·A`. This is the exact irregular-sample quadratic and reduces to
`s·F + 0.5·s·(s+1)·A` when the intervals match. The same ratio normalizes the causal shading-trend
training target. Missing timestamps or an interval ratio beyond 4:1 disables the second derivative
for one interval and falls back to constant velocity; the freshly stored field re-primes the next.
The CPU-side interval comparability, `h0/h1`, and `h0/(h0+h1)` calculations are
centralized in `src/framegen/temporal.hpp` and contract-tested at the 4:1 boundary.

Acceleration has four structural guards: both fields must retain confidence; a 0.05-field-texel
dead zone removes sub-pel estimator noise; acceleration fades out when large relative to observed
speed and is capped to ±1 field texel per component; and the existing full-resolution two-source
agreement still decides whether the motion gather reaches the output. The retained field carries a
real-frame ID. If shared-queue admission skips even one interval, the ID is non-consecutive and the
original constant-velocity shader runs instead. Scene invalidation clears the ID. Cheaper pipelines never
allocate, copy, bind, or dispatch this path.

Guided treats the 1/8-resolution field as a set of motion hypotheses instead of bilinearly
turning a foreground vector and a background vector into a vector that no object followed. For each
full-resolution output pixel, `reconstructField` evaluates the four covering field texels by
gathering the previous real frame along each candidate and comparing it with the current pixel.
It selects a discrete layer only when the candidates' vectors diverge and one correspondence wins
unambiguously; smooth or textureless regions retain the bilinear field. The preceding interval is
selected from the history layer whose velocity is nearest to that reconstructed current flow, so a
foreground/background mixture cannot masquerade as acceleration. This is a causal, vendor-neutral
software approximation of the most valuable part of dense optical flow: motion-boundary ownership.

The Guided disocclusion resolver extends that ownership test across a third
real frame without retaining another output-ring slot. Two internally owned
1/8-resolution luma images form a frame-ID-keyed ping-pong: one remains the
`current-2` evidence for every initial/deadline/refill warp of the current interval,
while the other receives `lumaPrev` (`current-1`) after the warps and becomes
evidence for the next interval. Refill sees that ID already published and skips
the copy. This avoids another full-resolution history pin in the ownership-aware output ring and costs only a
field-resolution image copy (cost in Appendix A).

The shader searches the eight adjacent field hypotheses only where centre
confidence is low **and** `.a` reports a large local FB round-trip error. A
candidate must be a distinct, FB-consistent motion layer; its reprojected path
through the preceding field must remain on-screen and agree across current
color, previous color, and the two-interval-old luma. The accepted output color
always comes from the newest real frame, not the reservoir, so the mechanism
cannot create a stale historical-color trail. Its capped trust blends into the
bounded-extrapolation fallback (this shader has no sharp fallback, 3.2 stage 3). Scene-cut
sentinels, skipped
field IDs, allocation failure, bidir, or a ladder step to Predict disable it
without a separate pipeline. This is screen-space evidence, not depth: it
cannot reconstruct out-of-screen or long-hidden content.

### 3.3 Bidirectional interpolation (B3, `GAMESCOPE_FRAMEGEN_BIDIR=1`, opt-in)

Extrapolation's structural residual is *temporal*: killed regions hold the current frame and jump
at the base rate while confident regions glide every vblank ("screenshot sharp, motion strange").
Bidir removes it by generating BETWEEN the two real frames instead of past the newest one:

- **Estimation** (`framegen_prepare_motion`): forces the FB chain on and adds one more `fbcheck`
  dispatch with the fields swapped, producing a **checked reverse field** (`mvFieldRevChk`) — the
  reverse chain B2 already pays for becomes a first-class gather field. (~+7 µs, one extra 1/8-res
  image.)
- **Warp** (`cs_framegen_motion_bidir.comp`): for phase `t`, gather
  `fromCur = cur(uv + F·(1−t))` and `fromPrev = prev(uv + R·t)`, blend by
  `w = conf × phase-proximity` (`wF = confF·t`, `wR = confR·(1−t)`). Occlusion side-selection
  falls out of the weights: content that exists in only one frame keeps only that direction's
  confidence. A **mixedness-scaled agreement test** kills even-blend double exposures, and the
  fallback is a **phase-correct crossfade** `mix(prev, cur, t)` — real content, correct for
  translucency (both motion layers fade on schedule), temporally smooth (killed regions dissolve
  instead of hold-jumping). No TAA clamp needed (no term invents colors) — the bidir warp is
  *cheaper* than the forward warp (measured: 0.095 vs 0.136 ms @1080p int on Blackwell, one rig).
- **Learned policy** (`cs_framegen_motion_net.comp`): the endpoint-trained net runs for both
  checked directions, but defaults to a **confidence veto**. Its flow scale and positive-confidence
  scale are zero, so `.rg` remains bit-exact to the FB-checked field and `.b` can only decrease.
  Online training recomputes the raw-field fallback competition but backpropagates only into W3's
  confidence row/bias; flow rows, shading row and the shared trunk receive exact-zero gradients.
  This is not merely a clamp: two endpoint-photometrically-equivalent vectors can trace different
  intermediate paths (aperture/repeated-texture ambiguity), so endpoint supervision is not an
  independent validator for VFI geometry. Causal acceleration, guided reconstruction, the three-frame
  reservoir and shading persistence are forward-prediction mechanisms and are not scheduled here.
- **Phase placement:** bidirectional phases are timestamp-exact since the deadline timeline; no
  gap-rounded phase-bias control remains.
- **Scheduling** (the substantive change). The pending queue is an absolute-deadline
  **presentation timeline**; see the diagram below. Five rules:
  - **Timeline.** The bidir epoch planner appends timestamped interpolations and real endpoints, even
    when a source interval contains no intermediate display target. The backends flip
    `vulkan_framegen_bidir_flip_texture(composite)`. In steady state the queue front at a real paint
    is an older real frame, so each real frame presents at the first refresh at or after its record
    time plus D. D is the intrinsic interpolation latency, which is why bidir is opt-in.
  - **Drain.** Interpolations drain on repeat vblanks through the normal consume path. Discard and
    invalidate paths preserve queued `bReal` entries (content painted but not yet seen); ring
    advance pins their slots.
  - **No supersede.** The supersede discard is skipped in `steamcompmgr`, so a newer real frame does
    not throw away the queue.
  - **Snap back to live.** Composites outside the timeline (dormant, prime and keep-up frames)
    flip immediately and drop the stale pending entries. Bidir therefore degrades to zero added
    latency when the game keeps up. Overlay-only recomposites present the queue front instead: the
    game content is still queued, and the overlay rides the next real frame.
  - **Ladder routing.** The degraded ladder rung routes to **blend** (the phase-correct
    crossfade), never to extrapolate, because an extrapolated slot would come from a different
    timeline.
- **Multiplier.** In bidir mode N caps interpolations at N-1 per real pair (`plan_bidir_step`). Slots beyond the cap stay empty and repeat. Measured at about 41 fps on 120 Hz: x2 filled 71% of slots, x3 98%, x4 100%. See [Multipliers](framegen-reference.md#multipliers).
- **Combinations.** `GAMESCOPE_FRAMEGEN_BIDIR` with `GAMESCOPE_FRAMEGEN_BASE` is allowed and decided
  per scene: base-layer scenes run causal, and scenes that fall back to output-space generation run
  bidir. `GAMESCOPE_FRAMEGEN_BIDIR` with `GAMESCOPE_FRAMEGEN_VRR_HYBRID` is not allowed; bidir is
  ignored and the log says so.

Example timeline at 120 Hz (T = 8.3 ms), a 40 fps source (25 ms between reals) and D = 33 ms.
Each real frame is shown D after it was recorded, and the slots between two reals show
interpolations at phase 1/3 and 2/3:

```text
recorded (live):   R0 . . . . . . . . . . . . . R1 . . . . . . . . . . . . R2
                   |<-------- 25 ms ---------->|
display slot T:    | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
shown (D = 33 ms):                  R0    I(1/3)  I(2/3)   R1    I(1/3)  I(2/3)  R2
                                    ^
                                    R0 is on screen at record(R0) + D
```

#### Known limits and experiments (3.3)

Research notes that sit beside the mechanics above. Neither changes the default.

- **Endpoint-grid correction (experimental, Guided only):** `F` is anchored at the current
  endpoint and `R` at the previous endpoint, while the established warp samples both at the
  intermediate output coordinate. `GAMESCOPE_FRAMEGEN_BIDIR_TRACE` performs one bounded Picard
  step toward the exact relations `x1=p+(1-t)F(x1)` and `x0=p+tR(x0)`. The correction is rejected
  unless both provisional endpoints are in bounds, both checked confidences survive, and
  `F(x1)+R(x0)` closes within a motion-scaled tolerance. Flow and confidence move together: the
  latter is the already checked verdict at that endpoint, not a learned confidence raise. A
  specialization constant keeps the trace instructions out of the trace-off pipeline. There is
  no new image or dispatch; the trace variant adds two low-resolution-field texture samples and
  ALU per full-resolution output pixel. Default remains off pending broader live validation (measured);
  measured results are in [Appendix A](#appendix-a-measurements-dated-non-contractual).
- **Fast-camera residual:** when both directions fail validation over a large source interval, the
  only information-preserving fallback is the phase-correct endpoint crossfade. At ~20 fps the
  endpoints are ~50 ms apart, so violent camera motion can briefly look like a soft paint/splash
  edge before the next real endpoint clears it. Hard endpoint selection removes the blended color
  but introduces hold/jump judder; reusing rejected flow restores sharpness by reintroducing the
  tearing/shape artifacts the checks rejected. Do not change this trade without E2 final-color
  temporal evidence or a genuinely independent motion/depth signal.

### Bidirectional delay controller

**Goal.** Bidir shows each real frame a little late so the interpolated frames before it can be shown first. The controller picks the smallest delay D that keeps this smooth, and keeps it steady so the picture does not stutter when D changes.

**Example.** At 120 Hz, T = 8.3 ms. The game runs at about 40 fps and its frames arrive up to E = 25 ms apart. Then D = round_up(E) + T = 25 + 8.3 = 33 ms. The metrics line reports it as `bidir_delay_ms=33.3`.

| Symbol | Meaning |
|---|---|
| T | One refresh interval (8.3 ms at 120 Hz). |
| E | The latency envelope: how long a real frame typically waits for the next one, measured as below. |
| D | The target delay actually applied: `D = min(round_up(E) + T, D_ring)`, in whole refresh intervals. |
| C | The presentation queue's capacity in frames (`bidir_pending_hard_capacity`, derived from the output ring and the multiplier). |
| P | The real-frame interval observed in the last second, each interval counted as at least T (see D_ring below). |

The implementation is `bidir_target_delay`, `BidirWindow_t`, `BidirRingBound_t` and `plan_bidir_step` in `src/framegen/deadline.hpp`.

Rules:

- T is one refresh interval. Each real frame is planned for the first refresh at or after its record time plus the target delay D = min(round_up(E) + T, D_ring), where round_up() rounds up to whole refresh intervals.
- For a source no faster than the display, once settled every real frame is shown at most D + T after gamescope records it, and consecutive real frames are shown within ±T of their record spacing.
- E is one latency envelope, sampled once per presenter opportunity, while no real frame waits behind another, as the time from the previous real frame's record to the opportunity that plans the current one. A frame limiter holds a frame before it is recorded, so changing the limit does not change D.
- One rule moves E, using the third-highest sample of the last second:
  - D is first taken once that second is full.
  - E rises to that sample as soon as it lies above D, that is, when three samples in the second lie above D wherever they fall.
  - E falls to it once it has stayed at least T below E for 2 s.
  - Fewer than three high samples in a second never move E.
  - E restarts only when the refresh interval, the display chain or the timestamp source changes.
- D_ring is the longest delay, in whole refresh intervals, whose real frames fit the presentation queue: D_ring = round_down((C − 3) × max(P, T)).
  - C is the queue's capacity in frames.
  - 3 leaves room for the current real frame and a burst of two.
  - (C − 3) × max(P, T) is the third-shortest time that C − 3 consecutive real frame intervals, each counted at least T, spanned in the last second. It is taken both between source-ready times and between records, and the longer of the two counts, so a change of the hold between them (a frame limiter settling, catch-up composites) does not move D_ring.
  - D_ring falls at once and rises only after it has stayed higher for 2 s.
  - When the held D exceeds D_ring, D shortens to D_ring at once, as one declared re-prime, and the queued real frames move earlier by the difference.
- A source no faster than the display never loses a real frame to the queue's capacity. At the ceiling the queue sheds interpolations and shows its oldest real frame early.
- A source faster than the display loses the newest queued real frame each time one more would not fit. Each loss is logged, counted in `dreal_drop`, and is not a re-prime.
- Outside a cut, a backlog whose lateness neither shrank nor grew for a second drops only the queued real frames of its excess, newest first. The drop is logged and counted in `dreal_drop`; it keeps D and is not a re-prime.
- `bidir_delay_ms` on the `GAMESCOPE_FRAMEGEN_METRICS=1` line reports the largest D in force during each window. Status, measured numbers and limits are in the user guide's bidirectional section ([framegen-howto.md](framegen-howto.md)).
- **Ring-overflow latch.** The ring bound is predicted from the span measurements only until the first translation. After that, an observed ring overflow (`observe_bidir_ring_overflow`) latches the bound. The latch releases when the predicted bound reaches `k_uBidirRingReleaseFactor` (3) times its value at the time the latch was set. The queue's hard ceiling `bidir_pending_hard_capacity` is the output ring images minus 2, capped at a latency-derived capacity of `2 * (multiplier + 1)`. The other constants are `k_uBidirEnvelopeSamples` (3), `k_ulBidirWindowNs` (1 s) and `k_ulBidirReleaseHoldNs` (2 s).

### 3.4 Self-supervised online adaptation (B4, default on in motion mode)

Every real frame is ground truth for the field that was just estimated from it. After the last
consistency check, a field-res **stats probe** (`cs_framegen_motion_stats.comp`,
`framegen_record_adapt_probe`) warps the previous frame's luma along the final checked field and
compares it against the current frame's luma per texel — a direct measurement of "would this field
have predicted the frame we actually got", which is the exact bet every generated slot in the
coming interval places. Per-texel verdicts LDS-reduce into a 96×1 `R32_UINT` counter image
(total / residual sum / mispredicting-but-surviving count / killed count / static-texel noise
floor / an 8-bin histogram of the FB round-trip error B2 already stores in `field.a` / four-bin
luminance histograms for both real frames over a 3×3 screen partition / the changed-texel counters
behind the `q_*` quality meter, §8). The probe runs once per real frame and costs nothing per generated frame (measured cost: Appendix A).

Two consumers, two latencies:

- **GPU, same batch** (`cs_framegen_motion_stats_apply.comp`): a field-res apply pass folds a
  batch-global **field trust** factor into the field's confidence channel —
  `trust = 1 − smoothstep(0.15, 0.45, mispredicting/surviving)`. The signal isolates regimes where
  the per-texel checks *lie*: texels that passed FB + feathering yet still mispredicted (lighting
  flashes, particle chaos, stroboscopic content). Low trust slides the whole frame toward the safe
  fallback (the crossfade in bidir; forward, the held real frame under the sharp fallback, or
  bounded extrapolation in the crossfade variant) — smoothly, with zero latency, and
  self-recovering the moment the field predicts again. This is deliberately how pipeline-driven
  degradation ships without forcing #04's discrete ladder to react: rung changes are held and
  recovery requires long capacity evidence, while per-frame trust can move smoothly. Perf note: baking
  trust into the field (which the warps already fetch per pixel) is load-bearing — reading the
  stats image from the full-res warps, even once per workgroup, measured ~10% of the warp on
  NVIDIA (repeated surface loads of one texel serialize on the cache).
- **GPU, same batch, content-cut guard**: a one-workgroup finalize pass compares the nine regional
  histogram pairs, then requires three independent signals before declaring a cut: at least seven
  regions changed, global histogram total variation above 0.25, and both motion-compensated
  residual and unreliable-field coverage are high. A pan can break correspondence without changing
  regional distributions; particle chaos can kill confidence without satisfying the histogram
  tests. On a cut, a negative internal `field.a` sentinel makes causal warps copy the newest real
  frame and bidir choose the nearest endpoint. The sentinel is retained into Predict history with
  zero confidence, so the unrelated interval cannot become a false acceleration estimate.
- **CPU, next batch** (`framegen_adapt_consume`): the counters copy into a host-mapped readback
  image; the one-batch-in-flight guarantee makes the mapped read race-free (parsed only after
  `hasCompletedFramegen`). `src/framegen/adaptation.hpp` owns the stateless counter decoder, slow
  EMAs (1/8), and two derived calibrations. The **FB tolerance** loosens (up to 2.5 texels) only on
  *ambiguity-without-error* — round trips fail while measured prediction error stays low (periodic
  textures: fences, grilles, tiling), where the kill would reintroduce fizzle. Entry/exit residual
  thresholds are hysteretic, and the tolerance can rise by at most 0.125 or fall by at most 0.25
  field texel per real frame. This is load-bearing temporal policy: the prior direct assignment
  could jump `0.75 -> 2.5` between adjacent frames and abruptly replace most of the confidence
  mask even though its input statistics were smoothed. The same-batch global trust remains the
  immediate safety response; only next-batch calibration authority is slewed. The
  **agreement window** widens by the measured static-scene noise floor (film grain, dithering) so
  inherent temporal noise stops mass-killing the motion term. An explicit
  `GAMESCOPE_FRAMEGEN_FB_TOL` pins the tolerance (auto-cal keeps its hands off). The renderer still
  owns completion, scene invalidation, readback lifetime, and application to the next dispatch.

`GAMESCOPE_FRAMEGEN_ADAPT=0` disables all of it (B3-bit-exact warps, no adaptation probe);
the measurement-only quality meter still runs when `GAMESCOPE_FRAMEGEN_METRICS=1` or
`--framegen-debug` is set and only reports the `q_*` metrics fields. With
`--framegen-debug`, consume logs
`adapt resid=… bad=…% killed=…% noise=… fbP75=… scene=… -> fbTol=… agree=…/…`.

### 3.5 Learned field refinement (Stage C, opt-in via `GAMESCOPE_FRAMEGEN_NET=<blob>`)

A tiny convolutional net (12→16→16→4 channels, 3×3 kernels, ~4.6k parameters,
`cs_framegen_motion_net.comp`) processes the checked forward motion field at field resolution once
per real frame, with no future-frame dependency. Causal prediction serves **corrections, not
pixels**: a flow residual bounded to ±2 field texels (`2·tanh`), an additive confidence
recalibration, and in Guided a zero-neutral shading-persistence focus. Bidir processes the checked
reverse field too, but defaults to confidence-only service: no learned flow, no confidence raise,
no shading focus. Safety is structural, three layers deep:

1. a zero final layer IS Stage B (the trainer zero-initializes the head, so the failure floor is
   the current behavior, and `--init` exports a bit-neutral blob);
2. causal corrections are bounded by construction (tanh flow, clamped confidence); bidir goes
   further and preserves checked geometry while allowing ML only to select the safe crossfade;
3. the B4 probe grades (and trust-scales) the **refined** field — the one the warps consume — so a
   mispredicting net is clamped in the same batch and shows up in the `adapt resid=…` log lines.

The fourth head does not synthesize color: the Guided warp applies an analytic aligned
`current-previous` color trend, phase-scaled, capped to 8% of local magnitude per interval, and
multiplied by focus, current/old field confidence, acceleration consistency, and a moderate-change
band. The output remains in the existing sampled color domain (output code values for 8/10-bit
formats, linear for scRGB), with no `[0,1]` clamp, preserving HDR highlights and wide-gamut
negatives. Inputs per texel (12
channels; `scripts/framegen-net-train.py` mirrors them exactly and the two must stay in lockstep):
both directions' flow/conf/round-trip error, destination luma, source luma warped along the field
(the photometric evidence), the B4-style magnitude-normalized residual, and a task-conditioning
flag (reserved). Refining the reverse field is the same problem with the lumas swapped, so **one
set of weights serves both dispatches** (binding symmetry, the fbcheck trick). Weights ride a
2048-wide `R32F` texture uploaded once via a mappable staging image. The forward consistency pass
already produces both checked fields, so learned forward prediction does not require `_BIDIR`.

**Causal training** is the same self-supervision B4 measures: warp the source luma along the refined
flow; each texel pays either the warp error (× refined confidence) or the fallback error
(× complement), plus a small confidence-independent flow term so rejected vectors remain
trainable. Confidence can rise only where the final corrected vector reconstructs the observed
pair and beats holding (the gate below). Capture with `GAMESCOPE_FRAMEGEN_RECORD`, train with
`scripts/framegen-net-train.py` (numpy-only, CPU, minutes; see [scripts/README.md](../scripts/README.md)), deploy the blob, and validate with the
live A/B the adapt log provides (net vs no-net `resid`/`bad%` on the same content).

**The "beats holding" confidence gate.** Lowering confidence is always allowed; raising it is
allowed only where the refined vector predicts the pair better than holding (zero motion).
`framegenEvidenceGate(refinedResid, heldResid)` in `src/shaders/framegen_evidence.h` is
`predicts · beatsHolding`: `predicts = 1 − smoothstep(0.03, 0.10, refinedResid)` fades the raise out
as the refined warp's normalized luma residual grows, and
`beatsHolding = clamp((heldResid − refinedResid) / max(heldResid, 0.03), 0, 1)` scales it by the
share of the zero-motion residual the vector removes, floored at the 0.03 "explained" level — on
static content any small vector "predicts" the pair, and a raise there would revive unconfident
vectors as drag. The same include holds `framegenLumaResid` (|a − b| over the brighter side, floored
at 1 code value), and both `cs_framegen_motion_net.comp` (inference) and
`cs_framegen_motion_net_train.comp` (online training) include it, so serving and training cannot
disagree about when a raise is allowed. Before commit `88083af` they could: the online trainer still
used the older gate, `predicts` alone, while inference and the offline trainer required a raise to
beat holding (a train/serve skew). `evidence_gate()` in `scripts/framegen-net-train.py` mirrors the
include. The conservative bidir service never raises confidence, so the gate only matters for
causal prediction.

**Bidir training deliberately has a different authority boundary.** It evaluates the fallback
competition on the raw checked field, sets both flow-output gradients to zero, forbids positive
confidence, and stops confidence backpropagation at W3: only the confidence output row/bias can
change. This retains in-situ scene adaptation without allowing an endpoint-only objective to alter
intermediate-time geometry or the shared representation. Only confidence-row weights and the bias are
allowed to move; the validation evidence is in Appendix A.

**Kernel shape** (measured on NVIDIA Blackwell; re-benchmark on the present GPU before tuning):
one fused dispatch per direction, 8×8 output tiles from 14×14 feature tiles (3-texel receptive
apron), activations and per-layer weights staged in LDS, everything vec4-packed so each MAC step
is a `dot()` (the scalar version was 2.2× slower — LDS load-instruction bound, not ALU). Two
measured dead ends: 128 threads/tile (worse; not occupancy-bound) and a position×channel-group
work split (2.3× worse; quadruples activation loads). It is the single most expensive per-real pass (costs in Appendix A), which is why it is
opt-in; a `packHalf2x16` LDS variant and a `VK_KHR_cooperative_matrix` path are the known next
steps if a bigger net earns them.

Per-resolution inference, training and Guided-focus costs are in Appendix A.

**In-situ learning (C2, `GAMESCOPE_FRAMEGEN_NET_ONLINE=1`).** The net keeps training *on the
framegen GPU while it serves*, so the correction function tracks the current scene instead of
whatever an offline blob was fit to. Causal mode trains flow+confidence; conservative bidir trains
only the confidence output row described above. Per real frame, riding the same batch command buffer: two
gradient dispatches (`cs_framegen_motion_net_train.comp`; each workgroup = one hash-placed 4×4
training tile, forward + full backprop in LDS, per-tile gradient written as one row of a slice
image — **no float atomics**, `VK_EXT_shader_atomic_float` isn't universal) and one optimizer
dispatch (`cs_framegen_motion_net_opt.comp`; one thread per parameter: slice-mean, Adam,
**decay toward the prior**, and an EMA publish). ~8M MACs per step — an order of magnitude below
the inference pass; `_NET_EVERY=N` strides it for weak GPUs. Training is recorded after the B4
scene verdict: a content cut writes zero to every gradient slice, so Adam sees no unlearnable
cross-scene motion (only its normal decay toward the prior). Three structural properties do the
safety work: training reads the RAW fields (the net never trains on its own output); inference
serves a slow EMA of the fast weights (the served function moves calmly — same
calmer-than-the-signal rule as the B4 EMAs); and the fast weights relax toward the prior every
step (~1-minute memory horizon), so the model can only stay away from its safe starting point by
continually re-earning the distance on fresh frames — a model that works *temporarily* for the
scene, structurally. The prior is, in order: a `GAMESCOPE_FRAMEGEN_NET_PROFILE=<path>` file if it
exists (which also gets the served weights written back every 1024 steps — persistent per-game
learning), the `_NET` blob, or a synthesized neutral init (He hidden layers, zero head). Validated
served weights are also retained across in-process resize/format resets and warm-start the recreated
GPU state, so a swapchain transition does not restart adaptation from the startup prior. Observed
live: from a neutral prior on vkmark, the confidence head recovers ~10 points of killed field
within seconds; warm-started from an offline blob it holds the blob's precision while tracking
(see `doc/framegen-proposals/README.md` for the measured A/B).

**Causal shading supervision (Guided only).** The old luma reservoir and preceding final field
reproject a third observed frame onto each forward training tile. Head four learns whether the
aligned `older→previous` luma trend actually persisted into the now-known current real frame. This
avoids the invalid two-frame objective that would reward blindly repeating every difference. Only
the fourth output row/bias receive this loss; it cannot perturb the shared trunk or the established
flow/confidence heads. Reverse tiles write an exact zero shading gradient. Detected scene cuts,
disabled B4/cut detection, stale frame IDs, a same-real-pair deadline/refill, missing resources,
cheaper pipelines, and bidir all disable the objective. GSFR v3 activates the head;
v1/v2 blobs are migrated only after forcibly zeroing their formerly undefined
fourth row and bias. `src/framegen/net_profile.hpp` centralizes that metadata,
finite-weight validation, and compatibility transform; the renderer retains
profile state and worker-thread lifetime, while `src/framegen/atomic_file.cpp`
provides the format-agnostic commit primitive.

Optimizer robustness is two-layered. The GPU rejects non-finite gradient slices, clamps each
parameter gradient to the necessary `[-1,1]` bound implied by the offline trainer's global-norm
clip, and locally resets any poisoned Adam scalar to its finite prior. The CPU readback still
validates every served weight before persistence; on failure it forces both optimizer state and
the served weight texture through the prior upload before the next inference, closing the old
one-batch "detected but still served" window.

### 3.6 Prior art — how the algorithm maps to the literature

Everything above is home-grown, but each piece has a recognizable ancestor in
the frame-generation literature. The table maps our components to the closest
published method and states how ours differs; full citations (venue + DOI/arXiv,
each primary-source cross-checked) live in
[`research-framegen.md`](research-framegen.md), and
[proposal #07](framegen-proposals/07-frames-only-sota-alignment.md) argues the
alignment and the remaining gaps. We are **frames-only** — no engine motion
vectors, depth, or G-buffers — so the engine-integrated methods below are
*analogs, not ports*.

| Our component | Closest published method | How ours differs |
|---|---|---|
| Forward extrapolation as the zero-latency default (§0, §3.1) | **GFFE** (Wu et al., ACM TOG 2024, arXiv 2406.18551, DOI 10.1145/3687923); frames-only baseline **DMVFN** (Hu et al., CVPR 2023, arXiv 2303.09875) | GFFE uses rendered depth+MVs and a layered background buffer; we have neither. DMVFN is the closest frames-only peer. |
| Pyramidal SAD block-matcher (§3.2, stage 2) | **AMD FSR 3** FidelityFX Optical Flow (GPUOpen): 8×8 blocks, 24×24 SAD search, luma-pyramid coarse-to-fine | Same classical construction; learned flow (**RIFE**; **SEA-RAFT**, ECCV 2024, arXiv 2405.14793) is the ML alternative we don't run. |
| Two-source agreement / fallback arbitration (§3.2, stage 3) | **FSR 3**: blends optical-flow vs MV interpolation, *preferring the better color match* (GPUOpen) | We arbitrate per-pixel between the motion gather and the fallback (by default a confidence-scaled hold of the real frame, bounded extrapolation in the crossfade variant), by the same color-match logic. |
| Forward-backward consistency check (§3.2, stage 2.5) | Classical **forward-backward / left-right flow-consistency** occlusion detection (standard across optical-flow & VFI) | Standard technique; no single survey paper to cite. |
| Predict quadratic acceleration (§3.2.1) | **Mob-FGSR** (Yang et al., SIGGRAPH 2024, DOI 10.1145/3641519.3657424): quadratic / uniform-acceleration motion | Mob-FGSR forms it in *world space* from depth+MVs; we form it in *screen space* from two consecutive motion fields. |
| Guided reconstruction (§3.2.1) | Motion-boundary ownership from **dense optical flow** (research §2); edge-aware joint upsampling | Local contribution: a hypothesis-testing software approximation — no flow net or vendor block. |
| Bidirectional interpolation (§3.3) | The **VFI** family — **RIFE** (Huang et al., ECCV 2022, arXiv 2011.06294), **FILM** (Reda et al., ECCV 2022, arXiv 2202.04901) | The interpolation regime and its intrinsic ≥1-interval latency (§0). Ours is a hand-written warp-blend, not a learned VFI net. |
| Learned refiner + shading focus, Stage C (§3.5) | **GFFE**'s lightweight correction-CNN template (its Shading Correction Network); self-supervision cf. **FILM/RIFE** | Main heads refine the *motion field*; Guided's fourth head gates a bounded analytic color trend. Unlike GFFE SCN, it has no output-res feature warp or invented RGB. |
| B4 adaptation + C2 in-situ learning (§3.4, §3.5) | Online / test-time adaptation | No direct frames-only precedent in the surveyed work; home-grown. |

The prototypes cross-reference the literature too: **#02** base-layer late
UI/cursor composite is the survey's UI-compositing recommendation (research §5;
cf. DLSS 4 UI handling, FSR reactive masks); **#03** optical-flow donor is the
DLSS 3 hardware-OFA route — noting DLSS 4 *replaced* the OFA with a learned flow
net (research §4), so a learned front-end (**SEA-RAFT** / **NeuFlow v2**, §2) is
the strategic endpoint; **#01/#06** deadline pacing echoes DLSS 4's hardware flip
metering (§4).

---

## 4. The decision state machine

`framegen_record_real_frame` resolves the whole decision once per real frame. It first
filters **overlay-only repaints** by base-layer pointer identity — a MangoHud tick or
fading notification re-composites the same game content and must not poison the pacing measurement —
then invalidates history on scene change (§4.5), then decides.

### 4.1 THE structural fact: two control regimes selected by `hasFramegenQueue()`

This is the single biggest newcomer trap. There are **two control regimes plus a classic A/B variant**, selected by whether the GPU gives gamescope a second queue. Find yours first: if the startup line `framegen: enabled ...` ends in `(dedicated queue)`, read 4.1.1; if not, read 4.2.

| Regime | When | Scheduler | How to confirm |
|---|---|---|---|
| Dedicated queue (default) | The GPU exposes a second queue in the queue family gamescope uses | Causal deadline scheduler: one disposable generated slot at a time (4.1.1) | Startup log `framegen: enabled mode=... multiplier=N (dedicated queue)` and `framegen: causal fixed-refresh deadline scheduling active by default` |
| Shared queue | Single-queue GPU, or `GAMESCOPE_FRAMEGEN_SINGLE_QUEUE=1` | Leaky-bucket gate, at most 1 generated frame per real frame (4.2) | The `enabled` line has no `(dedicated queue)` suffix |
| Dedicated queue, `GAMESCOPE_FRAMEGEN_CLASSIC=1` | Validation and A/B only | Gap-rounded batches (4.3) | `(dedicated queue)` suffix, but no `causal fixed-refresh` line |

The regimes in words:

- **DEDICATED queue** — causal fixed-refresh scheduling is deadline-driven by default. It plans at
  most one disposable generated target against the absolute display grid, admits it from predicted
  source readiness and measured cost, and replans after feedback, consumption, or an honest repeat.
  `GAMESCOPE_FRAMEGEN_CLASSIC=1` retains the old full-batch behavior for validation only.
- **SHARED queue** (single-queue family, or `GAMESCOPE_FRAMEGEN_SINGLE_QUEUE`) — the **leaky-bucket
   admission gate** (§4.2) requires a proven empty vblank + 2 stable frames; `nGenerate` is
  **hard-capped to 1** to bound head-of-line work in front of the next composite. It uses the same
  timestamp query pool and deadline ladder, keyed to the scratch timeline rather than the dedicated
  framegen timeline; reactive supersede/discard remains the final safety net.

#### 4.1.1 Default causal fixed-cadence behavior

Without bidirectional mode, on a dedicated framegen queue, fixed-refresh scheduling uses the
deadline timeline and never delays a real frame. For each refresh slot the scheduler uses the
renderer buffer's acquire-ready timestamps (`commit_t::present_time`) to learn cadence, bounded
trend and recent late error. It generates one disposable forward-predicted backup only when the
next real frame is not confidently due before the compositor's exact wake deadline. At
presentation the order stays: real frame, ready generated frame, then hardware repeat (§2 step 8).

Per vblank the sequence is:

1. If the next real frame is confidently due before the wake deadline, plan nothing; the real frame will take the slot.
2. Otherwise plan one disposable generated slot for the exact display target and submit it to the framegen queue.
3. At the vblank, a ready slot is consumed. A slot that is not ready is dropped and the hardware repeats the last image.
4. Feedback from the flip (consumed, late or repeated) updates the cadence predictor, and the scheduler replans for the next vblank.
In this mode the multiplier N only sizes buffers (generated pool 2N, output ring 4+2N). It does
not set how many frames are made: the scheduler fills every free display slot until the next real
frame (`plan_next_causal_slot`), up to the forward-prediction cap. A game that leaves no free slot
is dormant. Bidirectional mode differs: N caps interpolations at N-1 per real pair (3.3).

**Native DRM commit lead.** The generated-only KMS commit deadline uses a learned minimum viable
lead. Successful real commits with a lead below one refresh interval feed a low-water mark that
relaxes upward by at most 1/64 refresh per proof (`update_minimum_viable_present_lead`,
`src/framegen/deadline.hpp`), so an isolated low sample expires. The deadline adds a safety margin
of one tenth of a refresh (`ulPresentMarginNs = ulVblankNs / 10u` in `src/rendervulkan.cpp`).
During warm-up the habitual commit-lead average is used instead. A late generated flip backs the
learned lead off by 1/8 refresh, never beyond the habitual average, and logs `minimum viable
commit lead backed off`. The back-off applies only with the causal scheduler active, outside
VRR-hybrid and bidir. Nested backends keep their existing pacing.

**Selecting and confirming it.** Causal mode is low latency; bidirectional mode (§3.3) waits for
the next real frame and usually has better endpoint evidence. Leave `GAMESCOPE_FRAMEGEN_BIDIR`
unset to get causal mode. The startup log must contain `causal fixed-refresh deadline scheduling
active by default`; if it reports bidirectional interpolation instead, a conflicting setting is
still in the environment and the run is not a causal comparison. For a clean causal comparison
clear the scheduling switches too. To also compare image quality like for like, clear the learned-net variables.

```bash
# scheduling-affecting
unset GAMESCOPE_FRAMEGEN_BIDIR GAMESCOPE_FRAMEGEN_VRR_HYBRID GAMESCOPE_FRAMEGEN_BASE \
      GAMESCOPE_FRAMEGEN_RECORD_COLOR
# quality-affecting (does not change the scheduler)
unset GAMESCOPE_FRAMEGEN_NET_ONLINE GAMESCOPE_FRAMEGEN_NET_PROFILE
```

The timing predictor adapts in memory only: it has no profile file and adds no
Vulkan dispatch, and it is independent of image-model training (§3.5; continue in-situ learning
with a writable `GAMESCOPE_FRAMEGEN_NET_PROFILE`).

For a short scheduling trace set `GAMESCOPE_FRAMEGEN_DEBUG_EVERY=1`, and use `60` for visual
judgment so logging does not become CPU jitter. Nested Wayland validates execution and dual-GPU
import; native DRM is required to judge KMS phase accuracy, input latency, VRR interaction and
final scanout pacing.

### 4.2 Per-frame generate/dormant gate

> Applies only to the **shared queue** and to `GAMESCOPE_FRAMEGEN_CLASSIC=1`. The default scheduler is 4.1.1. Sections 4.2 and 4.3 are the legacy and fallback scheduling logic.

Signals: `bLeavesEmptyVblank` (`prev!=0 && now-prev ≥ 1.5·interval`); `bGpuHasHeadroom =
hasCompletedFramegen(lastGeneratedSeqNo)` (the oversubscription guard → **at most one batch in
flight**; `hasCompletedFramegen(0)` is trivially true so the first frame is never blocked);
`bGeneratable = bLeavesEmptyVblank && bGpuHasHeadroom`; `bCanSpeculate = g_device.hasFramegenQueue() && bGpuHasHeadroom && ulPrevRealFrameTimeNs != 0` (true only on a dedicated queue). Final gate
`bShouldGenerate = bCanSpeculate || bReactiveReady`. These gates remain authoritative for the
shared queue and classic fallback. The dedicated deadline path adds per-target source admission and
never queues more than one causal slot.

**Leaky bucket:** `nStableFrames` is updated by `update_cadence_confidence()` and saturates in
`[0, k_uCadenceConfidenceRequired=4]`; generatable → `+k_uCadenceConfidenceGain=2` (clamp 4), else →
`−k_uCadenceConfidenceLeak=1` (floor 0). All three constants and both functions live in
`src/framegen/scheduling.hpp`. `bReactiveReady` is `reactive_generation_ready()`, i.e.
`bGeneratable && nStableFrames ≥ 4` → needs **2 consecutive** generatable
frames from cold; leak(1) < gain(2) so a single jittered interval never disarms an established
stream. Reset to 0 on invalidation. **Fully bypassed on the dedicated queue.**

### 4.3 Classic multiplier x2/x3/x4 adaptation

> Applies only to the **shared queue** and to `GAMESCOPE_FRAMEGEN_CLASSIC=1`. The default scheduler is 4.1.1, where the multiplier is a resource ceiling rather than a batch size.

`nMeasuredGapVblanks = round((now−prev)/interval)`, min 1. `nGapVblanks = bCanSpeculate ?
max(measured, max(2, eff.multiplier)) : measured` — the dedicated queue plans ≥ the multiplier's
slots even after a fast interval. `nGenerate = min(gap−1, max(1, eff.multiplier−1))` — **one fewer
than the whole-vblank gap** (the final slot *is* the next real frame). Shared-queue caps to 1.
Per slot in the retained classic `framegen_submit_batch`: `phase = k/nGapVblanks`; `strength = clamp(phase ·
(g_flFramegenStrength / 0.5), 0, k_flFramegenMaxForwardStrength=1.5)`. `strength 0.5` reproduces the
classic x2 half-step. Production causal and bidir phases instead come directly from their exact
display-target timestamps.

### 4.4 Deadline degradation and recovery ladder (#04)

Evaluated **only on frames that actually generate** (after the dormant early-return), so an
idle/dormant stretch never moves the rung.
The stateless rung transformation lives in `src/framegen/policy.hpp`; Vulkan timing and the
scene-local rung counter remain owned by `rendervulkan.cpp`.

- `deadline = vblankInterval · k_uDeadlinePercent(85) / 100` (`src/framegen/scheduling.hpp`).
- `ulCurRungCostNs = framegenRungCostNs(nDegradeSteps, nCurGenForLadder)` — the live-measured GPU
  cost of the current config's batch shape (per-(rung, gen-count), 7/8 EMA; §6).
- Guard: `nMaxDegradeSteps>0 && cost!=0 && samples>=3 && steps<max`. Three completed
  batches of that exact `(rung, generated-count)` shape keep a cold or anomalous first sample from
  making an irreversible scene-local pipeline decision.
- If `nDegradeHold>0`: decrement only (cooldown — burns down even during a dip below deadline; it is
  **wall-frames, not over-budget-frames**).
- Else if `cost > deadline`: peek `nextEff = framegen_effective_config(steps+1)` and **step only if
  it truly reduces work** (`nextEff.mode != curEff.mode || nextEff.pipeline != curEff.pipeline || nNextGen < nCurGen`) — a
  pure multiplier notch that doesn't lower the generated count this gap would drop work for zero
  GPU saving. On step: `nDegradeSteps++`, `nDegradeHold = k_uDeadlineHoldFrames(4)`.

**The ladder on one line:**

```text
ceiling (what you asked for) -> guided -> predict -> learned -> checked -> warp -> extrapolate
  -> (classic/shared/bidir only) multiplier xN -> x2   (floor: it never stops generating)
  (in bidir the mode rung goes to the blend crossfade, not extrapolate (3.3))
step down: when the measured cost exceeds the deadline; hold 4 decisions after each step
step up:   one rung after 90 consecutive large-headroom decisions
```

The multiplier rungs apply only to the classic, shared-queue and bidir paths. In default forward causal mode they remove no work; the real steps there are the pipeline and mode rungs. The multiplier is not a performance lever.

Recovery is switched off on the bidir and VRR-hybrid paths (`framegen_recovery_active_for_path()`), even though `GAMESCOPE_FRAMEGEN_RECOVER` defaults to on. With `GAMESCOPE_FRAMEGEN_HUD=1` the current rung is visible in the `ladder` row. Illustrative example (the numbers depend on your settings):

```text
ladder   learned/motion rung 1/3  hold
```

`hold` means gamescope just stepped down and is waiting; `full` means no step-down; `watching` means it is measuring headroom for a step up (all states are in the [HUD rows](framegen-reference.md#hud-rows)).

**Rung ordering** (`framegen_effective_config`): applies `n` degradations to the ceiling
`{g_eFramegenMode, g_eFramegenPipeline, max(2, g_nFramegenMultiplier)}`. Motion walks
**guided → predict → learned → checked → warp → extrapolate**, beginning wherever the user set the ceiling, then
the multiplier steps toward x2. `framegen_max_degrade_steps()` is a thin wrapper over `gamescope::framegen::max_degrade_steps` (`src/framegen/policy.hpp`), which counts the selected pipeline rank plus
the final motion-to-extrapolate step and multiplier notches. There is deliberately **no "stop generating" rung** — the ladder always keeps
generating (so its GPU-time input never starves); the genuine "even x2 overruns" case is left to the
reactive pacing gate. It **never mutates** `g_eFramegenMode`/`g_eFramegenPipeline`/`g_nFramegenMultiplier`
(those are the user's ceiling and size the pool); it only reads an *effective* config. The selected
rung remains monotonic throughout each post-step hold. Afterward, 90 consecutive large-headroom
decisions may re-probe exactly one adjacent richer rung, followed by probation and scene-local
back-off. Native fixed-refresh recovery measures that headroom against full slot capacity; a
phase-late decision does not erase the streak, and one isolated miss does not degrade.

### 4.5 History invalidation and priming

`vulkan_framegen_invalidate_history` drops history and resets `nStableFrames=0`,
`nDegradeSteps=0`, the learned rung costs (`framegenResetRungCosts`), and the retained ring slots on:
focus change, layer-count change, output-EOTF change (SDR↔HDR without a format change), long
frame-gap (`k_ulFramegenMaxRealFrameGapNs`, a stall/load-screen), resolution/format change. **The
first real frame after any invalidation only primes** (sets `valid=true` and returns
without generating) — so **every scene change costs one non-generating frame** before generation can
resume.

### 4.6 State diagram

```
        invalidate_history (focus / EOTF / layer-count / gap / resolution / format)
        ─── resets nStableFrames=0, nDegradeSteps=0, rung costs, history ───┐
                                                                            │
                                                                    [PRIME] │ (1 non-generating frame)
                                                                            ▼
  DORMANT ──(bLeavesEmptyVblank && bGpuHasHeadroom, x2 from cold)──► GENERATING
    ▲   ▲                                                              │  │
    │   └──(shared queue) nStableFrames leaks < 4 ────────────────────┘  │ every generating frame:
    │                                                                     │   ladder may step DOWN
    └──(!bGpuHasHeadroom: prior batch in flight)── BUSY ──────────────────┤   (guided→predict→learned→
                                                                          │    checked→warp→extrapolate
                                                                          │    →x3→x2 on classic/shared/bidir
                                                                          │    only; no dormant rung)
                                                                          │   ladder may also step UP:
                                                                          │   hysteretic recovery to the
                                                                          │   richest rung whose measured
                                                                          │   cost fits the slot budget
  Dedicated causal queue: source prediction + absolute target admission  │
  → at most one disposable slot; feedback/consume/repeat replans          ▼
  Classic A/B only: refill_idle extends one gap-rounded slot on drain
```

---

## 5. Latency & correctness invariants (with enforcement points)

The four named invariants of Section 0, with the code that enforces each. The numbers 1 to 4 are kept so older references still resolve; the names are the stable handle.

1. **NEVER-QUEUED: real frames never delayed** *(default path; bidir §3.3 is the deliberate exception; the forced
   composite's ≈ +2 ms is measured below)*. Generation
   is a *separate* command buffer submitted *after* composite; the real present waits only on the
   composite `sequence`. On a dedicated queue a slow generation on queue 1 cannot
   head-of-line-block composite on the in-order queue 0. Push constants (≤64 bytes, with exact C++
   size/offset assertions against the GLSL contract)
   instead of the shared upload arena keep framegen from racing the composite host bump allocator across
   queues. **Under bidir** this still holds at the *generation* level (generation never blocks the
   composite), but the *presentation* timeline intentionally holds each real frame for the target delay D so the
   interpolations before it can show first — a scheduling choice, not a generation stall.
2. **NEVER-LATE-GENERATED: generated frames never waited on.** All completion checks are **non-blocking**
   `hasCompletedFramegen` peeks (`GetSemaphoreCounterValue ≥ seq`): the present-decision peek,
   oversubscription guard, and ring-advance read pin all use the same rule. `consume` drops the front slot if incomplete rather than stalling
   scanout. A not-ready slot becomes a free HW repeat.
3. **REAL-DISCARDS-PENDING: a real frame latching discards the ENTIRE pending batch.**
   `vulkan_framegen_discard_generated_frame` clears the whole `pending` vector on
   `superseded_by_real_frame`, `superseded_by_repaint`, or `generation_too_slow` (the reasons emitted in `src/steamcompmgr.cpp`).
   Correctness of the split between `steamcompmgr` and the backend hinges on discard emptying
   `pending` before `paint_all`.
   Bidir is the explicit exception: its queued reals are a timestamp-ordered deadline timeline and cannot be
   discarded. Under overload it sheds oldest interpolation entries only, reserves space for the
   incoming real endpoint, and reduces new interpolation admission if queued reals already consume
   the `2*(multiplier+1)` bound. The cap is applied to the post-append batch, not merely the old size.
4. **FORWARD-ONLY: causal generation extrapolates only.** Per-slot `strength = phase ·
   (g_flFramegenStrength / 0.5)` is a forward coefficient capped at 1.5. Bidir alone interpolates
   between real endpoints; the debug `blend` shader remains debug-only.

**Supporting invariants:** only base-layer commits count as real; classic fixed-slot framegen
suppresses VRR and tearing, while the explicit VRR-hybrid path is the sole VRR exception. A
display opportunity is exactly one of real, generated, or hardware repeat; overlay-only
repaints are filtered by base-layer pointer identity (Section 4). When the arbiter in
`steamcompmgr.cpp` paints for another reason while a prediction is pending, the prediction is
discarded as `superseded_by_repaint`, or as `generation_too_slow` if it was not yet ready. **One batch in flight** (`bGpuHasHeadroom`) is the single assumption making three
lockless structures safe (§6).

**What framegen costs a real frame (measured).** NEVER-QUEUED means a real frame never queues behind
generated work; it does not make framegen free. With framegen on, a real frame takes about **+2 ms**
longer from source-ready to flip than with framegen off, at p50 and p95 (headless host, dual GPU,
debug and optimized builds alike; GravityMark 1M and Xonotic; Appendix A). All of it sits after the
latch, in the `rec_flip_ms_*` half of the latency split (§8). Framegen off with
`gamescope_composite_force=1` (the `composite_force` convar through its environment override)
matches framegen on, so the cost is the full composite that both backends force for every real
frame while framegen is on (`bNeedsFullComposite |= vulkan_framegen_is_enabled()` in
`DRMBackend.cpp` and `WaylandBackend.cpp`, §2 step 3); framegen's own work adds 0–0.5 ms.
**The composite stays.** Presenting real frames directly and compositing the history off the
critical path was designed and rejected after review, for three reasons:

- The client image is exclusive with FOREIGN queue-ownership transfers (`insertBarrier`,
  `emitExternalAcquireBarriers`), so a host/KMS read concurrent with the history composite is not
  portable.
- The dual-GPU staging path (`stage_non_device_local_base_imports`) replaces the commit's texture
  with a non-flippable staging image.
- Recording history after the commit would break the present tags and timing anchors
  (`vulkan_framegen_begin_present`).

**Composite request after a generated batch (`GAMESCOPE_FRAMEGEN_BATCH_REPAINT`, opt-in).** After a
generated batch is submitted the compositor thread is only woken (`nudge_steamcompmgr()`; under an
active VRR hybrid the mid-interval timer is the only trigger). A wake is enough: under VRR every
wake arbitrates, and under a fixed refresh only the vblank timer may present a generated frame.
Before commit `518498f` the batch called `force_repaint()`, which also raises a composite request;
`GAMESCOPE_FRAMEGEN_BATCH_REPAINT=1` restores that, in causal mode only (not bidir, not VRR hybrid).
It trades the latency tail for real-frame throughput: on a GPU-bound client (GravityMark 1M,
2048x1280@120, headless) real frames rose from 34 to 44 FPS with steadier flips, but the
source-to-flip p95 grew from about 11 to about 25 ms, and in a blind A/B the viewer rated it laggier,
so it is off by default (Appendix A). It is also why an August build, which predates `518498f`,
"felt smoother but laggier".

---

## 6. Resource architecture

- **Zero-copy history.** `previousReal`/`currentReal` are `Rc<CVulkanTexture>` **aliasing composite
  ring slots**, not copies — saves a full-resolution copy per real
  frame (~66 MB/frame at 4K) on the weak card. The framegen ring grows from the classic **3**
  slots to **8/10/12 at x2/x3/x4** (`output_ring_size_for_multiplier`): two history endpoints
  plus multiplier-scaled nested-Wayland release depth and real-composite progress. Cheaper pipelines
  retain the smaller memory footprint. Reuse is governed by `IsInUse()`, not the size estimate;
  partial-overlay aliases remain suppressed while framegen is active.
- **Generated-frame scanout pool** `framegenOutputImages`, sized **`2·g_nFramegenMultiplier`**
  with a rolling cursor `% pool.size()`. It is **strictly disjoint** from the composite ring: generated
  frames flip directly and outlive generation. Allocation admits only a texture whose
  `CVulkanTexture::IsInUse()` is false: public refs cover pending slots and Vulkan command-buffer
  lifetime, while backend-fb refs cover DRM request/visible sets and Wayland compositor acquire to
  `wl_buffer.release`. If the backend retains the whole pool, the batch is shortened (or skipped)
  and scanout repeats; an acquired image is never rewritten. The three-image late-composite pool
  in base-layer mode uses the same ownership test.
- **Nested-Wayland lifetime.** Core `wl_buffer.release` means that the compositor no longer uses
  the buffer and its backing storage is reusable. Reattaching the same still-busy buffer does not
  create a countable lifetime token: `CWaylandFb` holds one aggregate compositor reference from
  the first acquire transition until release. Treating commits as independent uses leaks backend
  references when the compositor emits the aggregate release and eventually exhausts the rings.
  A redundant release is diagnostic-only and never decrements an absent lifetime reference.
- **Dedicated compute queue.** `createDevice` requests `queueCount=2` from `m_queueFamily` when
  `framegen && backend-supported && queueCount≥2 && !GAMESCOPE_FRAMEGEN_SINGLE_QUEUE`,
  retrieves index 1 → `m_framegenQueue`. Both queues share `REALTIME` global priority (Vulkan has no
  intra-family priority — this is a **decoupling** win, not a priority one). Single-family topology
  still works. Requested only when framegen is enabled, so other sessions are unaffected.
- **Own framegen timeline** `m_framegenTimeline` (monotonic `m_framegenSeqNo`); graceful downgrade to
  the shared queue if creation fails. Composite `waitIdle` explicitly drains it.
- **Per-queue descriptor ring.** `m_framegenDescriptorSets` (`k_uFramegenDescriptorSets`) alongside the composite `m_descriptorSets`
  (`k_uMaxConcurrentSubmits * 3`), both counted into one pool (`nTotalSets`); routed by `CVulkanCmdBuffer::m_bFramegen` set by `markFramegen()`. **No completion
  tracking** — a bare modulo counter, safe *only* because of the one-batch-in-flight guard.
- **Command-buffer split.** Generation into its own `CVulkanCmdBuffer` from the shared pool, submitted
  via `submitFramegen` in `src/framegen/device.cpp`: the second-queue path uses raw `vk.QueueSubmit`
  plus the dedicated framegen timeline, while the fallback uses normal `submit` plus the scratch timeline. Each path recycles
  through the command buffer's owning timeline (`framegenGarbageCollect` handles the dedicated path).
- **Finalized motion-field cache.** `uMotionFieldFrameId` identifies the checked/refined/trust-scaled
  field still resident in `mvField`/`mvFieldNet`. Same-pair deadline slots and classic idle refills reuse it and run only
  their warps; they do not repeat estimation, B4, net inference or online training. A new preparation,
  held-out probe, resource rebuild or history invalidation clears the identity. The next consecutive
  same-pipeline causal pair shifts it to `mvFieldHistory` before overwriting the working field.
- **Cross-queue WAR pin.** `genReadA/B/genReadSeqNo` pin history slots against ring reuse
  until `hasCompletedFramegen(genReadSeqNo)`. The compositor explicitly releases completed pins
  before target acquisition; unfinished pins survive history invalidation. Ring availability is
  checked dynamically because backend ownership depth is not a fixed Vulkan property.
- **Reset lifetime fence.** `lastFramegenWorkSeqNo` tracks every framegen-path submission, including
  descriptor-free base-history copies that intentionally do not affect the generation headroom gate.
  A resize, format change or base-mode transition retires that token before releasing history, pools,
  descriptor/query state or ring pins; this wait exists only on exceptional reset paths. A completed
  online-weight readback is consumed before the reset flush, preserving the newest in-situ step.
- **Timestamp query ring.** Depth 4, created once for the device lifetime if
  `timestampValidBits>0 && timestampPeriod!=0` on the
  selected queue family; TOP/BOTTOM_OF_PIPE brackets. The dedicated-queue composite wait is scoped to
  `ALL_COMMANDS`, so the opening timestamp cannot execute before the semaphore and charge upstream
  composite wait time to framegen. The single-queue fallback keys the same query associations to the
  scratch timeline, so low-queue-count devices retain deadline adaptation. Readback **without**
  `WAIT_BIT` performs modular subtraction in
  the queue family's `timestampValidBits` domain, then folds into `m_aFramegenRungCostNs[8][4]` via
  a 7/8 EMA keyed by **batch shape** (so x2-gaps don't inherit x4-batch timings). A saturating sample
  count supplies the ladder's three-sample warm-up. Scene reset also drops outstanding
  query-to-rung associations, so a pre-cut batch that completes late cannot seed the new scene.
  Individual query pairs are reset in the command buffer immediately before reuse; resize and
  format changes do not destroy or recreate the pool.

**The one assumption behind three lockless structures:** "at most one framegen batch in flight"
(`bGpuHasHeadroom`) is what makes the 30-slot descriptor ring, the depth-4 timestamp ring, and the
`genReadA/B` pins all safe without completion tracking. Any change allowing two concurrent batches
breaks all three.

Central state: `FramegenHistory_t g_framegenHistory`,
`FramegenMotionResources_t g_framegenMotion`, `FramegenDispatch_t g_framegenDispatch` — all touched
only from the compositor thread → no locking.

---

## 7. Vendor / perf model

The unifying truth: **bandwidth-bound, ALU-light.** LDS apron helps only on cache-poor GPUs; fp16
helps only on integer targets and its upside is capped because the apron storage stays fp32; the
no-LDS `direct` shader wins on large-L2 / Infinity-Cache parts where the neighbor cross is already a
cache hit.

- **Dispatch selection** `framegen_dispatch_for_format` — single-slot **format-keyed
  cache**, computed once per drmFormat. Its pure capability mapping lives in
  `src/framegen/dispatch_policy.hpp`: `useFp16 = supportsShaderFloat16 && !isFloatFormat`;
  `useR16FLuma` picks the luma format + `MOTION_LUMA_PAIR` vs `_RGBA`; `motionSupported = ABGR16161616F
  sampled+storage`. **Measured vendor override:** `vendorID==0x10DE` currently forces
  `useFp16=false` and `extrapolate=DIRECT` — but **deliberately leaves `extrapolatePair` on its
  independently capability-selected LDS precision** (no measured direct-pair shader yet).
- **Microbenchmark** `vulkan_framegen_benchmark` — times the real production helpers over
  a 3-res × 2-format sweep {1080p/1440p/2160p} × {ABGR2101010 int, ABGR16161616F float}, `nIters=200`,
  own 2-query pool, `waitIdle`-serialized; marks the production-selected variant with `(*)` and,
  when online ML is enabled, reports inference and training separately. Empirical
  basis for the NVIDIA=direct choice. The Guided versus Predict warp premium for the
  full-resolution boundary verdict is recorded in Appendix A. It uses the selected queue family's
  `timestampValidBits` and modular wrap subtraction, matching the live estimator.
- **Live measurement → ladder** (dedicated or shared queue): timestamp ring → non-blocking readback
  in `framegenGarbageCollect` → 7/8 EMA per (rung, gen-count). Cached-field refill warps are not mixed
  into that EMA: it represents the next real pair's full setup batch rather than a cheap same-pair
  continuation.

---

## 8. Observability

- **`g_bFramegenDebug`** + **`FramegenDebugShouldLog`** in `src/main.cpp` (gate
  `<=1 || counter==1 || counter%every==0`, period `g_uFramegenDebugEvery` default 60) throttle 8
  rendervulkan call sites that emit the scheduler's live state: priming, overlay-only
  filtering, the `busy/stabilizing/dormant degrade=N/M` line, the
  `generated N frame(s) … mode=… gpu=X.XXms` line, and the real/generated/repeat slot
  classification in `steamcompmgr`.
- **`GAMESCOPE_FRAMEGEN_HUD={1,2}`** is the low-overhead on-screen view of the same existing
  metrics/state. It uses only the last closed five-second window, refresh and current mode/net
  globals; capture and screenshot composites skip it.
- **`GAMESCOPE_FRAMEGEN_METRICS=1`** logs one `framegen-metrics:` line per five-second window and a
  `TOTAL` line at exit (`framegen_metrics_log` in `src/rendervulkan.cpp`; the histogram types are in
  `src/framegen/metrics.hpp`). Present tags and flip feedback are tracked whenever framegen *or*
  metrics is on (`framegen_present_tracking_enabled`), so the line is also a framegen-off baseline.
  New fields are appended at the end of the line and existing ones are never reordered; the full
  list is in the [configuration reference](framegen-reference.md#metrics-log-fields). Two groups
  matter for the mechanisms in this document:
  - **Latency split.** `src_rec_ms_p50/p95` (source ready → first paint, the latch, of that real
    frame: the wait for the compositor to pick it up) and `rec_flip_ms_p50/p95` (latch → actual
    flip: composite and present) split `src_flip_ms_*` for live real frames. The record time is
    stamped by `framegen_metrics_note_source_ready` at the first paint of a real-frame id (a repeat
    paint of the same content keeps it), and `framegen_metrics_add_feedback` adds the samples
    (`realSourceToRecord`, `realRecordToFlip`) when the flip feedback arrives. Delayed (bidir) reals
    are not split; they keep `dreal_flip_ms_*`. The percentiles come from 1 ms buckets (256 ms full
    scale) and report a bucket's upper edge, so differences below a millisecond are not resolvable.
    The forced composite of §5 lands in `rec_flip_ms_*`.
  - **Quality meter (`q_*`).** The counters count *changed* texels: those where holding the previous
    frame would mispredict the real frame (the unwarped luma residual exceeds the same bar the probe
    uses for "mispredicting"), whatever the field says, not texels the field calls moving. A static
    background predicts itself, and a near-zero vector that the sub-texel fit nudged past the static
    bar is not motion evidence. A window reports ratios of sums, so every changed texel weighs the
    same (`MotionQualityCounts`, `cs_framegen_motion_stats.comp`). Values logged by builds before
    commit `5c776da` are not comparable.
- **`--debug-dual-gpu-route` / `g_bDebugDualGpuRoute`** is a distinct,
  per-frame, **un-throttled** channel for the cross-GPU import path — do not confuse it with the
  rate-limited framegen debug.
- **E2 full-colour probe** (`GAMESCOPE_FRAMEGEN_RECORD_COLOR`) is an offline image-quality
  instrument, not a present benchmark. `framegen_record_color_probe_real` forms exact held-out
  A/B/C triplets; `framegen_submit_planned` generates paired strengths 0/0.5/1 for the recorded
  occlusion or endpoint-trace sweep in one command buffer and copies them after
  `framegenTimestampEnd`; `scripts/framegen-color-eval.py` (see [scripts/README.md](../scripts/README.md)) grades every candidate against the same
  B and reads GSCF v1/v2/v3. The copy/write path is excluded from the algorithm GPU timestamp but
  synchronous host file I/O can visibly disturb the real-only capture cadence.

### Debugging by symptom

Start from what you see, then follow the log line or metric to the mechanism. Field meanings are in the [configuration reference](framegen-reference.md#metrics-log-fields).

| Symptom | Log or metric | Meaning | Read |
|---|---|---|---|
| `gen=0` in the metrics line after enabling | `framegen: enabled ...` line; `(dedicated queue)` suffix | History not primed yet, the game fills every refresh (dormant), or the backend does not support framegen (headless and OpenVR ignore `--experimental-framegen`) | 4.5, `IBackend::SupportsFramegen()` in Section 1 |
| Quality drops and stays low | HUD `ladder` row, `GAMESCOPE_FRAMEGEN_RECOVER` | Generation cost exceeded 85% of the vblank, so the ladder stepped down. With bidir or VRR hybrid, recovery is off | 4.4 |
| `bidir_delay_ms` jumps | metrics line | The delay controller moved E or D_ring | 3.3 (delay controller) |
| `dreal_drop` above 0 | metrics line, log | The bidir queue dropped delayed real frames (fast source or backlog) | 3.3 (delay controller) |
| Only some scenes behave differently with `_BIDIR` and `_BASE` | startup log | Per-scene choice: base-layer scenes run causal | 3.3 |
| Black screen across the GPU split | `--debug-dual-gpu-route` | bug-179, open; workaround `ENABLE_GAMESCOPE_WSI=0` | 2.1, 10 |

---

## 9. Roadmap (`doc/framegen-proposals/`)

| # | Proposal | Status |
|---|---|---|
| **01** | VRR hybrid — present the real frame VRR-style for the full latency win, schedule the generated frame with a timer-armed mid-interval atomic flip | **PROTOTYPE IMPLEMENTED** (`GAMESCOPE_FRAMEGEN_VRR_HYBRID=1`, dedicated queue + active VRR only). `steamcompmgr` keeps `allowVRR` enabled only while the hybrid is requested, and the DRM backend inherits the real frame's VRR state so the two flips agree. One generated frame at phase 0.5 is flipped by `g_FramegenMidTimer` and planned by `framegen_vrr_hybrid_submit`. Built on #06's source-cadence predictor. **Not yet validated on a VRR panel** (see the proposal status). |
| **02** | Base-layer generation + late overlay/cursor composite — generate on the pre-upscale game layer, composite UI on top (no HUD ghosting) | **PROTOTYPE IMPLEMENTED** (`GAMESCOPE_FRAMEGEN_BASE=1`, no dedicated-queue requirement). Generation runs on `layers[0].tex`; `framegen_base_present_composite` late-composites current overlays/cursor through `vulkan_composite`, so generated frames receive the full FSR and color-management pipeline. **Divergence:** the draft's bandwidth reduction no longer applies after zero-copy history; base mode adds a base-sized history copy per real frame and a full composite per presented generated frame. `framegen_base_layer_usable` falls back to output-space generation for video-underlay, YCbCr, ReShade, or unsupported storage formats. |
| **03** | dGPU optical-flow donor — offload motion estimation to the render GPU's `VK_NV_optical_flow` OFA, ship a small flow field over PCIe | **Aspirational** (longest horizon). Zero OFA symbols; needs a second Vulkan device gamescope has never had; cross-vendor timeline interop rated unreliable. Motion mode is the shipped fallback. |
| **04** | Timestamp-driven adaptive degradation | **IMPLEMENTED** (`a75bfbe`) but **divergent** from the proposal — rungs are motion-pipeline/mode/multiplier notches (not a pyramid table), use an 85% budget (not 0.6), and have a 2D per-(rung,gen-count) 7/8-EMA, three-sample cold-start guard, fixed 4-frame hold, and slow evidence-gated one-rung recovery with probation/back-off. Native fixed-refresh recovery uses full slot capacity and two-miss hysteresis. `VK_EXT_calibrated_timestamps` is unused. |
| **05** | Tile classification + `vkCmdDispatchIndirect` + SDMA static fill — generate only over moving tiles, fill static tiles on the transfer engine | **Aspirational** (deferred). No transfer-only queue discovery / classify shader exists; the doc admits it depends on transfer-queue discovery that isn't there yet. |
| **06** | Causal fixed-cadence scheduling — treat each refresh as a deadline, present a real frame when ready, otherwise use a one-slot forward prediction | **DEFAULT BEHAVIOR** on the dedicated queue. `commit_t::present_time` supplies source-ready observations; `CadencePredictorState` learns bounded period/trend/late error; the deadline planner compares protected arrival with the exact display wake, carries target timestamps through submission, and replans after feedback or repeat. The multiplier is a resource ceiling rather than a batch size. The shared-queue fallback remains conservative. |
| **07** | Frames-only SOTA alignment and validation | **BOUNDED IMPLEMENTATIONS COMPLETE FOR E1, E2, B, A, D.** E2 now captures exact full-resolution held-out colour with paired candidates; optional LPIPS is available, while DISTS/FvVDP remain external evaluation work. The confidence-only bidir ML authority boundary remains the production contract. |

---

## 10. Sharp edges / WIP / open bugs

Open items and standing caveats come first; closed items are kept at the end for context.

- **bug-179 (open): black screen across a cross-GPU split through the WSI layer.** With the game
  rendering on one GPU and gamescope on the other, the WSI layer path hangs and the screen stays
  black; the cause is not located. Workaround: `ENABLE_GAMESCOPE_WSI=0` makes the game present
  through Xwayland. This is the failure behind the "black screen across the GPU split" symptom in
  [Section 8](#debugging-by-symptom). gamescope sets `ENABLE_GAMESCOPE_WSI=1` only when the variable is unset
  (`setenv(..., 0)` in `src/steamcompmgr.cpp`), so an explicit `0` is respected.
- **bug-178 (open): flaky bidir smoke case.** The bidirectional case of the nested smoke harness
  (`scripts/framegen-nested-smoke.py`, see [framegen-testing.md](framegen-testing.md)) fails about
  one run in three because one frame lands 4.7 ms late. It predates the current work; the cause is
  not located.
- **Dense particle swarms are the weak case.** The default sharp fallback frays their edges (3.2,
  stage 3) and the opt-in /4 field looks blocky on them, so /4 stays opt-in.
- **Acceleration looks net negative in real games so far.** In Far Cry Primal `predict` felt choppy
  and only tied `warp` (better image, less snappy), and `guided`, which contains it, costs the most
  GPU time for a close second place behind `learned`.
- **Unexplained deadline misses.** `warp` and `checked` missed frame deadlines in Far Cry runs for a
  reason not yet found.
- **`learned` is not the CLI default (`warp` is).** Blind tests of `learned` on Xonotic and
  GravityMark, and of the sharp fallback on `learned`, are pending. Player-facing guidance:
  [Choosing settings](../README.md#choosing-settings).
- **Speculate-every-frame WIP (classic path only).** In the *classic* dedicated-queue regime
  `bCanSpeculate` still ignores `bLeavesEmptyVblank`, so it generates a full batch **every
  real frame even when the game hits refresh** — a framegen pass of wasted bandwidth per vblank on the
  weak card, worst with Motion. The flagged "skip when comfortably keeping up" to-do (`1b949f0`) is now
  **implemented in the deadline policies**: default causal scheduling uses trained source-ready
  deadline admission, while #01 VRR hybrid (`k_uVrrHybridKeepUpPercent=220`) skips
  the midpoint when predicted cadence cannot leave two panel-safe halves. The
  classic A/B path still has no guard.
- **NVIDIA direct-pair not implemented.** The vendor override only swaps the single-slot extrapolate;
  `extrapolatePair` stays on LDS, so NVIDIA x3/x4 batches still pay the apron and lose the ~30–37% win.
- **Vendor predicate too narrow.** Single hardcoded `vendorID==0x10DE`. **The actual weak-card targets
  (7900 XT / 5700 XT) are not special-cased** and run the un-benchmarked LDS+fp16 default; the direct
  shader's own header claims RDNA3's Infinity Cache benefits too — dispatcher and shader comment
  disagree on RDNA3. Extend the predicate once measured on those parts.
- **Config semantics.** Framegen strength and multiplier both use strict parsers plus fatal range validation.
  Every `GAMESCOPE_FRAMEGEN_*` on/off switch reads through `parse_bool_setting` (`src/framegen/settings.hpp`),
  so an invalid value keeps the default and logs once; upstream `env_to_bool` is untouched. The Stage-B knobs
  `_FB` / `_AGREE` / `_ADAPT` are default-on (`=0` to disable), and the numeric settings (`_FB_TOL`/`_NET_LR`
  floats, `_RECORD_MAX`/`_NET_EVERY` uints) use the strict numeric parsers; `_NET`/`_RECORD`/`_NET_PROFILE` are
  paths. The `GAMESCOPE_FRAMEGEN_*` env vars are listed in [framegen-reference.md](framegen-reference.md), not in `--help`.
- **Backend nuances.** The composite ring size is fixed at allocation — **3** slots without framegen
  (`k_uOutputRingSizeDefault`), **8/10/12** at x2/x3/x4 with it (`output_ring_size_for_multiplier`,
  `4 + 2·multiplier`) — a mid-session framegen enable/disable
  requires a full `vulkan_remake_output_images` (`waitIdle` + reset), *not* a live adjustment. The
  Wayland generated-frame present hard-nulls planes 1..7 (overlays/cursor dropped for a generated
  slot). After a generated batch the compositor is only nudged (`nudge_steamcompmgr()`; the
  `bShouldPaint` override already handles the paint decision); `force_repaint()` is the opt-in
  `GAMESCOPE_FRAMEGEN_BATCH_REPAINT` path (§5).
- **Shared-queue path is intentionally conservative** — `nGenerate` is forced to 1 to bound
  head-of-line blocking. It now retains timestamp-driven degradation as well as reactive discard.
- **Motion is all-or-nothing under pressure** — the ladder sheds Motion before any multiplier notch,
  so `Motion-x4 → Extrapolate-x4 → x3 → x2` with no Motion-x3/x2 rung.
- **Descriptor ring has no completion tracking** — safety rests entirely on the one-batch-in-flight
  guard; the `k_uFramegenDescriptorSets` size is dispatch-capacity margin, not a substitute for ownership tracking.
- **`nMaxDegradeSteps`** is threaded through `framegen_submit_batch`, `framegen_submit_planned` and `framegen_vrr_hybrid_submit`, where it appears only in the generation debug log. The ladder logic itself consumes it in `framegen_record_real_frame` (guard, recovery and recovery-blocked logs).
- **Idle-refill runaway** is bounded only by the 1.5 strength clamp + `nMaxSlots` cap + the 250 ms
  `idle_frame_gap` invalidation.
- **sRGB-view naming** — history is bound with `setTextureSrgb(true)`, which selects
  `CVulkanTexture::srgbView()`. Despite the name that view is created with the **non-sRGB** Vulkan
  format (`DRMFormatToVulkan(drmFormat, false)`), so the sample returns raw output code values with
  no hardware transfer-function decode; `linearView()` is the one built with the `_SRGB` format.
  The contract is therefore: for integer formats the motion estimator and its 0.08/0.40 suppression
  thresholds operate on **output code values**, matching the shader comments' "gamma-encoded [0,1]";
  for float scRGB targets the samples are linear.

**Resolved, for context**

- **NVIDIA render / AMD present failed at device creation** — fixed in `2a3a6c0`: the WSI layer forced
  `VK_PRESENT_MODE_MAILBOX_KHR` and chained swapchain-maintenance structs unconditionally, which made
  device creation fail on the NVIDIA proprietary driver with `VK_ERROR_INITIALIZATION_FAILED`. The
  layer now chains them only when the driver supports both the extension and the feature (§2 step 1).
- **Wobble** — fixed in `a75bfbe` (nested-refresh clock, box luma, sub-pixel SAD, selective clamp);
  extrapolate stays structurally juddery on fast motion → prefer motion mode for pans.
- **Translucency (vkmark "penguins")** — tested: framegen preserves opacity; the earlier "goes opaque"
  claim was disproven (numbers in Appendix A). The residual is edge judder and ghosting.

---

## Appendix A. Measurements (dated, non-contractual)

Point-in-time results kept out of the mechanics sections because they change with drivers and
hardware. None of them is a contract; re-measure before relying on a number. Reproduction tools are
listed in [scripts/README.md](../scripts/README.md) and [framegen-testing.md](framegen-testing.md).

- **Endpoint trace (3.3), GravityMark.** Across three 12-frame held-out GravityMark sets with
  user-driven camera movement, strength `0.5` won 26/36 exact pairs and improved pooled MAE
  `0.145880→0.145183`, SSIM `0.42542→0.42735`, and edge error `1.12068→1.11562`. A same-binary
  Radeon 890M timestamp sweep measured the 1440p XB30 baseline/trace pipelines at `0.571/0.583 ms`.
- **B4 probe cost (3.4).** On Radeon 890M: 20 µs at 1080p, 24–25 µs at 1440p and 39 µs at 4K per
  real frame.
- **Conservative bidir training (3.5).** In a 1M-asteroid x4 validation the legacy persisted profile
  produced heavy artifacts; no-net was very good; a neutral net was better; confidence-only ML was
  best. After 1812 online steps a binary profile comparison confirmed all 4499 non-confidence
  parameters unchanged; only 139 of 144 confidence-row weights plus the bias moved. The user profile
  was not modified during that A/B.
- **Net kernel (3.5), RTX PRO 500 (NVIDIA Blackwell).** 0.59/1.05/2.29 ms per real frame at
  1080p/1440p/2160p.
- **Radeon 890M, v3 online path (3.5).** Causal inference measured 0.567/1.008/2.184 ms per real at 1080p/1440p/2160p integer; the fixed 16-tile training + Adam + profile-readback copy measured 0.35–0.38 ms at every resolution. With a neutral (zero-focus) head, the isolated `GAMESCOPE_FRAMEGEN_SHADING=0` A/B measured 0.044/0.055/0.142 ms additional Guided-warp cost per generated frame at those resolutions. This is the cached low-res focus sample and branch floor; active-focus cost depends on mask density and is charged to Guided's normal timestamp/degradation rung.
- **Field copy for refill (2/6).** 4–9 µs per real at 1080p–4K on Radeon 890M.
- **Guided versus Predict warp (7), Radeon 890M, 200-dispatch mean.** 0.649 vs 0.587 ms at 1440p
  integer and 1.517 vs 1.411 ms at 2160p integer, a 4–11% premium for the full-resolution boundary
  verdict.
- **Translucency (vkmark "penguins").** Framegen preserved opacity (0.52 vs 0.50); an earlier "goes
  opaque" claim was disproven. The residual is edge judder and ghosting; a human framegen-off A/B is
  still needed to finalize.
- **Sharp fallback versus crossfade, warp cost (3.2), 2026-10-16/17.** Radeon 890M, per generated
  frame: the same GPU time at 1080p, and about 10% less with the sharp fallback than with the
  crossfade (`GAMESCOPE_FRAMEGEN_SHARP_FALLBACK=0`) at 1440p and 2160p.
- **Latency split, framegen off versus on (5, 8), 2026-10-16/17.** Headless host, dual GPU, debug and
  optimized builds alike; GravityMark 1M and Xonotic; source-to-flip (`src_flip_ms_*`) at p50 and
  p95. OFF direct (framegen off, no forced composite) is the baseline. OFF composited
  (`gamescope_composite_force=1`) matches ON: both sit about 2 ms above OFF direct, all of it after
  the latch (`rec_flip_ms_*`). The difference is the forced composite; framegen's own work adds
  0–0.5 ms.
- **`GAMESCOPE_FRAMEGEN_BATCH_REPAINT` (5), 2026-10-16/17.** GravityMark 1M, 2048x1280@120,
  headless, GPU-bound client: real frames rose from 34 to 44 FPS with steadier flips, while the
  source-to-flip p95 grew from about 11 to about 25 ms. Laggier in a blind A/B, hence off by
  default.
- **GPU cost per generated frame by pipeline (7), 2026-10-16/17.** Real-game setup (hybrid laptop,
  game on the NVIDIA RTX PRO 500, gamescope on the Radeon 890M, 2560x1600@120; three Far Cry games):
  `warp` 1.1 to 1.3 ms, `learned` 1.2 to 3.1 ms, `guided` 1.2 to 3.8 ms depending on the game, against an
  8.3 ms frame at 120 Hz. Weaker GPUs or
  higher resolutions may need `warp` or `checked`.
