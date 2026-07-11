# Proposal 3 — dGPU Optical-Flow Donor via `VK_NV_optical_flow`

Status: exploratory / opt-in quality accelerator. Longer-term item, gated behind the
vendor-neutral on-GPU motion estimator (Tier 2) as its unconditional fallback.

Scope: replace/augment the compositor's on-GPU block-matching motion field for the
motion-compensated generated-frame path with a real optical-flow field produced on the
**render** (dGPU) side by NVIDIA's fixed-function Optical Flow Accelerator (OFA), shipped
across PCIe as a small dmabuf.

**Related work / prior art:** this is essentially the **DLSS 3** route — a
fixed-function optical-flow accelerator feeding frame generation. Note **DLSS 4
replaced the OFA with a learned AI flow network** (the OFA underperformed on
games — UI, particles, specular; research [`../research-framegen.md`](../research-framegen.md)
§4), so a learned flow front-end — **SEA-RAFT** (ECCV 2024, arXiv 2405.14793) or
**NeuFlow v2** (arXiv 2408.10161), research §2 — is the strategic endpoint; the
fixed-function donor is best treated as a measurement baseline.

---

## Motivation / problem it solves

The compositor-side frame generator (`src/rendervulkan.cpp`,
`framegen_record_real_frame` ~4416, dispatch ~4548) today produces the in-between frame by
**forward pixel-space extrapolation** (`cs_framegen_extrapolate.comp`) with adaptive motion
suppression and TAA-style neighborhood rectification. Its Tier 2 evolution adds an on-GPU
**motion-compensated** mode: a luma pyramid + block matching + warp, all running on the
compositing GPU.

Block matching on the compositor has two costs that scale badly exactly when we can least
afford them:

1. **Shader/ALU cost on the GPU that must never stall a composite.** The whole design's
   invariant (see the comment block at ~4512: "All framegen GPU work lives in its own
   command buffer, submitted after the composite … add no latency to the frames the game
   actually rendered") depends on the compositing GPU having headroom. Block matching a
   1080p→4K field is hundreds of GFLOP of correlation work; on an iGPU compositor it can be
   the dominant framegen cost and the first thing to blow the headroom budget that
   `framegen_record_real_frame` guards with `hasCompleted()` (~4494) and the stabilization
   hysteresis (~4504).
2. **Quality ceiling.** Block matching is a local, integer-search heuristic. It smears at
   disocclusions and thin structures, precisely where extrapolation already struggles.

Meanwhile, in the canonical dual-GPU deployment the **render GPU is NVIDIA** and already owns
the game frames. Its OFA is a *fixed-function* block that computes dense optical flow at
**zero SM/shader cost** — it does not compete with the game's graphics/compute work for
shader cores (it has its own contention surface, discussed below). DLSS 3 Frame Generation
uses exactly this unit. The idea: spend the OFA (which is otherwise idle) to produce a
high-quality motion field on the dGPU, and ship only the *tiny* flow field (~0.1–0.5 MB)
across PCIe instead of paying for block matching on the constrained compositor GPU.

This is strictly a **quality/efficiency accelerator**: opt-in, vendor-specific, with the
vendor-neutral on-GPU estimator as the mandatory fallback. It does not become a hard
dependency for the base feature (Constraint 3).

---

## Design overview

```
  dGPU (NVIDIA, render)                         PCIe                compositor GPU (e.g. AMD iGPU)
  ─────────────────────                         ────                ──────────────────────────────
  game frame N  (dmabuf)  ──local import──┐                          ┌── game frame N (dmabuf, 33 MB) ──► composite
  game frame N-1(dmabuf)  ──local import──┤                          │
                                          ▼                          │
                              luma-extract (R8_UNORM)                │
                                          ▼                          │
                        OFA: flow(N-1 → N) @ grid 4×4/8×8            │
                                          ▼                          │
                          flow field (R16G16_SFLOAT, ~0.1–0.5 MB) ───┼──► import flow field ──► warp pass
                          + timeline signal / fence                  │        (motion-comp generated frame)
```

Key architectural facts established by reading the tree:

- gamescope holds **exactly one** Vulkan device: the global `CVulkanDevice g_device`
  (`src/rendervulkan.cpp:2055`), selected by `selectPhysDev` (~392). That device is the
  **compositor** device. In dual-GPU mode gamescope imports the game's dmabuf into *this*
  device (`vulkan_create_texture_from_dmabuf` ~3731 → `CVulkanTexture::BInit` →
  `VkImportMemoryFdInfoKHR` with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT` ~2432/2479).
- gamescope has **no Vulkan device on the render GPU today.** The game (Wine/DXVK) owns its
  own device on the dGPU in its own process. gamescope only ever sees the *result* dmabuf.

Therefore this proposal requires gamescope to **stand up a second, minimal Vulkan device on
the dGPU physical device** — a "helper" context — whose only job is: import the game
dmabufs (local, no PCIe), run OFA, export the flow field. This is the honest architectural
gap and the reason this is a longer-term item; see *Integration points* and *Open questions*.

---

## Vulkan mechanisms & extensions (exact)

### The optical-flow producer (dGPU helper device)

Device-level extension: **`VK_NV_optical_flow`** (`VK_NV_OPTICAL_FLOW_EXTENSION_NAME`).
Its dependencies, all of which the helper device must enable:

- `VK_KHR_synchronization2` (core in 1.3) — for the `VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV`
  / `VK_ACCESS_2_OPTICAL_FLOW_READ_BIT_NV` / `..._WRITE_BIT_NV` barriers around the session.
- `VK_KHR_format_feature_flags2` (core in 1.3) — OFA image format capabilities are reported
  via `VkFormatProperties3`.
- Feature struct: chain `VkPhysicalDeviceOpticalFlowFeaturesNV { .opticalFlow = VK_TRUE }`
  into `VkPhysicalDeviceFeatures2` at device create (mirrors the existing feature chaining at
  ~737–772).

Queue: OFA executes on a queue from a family advertising **`VK_QUEUE_OPTICAL_FLOW_BIT_NV`**.
This is a *distinct* capability bit; the helper device must enumerate queue families
(`GetPhysicalDeviceQueueFamilyProperties`, as done for general/compute at ~418–431) and pick
one exposing that bit. On current NVIDIA drivers the OFA queue is typically a dedicated
family.

Session object and calls:

- `VkOpticalFlowSessionNV` created by **`vkCreateOpticalFlowSessionNV`** with
  `VkOpticalFlowSessionCreateInfoNV`:
  - `width`, `height` = render (game) resolution.
  - `imageFormat` = an OFA-supported **input** format (see below).
  - `flowVectorFormat` = an OFA-supported **output** format (see below).
  - `outputGridSize` = one of `VkOpticalFlowGridSizeFlagBitsNV`:
    `VK_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_NV` / `_2X2_` / `_4X4_` / `_8X8_`. We choose **4×4**
    (one vector per 4×4 input pixels → field of `ceil(W/4)×ceil(H/4)`) as the quality/size
    knob; 8×8 halves size again for the headroom-starved case.
  - `performanceLevel` = `VkOpticalFlowPerformanceLevelNV`
    (`VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_FAST_NV` for lowest latency).
  - `flags` = `VkOpticalFlowSessionCreateFlagBitsNV`; we want `..._ENABLE_TEMPORAL_HINTS_BIT_NV`
    off initially (we feed a fresh pair each frame), and no cost/global-flow outputs.
- Images are attached with **`vkBindOpticalFlowSessionImageNV`** at binding points
  `VkOpticalFlowSessionBindingPointNV`:
  `VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV` (frame N),
  `..._REFERENCE_NV` (frame N-1),
  `..._FLOW_VECTOR_NV` (output). (`_HINT_NV`, `_COST_NV`, `_GLOBAL_FLOW_NV` unused.)
- Execution: **`vkCmdOpticalFlowExecuteNV( cmd, session, &VkOpticalFlowExecuteInfoNV )`** on
  the OFA queue. `VkOpticalFlowExecuteInfoNV.flags` can carry
  `VK_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_NV` and optional `regionCount`/`pRegions`.
- Session images must be **transitioned to the special layout**
  `VK_IMAGE_LAYOUT_OPTICAL_FLOW_...` semantics via the usage query; concretely, the images
  are created with a `VkOpticalFlowImageFormatInfoNV { .usage = <VK_OPTICAL_FLOW_USAGE_*> }`
  chained into the `VkImageCreateInfo.pNext` (and into the format query).

**Format queries (do not hardcode).** Enumerate with
**`vkGetPhysicalDeviceOpticalFlowImageFormatsNV`**, passing
`VkOpticalFlowImageFormatInfoNV.usage`:
- `VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV` — supported *input* formats. On current NVIDIA
  drivers these include `VK_FORMAT_R8_UNORM` (single-channel luma — our target),
  `VK_FORMAT_R8G8B8A8_UNORM` / `VK_FORMAT_A8B8G8R8_UNORM_PACK32`, and NV12-style
  `VK_FORMAT_G8_B8R8_2PLANE_420_UNORM`.
- `VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV` — supported *flow-vector* formats. Query it; on
  current drivers this returns **`VK_FORMAT_R16G16_SFLOAT`** — one half-float per axis,
  the displacement in **pixels** at the grid granularity (the OFA's native S10.5 fixed point
  is exposed to Vulkan as R16G16_SFLOAT). Treat "query and assert" as the contract; do not
  assume S10.5 raw shorts.

Other usage bits (`VK_OPTICAL_FLOW_USAGE_HINT_BIT_NV`, `..._COST_BIT_NV`,
`..._GLOBAL_FLOW_BIT_NV`) are unused here.

### Cross-device sharing (dGPU → compositor)

- **Memory / the flow image**: `VK_KHR_external_memory_fd` +
  `VK_EXT_external_memory_dma_buf` (both already enabled on the compositor device at
  ~673–674). The flow image is allocated **exportable** (`VkExportMemoryAllocateInfo`,
  `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`, mirroring ~2457) and imported on the
  compositor with `VkImportMemoryFdInfoKHR` — the identical path
  `vulkan_create_texture_from_dmabuf`/`BInit` already uses for the 33 MB game frame.
- **Timeline semaphore**: `VK_KHR_external_semaphore_fd` +
  `VK_KHR_timeline_semaphore` (both already present: `timelineSemaphore = VK_TRUE` at ~759,
  extension at ~676). The producer signals a timeline point; the compositor's framegen
  command buffer waits on it via the existing
  `CVulkanCmdBuffer::AddDependency(pTimelineSemaphore, ulPoint)` (~1590) machinery, importing
  the fd through the existing `ImportTimelineSemaphore` shape (~1541, `VkImportSemaphoreFdInfoKHR`,
  `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT`, ~1572).
  **Caveat (critical, see Risks):** `OPAQUE_FD` timeline payloads are **driver-specific and
  not portable NVIDIA↔radv**. This is why the handoff is tiered below.

### Handoff tiers (ship in reverse order)

| Tier | Memory handoff | Sync handoff | Portability | PCIe cost |
|------|----------------|--------------|-------------|-----------|
| C (ship first) | CPU staging: OFA→linear image→`vkCmdCopyImageToBuffer`→mapped readback→`memcpy`→upload to compositor image | `vkWaitSemaphores` on the **dGPU** device (helper thread, CPU-side); no cross-vendor GPU sync at all | Always works | ~0.5 MB readback + ~0.5 MB upload, off critical path |
| B | dmabuf import of the (linear) flow image into the compositor device | still CPU fence **on the dGPU** side; compositor cmd buffer just samples the imported image (no cross-vendor semaphore) | Needs NVIDIA→amdgpu **dmabuf** interop (linear only) | ~0.5 MB DMA, zero-copy-ish |
| A | dmabuf import | cross-vendor **external timeline** wait on the compositor (`AddDependency`) | Needs NVIDIA↔radv `OPAQUE_FD` timeline interop — **not reliable today** | best |

The base feature never depends on Tier A/B working; Tier C is pure dmabuf-free CPU handoff and
degrades to "just use the vendor-neutral estimator" if even the helper device can't be created.

---

## Integration points in gamescope

- **Helper device creation.** New code paralleling `selectPhysDev` (~392) and
  `createDevice` (~600–807). `selectPhysDev` already enumerates *all* physical devices from
  the one `VkInstance` and logs every candidate (~410–505, incl. vendor/device IDs and DRM
  PCI props at ~565). A new `CVulkanOFADevice` (or a slimmed reuse of `CVulkanDevice`) would:
  pick the physical device whose DRM render node matches the **game's** render GPU (the
  non-compositor NVIDIA device the dual-gpu-route logging at ~182–206 already identifies);
  enable `VK_NV_optical_flow` + sync2 + external memory/semaphore fd; grab a
  `VK_QUEUE_OPTICAL_FLOW_BIT_NV` queue. Gate the whole thing on vendor == NVIDIA and the
  extension being present, else never instantiate it.
- **Feeding OFA inputs.** The game dmabuf arriving in dual-GPU mode is imported today only
  into `g_device` (`vulkan_create_texture_from_dmabuf` ~3731). Add a parallel import of the
  **same** `wlr_dmabuf_attributes` into the helper device (local to the dGPU → no PCIe). The
  history ring already keeps `previousReal`/`currentReal` copies on the compositor
  (`FramegenHistory_t`, swap at ~4525/4580); the OFA path needs the two *game-side* dmabufs
  (N-1, N), so the helper keeps its own 2-deep dmabuf ring keyed by the same frame-id counter
  `currentFrameId` (~4460).
- **Producing the flow.** A new helper-side submit (its own command buffer, OFA queue):
  luma-extract compute → barrier → `vkCmdOpticalFlowExecuteNV` → (Tier C) copy-to-buffer.
  Signals `g_framegenHistory.flowSeqNo` analogous to `generatedSeqNo` (~4565).
- **Consuming the flow.** Inside `framegen_record_real_frame`, right before the dispatch at
  ~4548: when `g_eFramegenMode == MotionComp` **and** the OFA flow for `(previousFrameId,
  currentFrameId)` is ready, bind the imported flow field as an extra texture and select the
  motion-comp shader instead of `cs_framegen_extrapolate`. If the flow is *not* ready (helper
  oversubscribed, import failed), fall through to the on-GPU estimator or plain extrapolation
  — never block.
- **Vector scaling.** The flow is at *render* resolution; the warp runs at *output*
  resolution after upscale (FSR/NIS path selection is visible in `vulkan_composite` at
  ~4593–4604). Push `flScaleX/Y = outputExtent / renderExtent` and the per-slot temporal
  fraction (the same "dynamic per-slot temporal fraction" the Tier 1 scheduler feeds
  `FramegenPushData_t` ~4185/4546) into the warp shader; sample the flow bilinearly (it is a
  coarse `W/4 × H/4` field).
- **Scheduler / present.** Unchanged. Generated frames are still consumed by
  `vulkan_framegen_consume_generated_frame` (~4299) and presented with **no Wayland commit**
  and no client `frame_done` — OFA changes only *how* the pixels are produced, not *when* or
  *whether* a generated frame exists. The `hasCompleted` headroom gate (~4494) is extended to
  also require the flow submit to have retired, so a slow OFA can never wedge the next real
  composite.

---

## Data & control flow (step by step)

1. Game presents frame **N** on the dGPU → gamescope receives the dmabuf (dual-GPU route).
2. gamescope imports the dmabuf into **`g_device`** (compositor; the 33 MB PCIe crossing that
   already happens) **and**, in parallel, into the **helper device** (local to dGPU, ~0 PCIe).
3. Compositor composites the real frame N and presents it. **This path waits on nothing from
   OFA.** (Real-frame critical path is identical to today.)
4. *After* the composite submit, `framegen_record_real_frame` runs (its own cmd buffer,
   ~4516). It now also — if OFA enabled and helper has both N-1 and N — kicks the **helper**
   submit: luma-extract(N-1), luma-extract(N), `vkCmdOpticalFlowExecuteNV(N-1→N)`, export.
5. The helper signals `flowSeqNo`. A helper thread does the Tier-C/B fence wait on the **dGPU**
   device and stages/imports the ~0.1–0.5 MB flow field to the compositor.
6. When the next slot's generated frame is built (still after a real composite, off critical
   path), the compositor warp samples the imported flow (scaled by upscale ratio × temporal
   fraction), producing the motion-compensated in-between frame into
   `g_output.framegenOutputImages[...]` (~4533).
7. The scheduler presents that generated frame in the empty vblank (~9130 in
   `steamcompmgr.cpp`) with **no client commit / no presentation feedback** (Constraint 2).
8. If at step 4/5 the flow for the current pair isn't ready in time, the warp uses the
   previous flow, the on-GPU estimator, or degrades to extrapolation — chosen per-frame,
   never blocking (Constraint 4).

Why it stays off the real-frame critical path: the OFA producer lives on a **different
device** entirely and is *pulled* by the already-decoupled framegen command buffer; the real
frame's present (step 3) has literally no edge to any OFA object. The single place a real
frame could ever be delayed — the *next* composite queuing behind framegen on the *same*
queue — is the exact hazard the existing `hasCompleted` gate (~4494) guards, and we extend
that gate to include `flowSeqNo`.

---

## Latency & throughput analysis

Concrete numbers, 1920×1080 render → 3840×2160 output, 8-bit:

- **Flow field size.** Grid 4×4 → `480×270` vectors × 4 B (R16G16) = **~518 KB**. Grid 8×8 →
  `240×135` × 4 B = **~130 KB**. Versus the **~33 MB** (3840×2160×4 B) output frame already
  crossing PCIe every real frame: the flow adds **0.4–1.6 %** to PCIe traffic in the
  dGPU→compositor direction. Negligible.
- **OFA execution time.** NVIDIA's OFA at `FAST` performance level, 1080p, quarter-res grid,
  is well under **1 ms** on Ampere/Ada (the unit is spec'd for real-time 4K flow; DLSS3 FG
  budgets it inside a frame). It is *fixed-function* → **0 SM cost**, so it does not steal
  shader cycles from the game.
- **Luma-extract pre-pass.** Two `R8_UNORM` downconverts of 1080p ≈ 2 M invocations each,
  sub-0.1 ms on the dGPU; can be skipped if the OFA accepts the game's RGBA format directly
  (query decides).
- **Added latency to the generated frame.** The dependency chain
  `frame N ready → luma → OFA → export → import → warp` is **entirely inside the inter-frame
  gap** the generated frame is meant to fill. The generated frame is displayed in the vblank
  *after* N; there is a full refresh interval (e.g. 16.6 ms @ 60 Hz, 6.9 ms @ 144 Hz) of
  slack. OFA (<1 ms) + PCIe DMA of ~0.5 MB (a few µs at ~16 GB/s) + import fit trivially.
- **Compositor savings.** Replacing on-GPU block matching removes the single largest framegen
  ALU cost on the constrained compositor GPU. On an iGPU compositor this is the difference
  between the framegen path fitting in headroom (staying past the `hasCompleted` gate) and
  being throttled out by the stabilization hysteresis (~4504). The warp pass that *consumes*
  the flow is cheap (one gather + bilinear-flow-sample per output pixel).
- **New cost: a second device.** The helper device holds its own allocator, queues, and a
  2-deep dmabuf ring on the dGPU (~2 × render-res images, on the GPU that has VRAM to spare).
  This is real memory/complexity but not on any latency-sensitive path.

---

## Interaction with VRR / HDR / anti-lag / tearing

- **VRR.** Framegen already suppresses adaptive sync while active (logged at ~4425; VRR gating
  lives at `allowVRR` ~2566 in `steamcompmgr.cpp` and `drm_prepare` ~2930 in `DRMBackend.cpp`).
  OFA does not change this; it changes only generated-frame content.
- **HDR.** OFA input must be a supported format; HDR game frames are 10-bit
  (`VK_FORMAT_A2B10G10R10` / fp16). Luma extraction to `R8_UNORM` for OFA is fine (flow is
  computed on luma, tone-mapping-invariant enough for motion), but the **warp** must run in
  the frame's real encoding — the history-EOTF guard already invalidates on EOTF change
  (~4450–4455), so a flow computed across an SDR↔HDR transition is simply dropped, same as the
  vendor-neutral path.
- **Anti-lag / Reflex.** **Untouched by construction.** OFA runs on gamescope's helper device
  in gamescope's process; it issues **no** work into the game's device/queue and produces
  **no** Wayland commit or presentation feedback (Constraint 2). The game/Wine/DXVK/Reflex
  pipeline cannot observe it. This is the same "critical rule" invariant as the base feature.
- **Tearing.** Immediate-flip / tearing (`steamcompmgr.cpp` ~9037) is orthogonal; generated
  frames are presented on the same non-tearing path the base framegen uses.
- **Overlay layers.** OFA flow is computed on the **base game layer only**. HUD/cursor/overlay
  layers (`FrameInfo_t::layers`, base-texture identity tracked at ~4434) must be excluded from
  the warp and composited fresh or held — identical requirement to the vendor-neutral estimator.

---

## Risks & mitigations (adversarial)

| Risk | Severity | Mitigation |
|------|----------|------------|
| gamescope has **no device on the render GPU**; standing one up is a large new subsystem | High (scope) | Ship the vendor-neutral on-GPU estimator (Tier 2) first as the real feature; OFA donor is an *optional* accelerator layered on later. Treat helper-device creation as best-effort: if it fails, the feature simply never turns on. |
| NVIDIA↔radv **timeline `OPAQUE_FD` interop is unreliable** | High | Tier A avoided by default. Ship Tier C (CPU fence on the *dGPU* device + staging copy) then Tier B (dmabuf memory only, still dGPU-side fence). No cross-vendor GPU semaphore on the default path. |
| NVIDIA→amdgpu **dmabuf import** of NV-allocated memory may fail / require linear | Med | Tier C needs no dmabuf at all (CPU staging). Tier B forces a **linear** flow image and validates import at init with a canary; falls back to C on failure. |
| OFA only on **NVIDIA Turing+ proprietary driver**; **NVK/open driver has no OFA**; Turing OFA quality is weaker than Ampere/Ada | Med | Strictly opt-in and vendor-gated. Detect at init (`VK_NV_optical_flow` present + `opticalFlow` feature). NVK / AMD / Intel render GPUs → never enabled, vendor-neutral path used. |
| Game frame is **tiled/modifier dmabuf** OFA can't consume | Med | Luma-extract pre-pass reads via a sampled import (modifier-aware, same path as `BInit`) and writes a plain `R8_UNORM`, decoupling OFA from the game's layout. |
| OFA contends for dGPU **copy/memory engines** and could perturb the game | Low–Med | OFA on `FAST` at quarter-res is <1 ms and fixed-function; schedule after frame N is already presented so it never sits in front of the game's own present. Measure frametime variance (below) and auto-disable on regression. |
| Flow is **N-1→N** (backward-in-time relative to the frame we extrapolate *forward*) | Med | Warp advects frame N by `+fraction × flow` for forward extrapolation; validate sign/scale against the estimator on known pans (see testing). Disocclusion holes filled by the existing neighborhood rectification. |
| Helper device **oversubscribed** (flow late) | Low | Per-frame readiness check; use previous flow / estimator / extrapolation. Extend the `hasCompleted` gate (~4494) to include `flowSeqNo` so a slow OFA can't wedge the next composite. |
| Extra **process-lifetime dmabuf fds** on the dGPU leak | Low | 2-deep ring with explicit close on eviction, keyed to `currentFrameId`; mirror the existing history swap discipline (~4525/4580). |

---

## Testing & validation plan

Counters/logs to add (behind `--framegen-debug`, extending the existing `framegen:` log
namespace at ~4275/4463/4571):

- `ofa: session created grid=%s in=%s out=%s queueFamily=%u` at helper init.
- Per pair: `ofa: flow id=%llu->%llu exec=%.2fms export=%.2fms pcie=%.1fKB` (timestamp
  queries around `vkCmdOpticalFlowExecuteNV` and the copy/import).
- `ofa: flow late for id=%llu (fell back to %s)` when the consumer misses the flow.
- A rolling counter of `flow_used / estimator_used / extrapolate_used` per 100 generated
  frames to see how often the accelerator actually engages.

Correctness / quality measurements:

1. **Flow sanity harness.** Feed synthetic pairs with a known rigid pan/zoom; assert the OFA
   field (after scale) matches the analytic displacement within tolerance, and that its sign
   convention matches the on-GPU estimator on the same pair.
2. **A/B image quality.** Same recorded gameplay, three modes (extrapolate / estimator /
   OFA-donor); compare generated frames against the *real* frame that a non-framegen run
   would have shown at that instant (offline, since generated frames have no ground truth
   live) using PSNR/SSIM/FLIP, focusing on disocclusion regions.
3. **Critical-path proof.** GPU timestamp on the *real* frame's composite→present must be
   statistically identical with OFA on vs off (Constraint 1). Assert no OFA object appears in
   the real frame's submit dependency set (static check of the composite cmd buffer's
   `m_ExternalDependencies`).
4. **No client-visible leakage.** Instrument the Wayland side: assert zero extra
   `frame_done`/presentation-feedback events per generated frame with OFA on (Constraint 2),
   and that Reflex/anti-lag markers (from DXVK) are unchanged vs base framegen.
5. **Headroom degradation.** Artificially stall the helper OFA queue; verify the consumer
   falls back and the extended `hasCompleted` gate prevents any real-composite delay.
6. **Interop matrix.** Validate Tier C on NVIDIA-render + {AMD, Intel, NVIDIA} compositor;
   attempt Tier B/A and record which driver pairs actually import the dmabuf / timeline.

---

## Incremental rollout / flags

Existing flags (`src/main.cpp:139–143`, globals `346–350`):
`--experimental-framegen`, `--framegen-multiplier`, `--framegen-mode {extrapolate,blend}`,
`--framegen-strength`, `--framegen-debug`.

Additions:

- `--framegen-mode motioncomp` — the Tier 2 vendor-neutral motion-compensated mode (this is
  the baseline that must exist first).
- `--framegen-ofa {auto,off}` (default `off` while experimental; later `auto`) — opt-in the
  OFA donor. `auto` = enable **iff** render GPU is NVIDIA **and** `VK_NV_optical_flow` +
  `opticalFlow` feature present **and** a `VK_QUEUE_OPTICAL_FLOW_BIT_NV` queue exists **and**
  the helper device + flow round-trip pass a one-time init canary; otherwise silently use
  `motioncomp`.
- `--framegen-ofa-grid {4,8}` (default 4) — quality/bandwidth knob (`4×4` vs `8×8`).
- `--framegen-ofa-handoff {cpu,dmabuf}` (default `cpu` = Tier C) — escape hatch to force the
  robust path or opt into zero-copy where interop is known good.

Gating chain (fail-safe at every step, never a hard dependency — Constraint 3):
`framegen enabled? → mode==motioncomp/ofa? → ofa requested? → render GPU NVIDIA? →
extension+feature+queue? → helper device created? → init canary flow round-trips? → engage.`
Any `no` falls back to the vendor-neutral estimator, which itself falls back to extrapolation.

---

## Open questions

1. **Do we ever get the game dmabuf into a gamescope-owned dGPU context cleanly?** In
   dual-GPU mode the dmabuf originates on the dGPU, so a helper-device import should be local
   — but is the fd/modifier the game exports always importable by a *second* NVIDIA device in
   gamescope's process, or only by the compositor device's import path? Needs a spike.
2. **Single-GPU deployments.** If render and compositor are the *same* NVIDIA GPU, there is no
   PCIe crossing and no second device — OFA could run on the one device directly. Is that a
   simpler first target than the dual-GPU case, even though the headroom argument is weaker?
3. **Temporal hints / multi-multiplier.** For x3/x4 (Tier 1's per-slot multiplier), do we run
   OFA once per real pair and reuse the field for all in-between slots (scaling fraction), or
   does enabling `..._ENABLE_TEMPORAL_HINTS_BIT_NV` and re-running per slot pay for itself?
4. **HDR luma choice.** Which luma transfer (PQ-domain vs linear vs simple max) gives OFA the
   most stable motion on HDR content without tone-mapping artifacts corrupting the field?
5. **Turing viability.** Is gen-1 OFA (Turing) good enough to be worth enabling, or should
   `auto` require Ampere+ (`deviceID`/arch gating)?
6. **Cross-vendor timeline, long term.** Track whether NVIDIA + radv converge on a portable
   external timeline handle type (syncobj) so Tier A becomes real and the CPU fence in Tier C
   can be dropped for the zero-copy path.
