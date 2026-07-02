# Proposal 5 — Tile classification + indirect dispatch + SDMA static-tile fill

Status: design draft (deferred Tier-2 item **T2.5b**)
Scope: an optimization *inside* the existing compositor-side x2 frame-generation
path. It changes **how** a generated frame is produced, not **when** it is
produced or **whether** it is presented. All Tier-0 invariants continue to hold:
real frames are never delayed, generated frames never emit client
frame_done/presentation feedback, no vendor-locked hard dependency, and the
whole path stays dormant when the compositing GPU has no headroom.

---

## Motivation / problem it solves

The current generator runs one full-resolution compute pass over the entire
output every time it fires. `framegen_record_real_frame()`
(`src/rendervulkan.cpp` ~4559) dispatches `cs_framegen_extrapolate.comp` at
`div_roundup(width, 8) x div_roundup(height, 8)` groups — i.e. **every pixel of
every generated frame** pays for a texture fetch of `previousReal` and
`currentReal`, the extrapolation math, and the TAA-style neighborhood
rectification, regardless of whether that pixel moved at all.

For a large class of real workloads this is almost pure waste:

- **Menus, pause screens, inventory/skill trees** — the entire frame is static
  except a cursor and a couple of animated glints.
- **Strategy / management / turn-based games** — a mostly-static map with a HUD;
  camera is stationary for seconds at a time.
- **HUD-heavy views generally** — large opaque overlays that never change
  between two real frames.

On a static region, forward extrapolation is a no-op by construction: with
`previousReal == currentReal` for that tile, the predicted step is zero and the
generated pixel equals the current pixel. We are spending the iGPU's full
compute throughput (and package power, which on a handheld directly costs the
dGPU its power budget via a shared TDP) to compute `out = in`.

The idea: **detect static regions cheaply, generate only the moving ones, and
fill the static ones with a DMA copy on an engine that costs zero shader-core
time.** On mostly-static content the generation cost collapses toward the cost
of a full-frame DMA copy plus a tiny classification pass. As a bonus, the
tile-motion map produced here is exactly the coarse input the Tier-2
motion-estimation mode (T2.6) needs, so it is computed once and reused.

---

## Design overview

Three cooperating passes replace the single monolithic dispatch, all still
inside the framegen command buffer submitted *after* the composite:

1. **Classify (1/8-res change detection).** A compute pass reads `previousReal`
   and `currentReal` downsampled to tile granularity and, per tile, computes a
   motion metric (SAD or max per-channel delta of the tile between the two real
   frames). It writes a per-tile `moving` flag into a **tile-flag buffer** and,
   for tiles marked moving, appends the tile's coordinate into a **compaction
   buffer** via an atomic counter. The final atomic count is written into the
   `groupCountX` field of a `VkDispatchIndirectCommand`.

2. **Generate (indirect, moving tiles only).** `vkCmdDispatchIndirect` launches
   the expensive extrapolation/rectification shader with exactly one workgroup
   per moving tile. Each workgroup reads its tile coordinate from the compaction
   buffer and writes only that tile of the generated image.

3. **Static fill (SDMA copy).** In parallel, `currentReal` is copied into the
   static regions of the generated image on the **dedicated transfer (SDMA)
   queue**, which is a DMA engine that consumes zero CU/EU time and otherwise
   sits idle during compositing.

The generated image is coherent only after both (2) and (3) complete and their
writes are made visible to the presenting queue. Synchronization is via a
timeline semaphore (already the device's primitive; see `submitInternal`
~1400) plus queue-family ownership transfers for the images the transfer queue
touches.

When no dedicated transfer family exists (see the RADV discussion below), the
static fill degrades to a compute copy on the same queue — still a win, because
the *expensive* generation shader is skipped on static tiles even though the
DMA offload is lost.

---

## Vulkan mechanisms & extensions

Everything here is **core Vulkan 1.2/1.3** — no vendor extension is required for
the base feature, satisfying constraint (3). The device is already created at
`VK_API_VERSION_1_2` minimum (`selectPhysDev` ~415) with
`VkPhysicalDeviceVulkan13Features` (~717).

### Indirect dispatch
- `vkCmdDispatchIndirect(cmdBuf, buffer, offset)` — reads a single
  `VkDispatchIndirectCommand { uint32_t x, y, z; }` from a device-local buffer.
  The generate pass sets `y = z = 1` and `x = movingTileCount`.
- The args buffer needs `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT` plus
  `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` (the classify pass writes `x` as an SSBO).

### Append/compaction buffer with atomic counters
- A device-local `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` buffer laid out as
  `{ uint counter; uvec2 tileCoords[maxTiles]; }`, or the counter kept in the
  indirect-args buffer's `x` field directly and the coords in a second buffer.
- The classify shader uses `atomicAdd(counter, 1)` (GLSL `atomicAdd` on an SSBO
  member, i.e. `VK_ACCESS_SHADER_WRITE_BIT` semantics) to reserve a slot, then
  writes `tileCoords[slot] = thisTile`. The counter must be zeroed each frame
  with `vkCmdFillBuffer` (a transfer op) before classify runs.
- Alternative to per-tile global atomics: a **subgroup-local reduction then one
  atomic per subgroup**. Each subgroup counts its moving tiles with
  `subgroupBallotBitCount(subgroupBallot(isMoving))`, does one `atomicAdd` for
  the whole subgroup, and each lane computes its own slot with
  `subgroupExclusiveAdd(uint(isMoving))`. This cuts atomic traffic by up to the
  subgroup width (32 on RDNA wave32, 64 on GCN wave64).

### Subgroup ops for the tile reduction
- `GL_KHR_shader_subgroup` (maps to `VK_KHR_shader_subgroup` / core 1.1
  `VkPhysicalDeviceSubgroupProperties`). Needed operations:
  `subgroupAdd`/`subgroupMax` (arithmetic + ballot categories) for reducing the
  per-pixel deltas within a tile down to one metric, and
  `subgroupBallot`/`subgroupExclusiveAdd` for the compaction. Query
  `supportedOperations` for `BALLOT | ARITHMETIC` and `supportedStages` for
  `COMPUTE` at init; fall back to LDS reduction + a single global atomic if a
  driver reports them absent (none of our targets do, but keep the guard).
- fp16 tile reduction can use `VK_KHR_shader_subgroup_extended_types` when
  `m_bSupportsFp16` (already probed at ~627) is set; otherwise reduce in fp32.

### The transfer (SDMA) queue — does RADV expose an SDMA-only family?
- Yes. On AMD, RADV exposes a queue family with
  `VK_QUEUE_TRANSFER_BIT` set and **neither** `VK_QUEUE_GRAPHICS_BIT` nor
  `VK_QUEUE_COMPUTE_BIT` — that is the SDMA/PCIe DMA engine. This is exactly the
  "dedicated transfer family" Vulkan recommends copies be routed to. The AMD
  Windows driver and RADV both surface it; the RADV kernel path drives the
  `sdma` ring.
- **Intel (ANV / Xe):** historically exposes only graphics+compute families;
  newer Xe hardware has copy engines (BLT), but exposure is not guaranteed.
  Treat SDMA as opt-in-if-present.
- **NVIDIA (nvidia / NVK):** exposes a transfer-only family (the copy engines).
- The classification and indirect-dispatch parts are vendor-neutral and always
  run; **only the DMA offload is gated on a dedicated transfer family existing.**

### Queue-family ownership transfer (SDMA copy ↔ compute generation)
The generated image is written by two different queue families in the same
frame (compute writes moving tiles, transfer writes static tiles), then read by
the presenting queue. Because the resource is created with
`VK_SHARING_MODE_EXCLUSIVE` (all gamescope images are), cross-family visibility
requires a **queue-family ownership transfer**: a matched
release barrier on the source family and acquire barrier on the destination
family, each a `VkImageMemoryBarrier` with distinct `srcQueueFamilyIndex` /
`dstQueueFamilyIndex`. gamescope already performs exactly this pattern for
foreign-queue DMA-BUF import/export in `insertBarrier()` (~2027-2041), so the
mechanism is in place; this proposal adds an *internal* (non-foreign) variant
between `m_transferQueueFamily` and `m_queueFamily`.

Because the two families write **disjoint regions** of the same image, we can
avoid a full-image ownership ping-pong by giving each family ownership of the
whole image for the duration and ordering their completion with the timeline
semaphore. The concrete recipe is in Data & control flow below.

### Sync primitives
- **Timeline semaphores** (`VK_KHR_timeline_semaphore`, already enabled —
  `vulkan12Features.timelineSemaphore = VK_TRUE` at ~759, and
  `m_scratchTimelineSemaphore` is the device's ordering primitive). The
  transfer submit signals a timeline value; the compute generate submit and/or
  the present consume wait on it.
- **`VkBufferMemoryBarrier`** between classify (writes counter + coords +
  indirect args) and generate (reads them as indirect + storage). Source stage
  `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`, dest stages
  `VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT` (for the `VkDispatchIndirectCommand`
  fetch — note indirect dispatch reads args at the *draw-indirect* stage, a
  common footgun) **and** `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` (for the coords
  SSBO). Access `VK_ACCESS_SHADER_WRITE_BIT` → `VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT`.
- **`VkImageMemoryBarrier`** between the two writers and the presenting reader,
  plus the ownership transfers above.

---

## Integration points in gamescope

All line regions are in `src/rendervulkan.cpp` unless noted.

1. **Queue discovery — the gap (`selectPhysDev` ~423-431).** Today the loop
   discovers only `generalIndex` (`GRAPHICS|COMPUTE`) and `computeOnlyIndex`
   (`COMPUTE` without `GRAPHICS`). **There is no search for a transfer-only
   family.** Add a third scan:
   ```
   else if ((flags & VK_QUEUE_TRANSFER_BIT) &&
            !(flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)))
       transferOnlyIndex = std::min(transferOnlyIndex, i);
   ```
   Store `m_transferQueueFamily` alongside `m_queueFamily` /
   `m_generalQueueFamily` (~474-475) and log it under `g_bDebugDualGpuRoute`
   (~505-514) so the route diagnostics show whether SDMA fill is available.

2. **Device + queue + pool creation (`createDevice` ~638-654, ~800-806, and the
   pool creation at ~969-988).** Add a third `VkDeviceQueueCreateInfo` for
   `m_transferQueueFamily` when it differs from the others, bump
   `queueCreateInfoCount` accordingly (currently `m_queueFamily == m_generalQueueFamily ? 1u : 2u`,
   ~748), `vkGetDeviceQueue` into `m_transferQueue`, and create a third
   `VkCommandPool m_transferCommandPool` mirroring the general pool at ~982-988.
   The transfer queue does **not** need global-priority realtime; leave its
   `pNext` null (it must never front-run a composite anyway, and it is a
   different engine).

3. **Framegen resource allocation (`framegen_ensure_resources` ~4373 and the two
   `framegen_create_*` helpers ~4347-4371).** Add allocation of:
   - the tile-flag buffer, coords/compaction buffer, and indirect-args buffer
     (device-local SSBO + indirect usage), sized from the classified tile grid
     `ceil(width/T) x ceil(height/T)`;
   - the output image createFlags (`framegen_create_output_texture` ~4362)
     already set `bStorage` (compute writes) — they must **also** guarantee
     `VK_IMAGE_USAGE_TRANSFER_DST_BIT` so the SDMA copy is legal. `bOutputImage`
     images are flippable; confirm the modifier/tiling chosen still permits an
     SDMA `vkCmdCopyImage`/`vkCmdCopyImageToImage`. If a linear/DMA-incompatible
     modifier is forced, fall back to compute fill for that image.

4. **The generation hook (`framegen_record_real_frame` ~4516-4567).** This is
   where the single `pCmdBuffer->dispatch(...)` (~4560) is replaced by the
   classify → compact → indirect-generate sequence, and where the transfer
   submit for the static fill is issued. The existing structure already:
   - allocates a dedicated framegen command buffer (~4516),
   - copies the real frame into `currentReal` (~4518),
   - submits framegen work in its own command buffer *after* the composite so
     the real present never waits on it (comment ~4512-4515),
   - records `generatedSeqNo` for the completion gate consumed by
     `vulkan_framegen_consume_generated_frame()` (~4310, ~4494).
   The multi-pass version keeps all of that; it just fans the GPU work across
   two queues and records the transfer timeline value into the history so the
   present-side completion check (`hasCompleted(generatedSeqNo)`, ~4310) waits
   on **both** engines (use the max of the two seq/timeline points, or make the
   compute generate wait on the transfer semaphore so a single seq no suffices).

5. **Dispatch/barrier plumbing (`CVulkanCmdBuffer::dispatch` ~1752-1904,
   `copyImage` ~1906-1935, `insertBarrier` ~1999-2053).** New command-buffer
   methods are needed: `dispatchIndirect(buffer, offset)` (mirrors `dispatch`
   but calls `vkCmdDispatchIndirect`), `fillBuffer` (zero the counter), a buffer
   barrier helper, and a transfer-queue copy that records the release/acquire
   ownership barriers. `insertBarrier` already models queue-family ownership for
   the foreign case (~2037-2038) and can be generalized.

6. **Submit path (`submitInternal` ~1400-1458).** Note the comment at ~1435
   "*no need to ensure order of cmd buffer submission, we only have one queue*".
   That assumption breaks once a transfer submit is added, so the transfer
   submit must go through a path that (a) uses `m_transferQueue` and
   `m_transferCommandPool`, and (b) participates in the timeline ordering
   explicitly. The compute framegen submit then adds the transfer's timeline
   point to its wait list (the same mechanism as `GetExternalDependencies()`
   ~1426-1431).

No changes are required in `steamcompmgr.cpp` — the slot scheduler and present
path (~9130) consume the generated frame through the unchanged
`vulkan_framegen_has_pending_generated_frame()` / `..._consume_generated_frame()`
interface. DRM/Wayland backends are likewise untouched: the generated image is
the same `bFlippable` output texture they already flip.

---

## Data & control flow

Per real frame, after the composite command buffer has been submitted (so the
real frame's present is already unblocked):

```
[compositor queue, framegen cmd buffer #A]                [transfer queue, cmd buffer #B]
 0. copyImage(realFrame -> currentReal)   (existing ~4518)
 1. fillBuffer(counter = 0)                                (waits: timeline >= composite done for currentReal)
 2. barrier: currentReal writable -> read                  1'. release currentReal ownership: compute -> transfer
 3. CLASSIFY dispatch (1/8-res):                           2'. acquire generated image ownership on transfer
      per tile: metric(prev,curr); if moving ->            3'. copy currentReal -> generated (STATIC tiles only,
        atomicAdd -> coords[]; write flag                        via per-tile region list OR full-image copy then
      write indirect.x = counter                                 compute overwrites moving tiles)
 4. barrier (buffer): SHADER_WRITE ->                      4'. release generated ownership: transfer -> compute
      INDIRECT_COMMAND_READ | SHADER_READ                       signal timeline T_xfer
 5. acquire generated image ownership: transfer -> compute (wait timeline T_xfer)
 6. GENERATE dispatchIndirect(indirectArgs):
      workgroup g reads coords[g]; extrapolate+rectify
      that tile; write generated[tile]
 7. barrier: generated GENERAL -> present layout,
      make writes visible; signal generatedSeqNo
```

Two viable static-fill strategies, trading DMA precision for simplicity:

- **(A) Full-image DMA + compute overwrite.** Transfer does one full
  `currentReal -> generated` copy; compute then overwrites only moving tiles.
  Simplest (one big linear DMA, no per-tile region list, coalesces perfectly),
  but moving tiles are written twice. Net cost ≈ one full-frame DMA + moving-tile
  compute. Best when the moving fraction is small (the common case) and DMA
  bandwidth is cheap relative to compute.
- **(B) Region-list DMA of static tiles only.** Classify also compacts the
  *static* tile list; transfer copies only those regions
  (`vkCmdCopyImage` with N `VkImageCopy` regions, or several batched copies).
  No double-write, but many small copies hurt SDMA efficiency and add
  descriptor overhead. Prefer (A) unless profiling shows the double-write
  dominates.

**Ordering choice that keeps it off the real-frame critical path.** Steps 1-7
all execute *after* the composite submit for the current real frame. The real
frame's present (`steamcompmgr` present path) waits only on the composite's
completion, never on `generatedSeqNo`. If the transfer or generate work is late,
`vulkan_framegen_consume_generated_frame()` already checks
`hasCompleted(generatedSeqNo)` (~4310) and **discards** the generated frame
rather than stalling scanout — so adding a second engine cannot delay a real
frame; the worst case is a dropped generated frame (a repeated real frame),
which the existing hysteresis/stabilization logic (~4494-4510) already handles.

---

## Latency & throughput analysis

Let the output be `W x H` at `Bpp` bytes/pixel (8 for fp16 RGBA, 4 for
`ARGB8888`), tile size `T` (say 32). Grid = `ceil(W/T) x ceil(H/T)` tiles.

- **Classify pass bandwidth.** Reads both `previousReal` and `currentReal`. If
  it samples every pixel, that is `2 * W * H * Bpp` of read traffic — the same
  read volume the old monolithic shader already paid, but with **no** rectify
  math and **no** output write except one tile flag + occasional atomic. To make
  classify genuinely cheap, sample at reduced density: read one representative
  texel (or a bilinear-downsampled 1/8 tap) per, say, 4x4 block, cutting classify
  read traffic ~16x to `~2 * (W*H/16) * Bpp`. Change detection is robust to this
  because a moving object touching a tile flips it regardless of sub-tile
  sampling. Output: one `uint` flag per tile (`grid` bytes) + up to `grid`
  atomics.
- **Generate pass.** Cost scales with the **moving fraction** `f`:
  `~f * W * H` extrapolation invocations instead of `W * H`. On a menu with
  `f ≈ 0.02`, that is a ~50x reduction in the expensive shader's work. On a
  full-screen pan (`f ≈ 1.0`), it degrades to the current cost plus the small
  classify overhead — i.e. bounded worst case.
- **Static fill (DMA).** Strategy (A) moves `W * H * Bpp` once over the SDMA
  engine. For 1920x1080 fp16 that is ~16.6 MB; at ~16 GB/s effective SDMA
  bandwidth on a typical iGPU that is ~1 ms of DMA — but it runs on a **separate
  engine concurrently with compute**, so it does not add to CU-bound generation
  time. On dual-GPU routing the copy is intra-GPU (compositing GPU's local
  VRAM/UMA), **not** across PCIe: `currentReal` and `generated` both live on the
  compositing device, so this proposal adds **zero cross-PCIe traffic**. (The
  only PCIe traffic in the whole framegen design is the pre-existing dGPU→iGPU
  transfer of the game's rendered frame, which is unchanged.)
- **Package power.** The dominant win on handhelds: skipping the expensive
  shader on static tiles idles the CUs, and the DMA engine draws far less than
  the shader array. On a shared-TDP APU this returns power budget to the dGPU,
  indirectly *raising* real-frame rate — the opposite of the usual framegen
  tax.
- **Added latency to generated frames.** The extra classify pass plus the
  buffer barrier plus the ownership transfers add a few tens of microseconds and
  one extra queue submission of scheduling latency. Because generation already
  targets an *empty vblank* and is gated by `hasCompleted()` on the present
  side, this latency is absorbed entirely within the gap it fills; it never
  touches a real frame. If it ever exceeds the gap, the frame is discarded, not
  delayed.

---

## Interaction with VRR / HDR / anti-lag / tearing

- **VRR / tearing.** Unchanged from the base feature: framegen already forces
  `allowVRR` off and suppresses tearing flips while active (logged at ~4425;
  VRR gate in `steamcompmgr.cpp` ~2566, tearing ~9037). This proposal does not
  alter present cadence or introduce new commits — the generated image is still
  a normal flip of a `bFlippable` output texture, so no new VRR/tearing
  interaction arises.
- **HDR.** Classification runs on the same encoded values the generator already
  consumes. The SAD/max-delta metric is computed in the stored encoding
  (typically the fp16 output). One caveat: in HDR the perceptual significance of
  a raw-value delta is non-linear, so a fixed threshold may over- or
  under-trigger in highlights. Mitigation: scale the threshold by a rough
  luma/OETF factor, or compute the metric on a log/PQ-domain proxy. The existing
  EOTF-change history invalidation (~4450-4455) still applies and forces a clean
  reset across SDR↔HDR transitions.
- **Anti-lag / Reflex.** No new client-visible work. Generated frames still emit
  no Wayland commit and no presentation feedback (the critical rule), so the
  game/DXVK/Reflex pipeline is unaffected. Moving work to the SDMA engine does
  not create any client-visible latency or new frame_done signals.

---

## Risks & mitigations

| Risk | Severity | Mitigation |
| --- | --- | --- |
| No dedicated transfer family (Intel/older HW) | Med | Static fill degrades to a compute copy on `m_queueFamily`; expensive-shader skip is preserved, only DMA offload lost. Gate SDMA path on `m_transferQueueFamily != ~0u`. |
| Ownership-transfer bugs → corruption/validation errors | High | Reuse the audited `insertBarrier` foreign-queue pattern (~2037-2038); prefer strategy (A) (single full-image copy, one ownership hand-off) over per-region ping-pong; validate under `VK_LAYER_KHRONOS_validation` with sync validation on. |
| Tile-boundary seams between DMA-filled static and compute-generated moving tiles | Med | Rectification already blends toward the current frame at high delta; keep a 1-tile "halo" so moving-tile shaders read neighbor context; ensure both writers agree on identical values at shared static pixels (strategy A guarantees this since compute overwrites full tiles). |
| Classify under-detects small/fast motion (aliasing at 1/8 res) | Med | Use max-delta (not mean SAD) so a single changed texel trips the tile; sample density tunable via flag; err toward marking moving (false positives only cost compute, never correctness). |
| Classify over-detects (film grain, dithering, noisy shaders) → f→1 | Low | Threshold with a small dead-band; optionally temporal EMA of per-tile metric. Worst case is the current full-frame cost, so no regression floor is crossed. |
| Indirect-args race (generate reads stale/uninitialized count) | High | Mandatory `vkCmdFillBuffer(counter,0)` before classify + buffer barrier with `DRAW_INDIRECT_BIT` dest stage before `dispatchIndirect`. Missing the draw-indirect stage is the classic bug — assert it in review. |
| Double submission serializes in front of next composite | High (violates Tier-0) | Framegen work (both queues) stays on `m_queueFamily`/`m_transferQueueFamily`, never the composite's critical submit; existing `gpu_oversubscribed` gate (~4494) already backs off if generation hasn't drained by the next real frame. |
| SDMA can't copy the chosen output modifier/tiling | Med | Probe copy support at resource creation; fall back to compute fill per-image; log under dual-gpu-route. |
| Extra buffers/images inflate VRAM on the compositing GPU | Low | Tile buffers are tiny (`grid` * a few bytes); output ring already exists (`framegenOutputImages`). |

---

## Testing & validation plan

**Correctness**
- Golden-image test: for a synthetic pair (`previousReal`, `currentReal`) with a
  known moving rectangle, assert the classify pass marks exactly the overlapping
  tiles (±1 tile halo) and that the composed generated image is bit-identical to
  the monolithic-shader output within the moving region and equals `currentReal`
  in the static region.
- Run the whole path under `VK_LAYER_KHRONOS_validation` with **synchronization
  validation** enabled; zero WAR/RAW/ownership warnings is a gate. Specifically
  exercise the transfer↔compute ownership transfer and the indirect-args buffer
  barrier.
- Fuzz `f` from 0 to 1 (moving-rectangle sweep) and confirm no crashes, no
  validation errors, and monotonic cost.

**Counters / logs (extend the existing `g_bFramegenDebug` / dual-gpu-route
logging near ~4569)**
- `movingTiles`, `totalTiles`, `movingFraction` per generated frame.
- `classifyUs`, `generateUs`, `sdmaFillUs` (GPU timestamps via a query pool
  around each pass).
- `sdmaAvailable` (bool) and whether the compute fallback fired.
- `discarded_generation_too_slow` already exists (~4312/4494); add a breakdown
  of whether the transfer or compute engine was the straggler.
- A one-shot init log line reporting the discovered `m_transferQueueFamily`
  (mirroring the queue-family log at ~504).

**Performance proof**
- Compare package power and iGPU busy% on a static menu with framegen on,
  before vs. after this proposal — expect a large drop in iGPU busy and package
  power at equal output cadence.
- Confirm real-frame frametime distribution is **unchanged** (the invariant):
  the p99 real-frame frametime must not regress relative to the monolithic
  generator.
- Cross-PCIe traffic counter (from the dual-GPU route diagnostics) must be
  unchanged — this proposal adds none.

---

## Incremental rollout / flags

- The base feature is still gated by `g_bExperimentalFramegen` +
  `g_nFramegenMultiplier == 2` (`vulkan_framegen_is_enabled`, ~4246).
- New CLI/env flags:
  - `--framegen-tile-classify` / `GAMESCOPE_FRAMEGEN_TILE_CLASSIFY=1` — enable
    the classify + indirect-generate path (default off initially, then default
    on once validated). When off, the monolithic dispatch at ~4560 is used
    verbatim.
  - `--framegen-sdma-fill` / `GAMESCOPE_FRAMEGEN_SDMA_FILL=1` — enable the
    dedicated-transfer static fill. Auto-disabled if no transfer-only family;
    when off, static fill uses the compute queue.
  - `GAMESCOPE_FRAMEGEN_TILE_SIZE` (default 32) and
    `GAMESCOPE_FRAMEGEN_MOTION_THRESHOLD` for tuning/debug.
- **Fallback chain:** (tile-classify off) → monolithic shader;
  (tile-classify on, sdma off/unavailable) → indirect generate + compute static
  fill; (full path) → indirect generate + SDMA static fill. Each layer is
  independently switchable so a regression can be bisected without disabling
  framegen entirely.
- Reuse the existing dormancy/hysteresis machinery unchanged (~4478-4510): the
  optimization only changes work *inside* a generation that the scheduler has
  already decided to perform.

---

## Open questions

1. **Tile size.** 16 (finer motion masks, more atomics, more indirect groups)
   vs. 32 (coarser, cheaper classify, better SDMA region coalescing in strategy
   B) vs. 64. 32 aligns to two 8x8 generate workgroups per tile edge and matches
   common SDMA copy granularity; needs profiling per resolution.
2. **Strategy A vs. B threshold.** Is the full-image DMA double-write ever worse
   than region-list copies in practice? Likely only when `f` is large *and* SDMA
   bandwidth is the bottleneck — but at large `f` the whole optimization yields
   little anyway. Measure before adding the region-list complexity.
3. **Halo width for moving tiles.** Neighborhood rectification reads a small
   window; a 1-tile halo is probably sufficient, but disocclusion at tile edges
   might want 2. Tie to the rectification kernel radius.
4. **HDR metric domain.** Is a PQ/log-domain delta worth the extra math, or does
   a luma-scaled linear threshold suffice across the SDR/HDR range?
5. **Reusing the motion map for T2.6.** The per-tile flag is boolean here; the
   motion-estimation mode wants per-tile *vectors*. Should classify optionally
   emit a coarse motion vector (a few candidate offsets, cheapest SAD) so both
   features share one pass, or keep them separate to avoid coupling their
   rollout?
6. **Transfer-queue global priority.** Should the SDMA submit ever contend with
   other DMA (swapchain acquire copies, screenshot/pipewire capture at ~3243,
   ~3847)? If capture and framegen fill share the one transfer queue, does
   capture need priority, or a second transfer queue instance?
7. **Timeline unification.** Is it cleaner to make the compute generate wait on
   the transfer timeline (single `generatedSeqNo` gate on the present side), or
   to track two completion points and `max()` them in
   `vulkan_framegen_consume_generated_frame`? The former keeps the present-side
   check at ~4310 untouched.
