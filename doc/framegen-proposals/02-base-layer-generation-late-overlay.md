# Proposal 2 — Base-layer frame generation with late overlay/cursor composite

Status: design / implementation-oriented
Scope: `src/rendervulkan.cpp`, `src/rendervulkan.hpp`, `src/steamcompmgr.cpp`, `src/Backends/DRMBackend.cpp`, `src/shaders/cs_framegen_extrapolate.comp`
Depends on: the existing experimental framegen path (commits `313b9af`, `c33daa3`).

---

## Motivation / problem it solves

The current generated-frame algorithm runs on the **final composited output** — the
image produced at the very end of `vulkan_composite()` in `src/rendervulkan.cpp`. The
hook is at line ~4876:

```cpp
if ( !GetBackend()->UsesVulkanSwapchain() && !partial && pPipewireTexture == nullptr && pOutputOverride == nullptr )
    framegen_record_real_frame( compositeImage, frameInfo );
```

`compositeImage` (line ~4657-4661) is the swapchain/output-sized image *after* FSR
EASU→RCAS upscale and *after* every overlay layer (HUD, Steam overlay, cursor,
notifications) has been blended on top (the RCAS dispatch at line ~4702-4711 binds
all layers via `bind_all_layers` and writes `compositeImage` at
`currentOutputWidth × currentOutputHeight`). Two concrete problems follow:

1. **Generation cost scales with output resolution.** History copies
   (`copyImage`, line ~4518) and the extrapolate/blend dispatch (line ~4560) run at
   full output size — 3840×2160 on a 4K display. At fp32 RGBA that is ~33 MB per
   history image and ~99 MB of image traffic per generated frame just for the two
   input reads plus one write, before any TAA neighborhood taps. On a bandwidth-
   constrained iGPU/second GPU this is the dominant term and the first thing to
   starve when the compositing GPU runs out of headroom (constraint 4).

2. **UI/HUD/cursor smear — the classic framegen artifact.** Because overlays are
   already baked into `compositeImage`, the pixel-space extrapolation in
   `cs_framegen_extrapolate.comp` treats a moving mouse cursor and a static HUD the
   same way it treats game pixels: it extrapolates them. A cursor that the user
   flicks across the screen ghosts/tears on every generated frame; a HUD counter
   shimmers. These elements have *no motion model* that pixel extrapolation can get
   right, and they are exactly the high-contrast, high-frequency content that the
   neighborhood-rectification clamp (shader line ~64-66) cannot rescue.

**The upgrade:** generate on the **base game layer** (`frameInfo->layers[0]`) at its
native pre-upscale resolution (e.g. 2560×1440), then push the generated base frame
through the *same* EASU/RCAS pipeline to output resolution, and composite the real
overlay layers *on top* of the generated+upscaled base at their current positions.
Wins: generation bandwidth drops by `(base/output)²` (≈0.44 for 1440p→4K), and
overlays/cursor are pixel-perfect on generated frames because they are never
extrapolated — they are re-composited fresh.

---

## Design overview

Three structural changes, none of which touch the real-frame critical path:

1. **Record history at the base layer, not the output.** `framegen_record_real_frame`
   copies and predicts from `frameInfo->layers[0].tex` (the game buffer,
   pre-upscale) instead of `compositeImage`. History textures, the extrapolate
   dispatch, and the `framegenOutputImages` ring all shrink to base resolution.

2. **Snapshot the composite recipe.** At record time we snapshot enough of the
   `FrameInfo_t` to reconstruct the post-base composite deterministically: the
   base layer's `scale`/`offset`/`colorspace`/`filter`/`ctm`/`hdr_metadata_blob`,
   the `useFSRLayer0`/`useNISLayer0`/`blurLayer0` flags, `outputEncodingEOTF`,
   `applyOutputColorMgmt`, and the overlay layers `layers[1..layerCount-1]`
   (holding `Rc<CVulkanTexture>` refs keeps their buffers alive). This snapshot is
   the "generated FrameInfo template."

3. **Present through the real composite pipeline.** When the backend consumes a
   generated frame, instead of flipping a single-layer NEAREST full-res buffer
   (current `DRMBackend.cpp` line ~3618-3648), it substitutes the *generated base
   texture* into `layers[0]` of the snapshot template and calls the normal
   `vulkan_composite()`. EASU/RCAS upscales the generated base identically to a real
   frame, and `bind_all_layers` blends the overlays on top. The cursor layer can be
   re-fetched at its **latest** position at this moment (the "late cursor"
   composite), so the pointer stays crisp and responsive even on generated frames.

The base-layer texture is already the natural input: FSR's EASU dispatch reads
`frameInfo->layers[0].tex` and writes `g_output.tmpOutput` at
`layers[0].integerWidth()/integerHeight()` (line ~4670-4700). We are simply
inserting our prediction *before* that dispatch on the generated-frame path.

---

## Vulkan mechanisms & extensions

This proposal deliberately adds **no new hard Vulkan dependency** — it reuses the
mechanisms the base feature already relies on (constraint 3). Concretely:

- **Compute pipelines & storage images.** The predictor is a compute shader
  (`SHADER_TYPE_FRAMEGEN_EXTRAPOLATE` / `_BLEND`), dispatched via
  `CVulkanCmdBuffer::dispatch`. Generated targets are `VkImage`s created with
  `VK_IMAGE_USAGE_STORAGE_BIT` + `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT`-equivalent
  (`createFlags.bStorage`, `bOutputImage`, `bFlippable` in
  `framegen_create_output_texture`, line ~4362-4371). Under this proposal the base
  history textures (`framegen_create_texture`, `bSampled|bTransferSrc|bTransferDst`)
  become base-sized; the *ring that the generated base is written into* becomes a
  new base-sized, `bStorage|bSampled` ring (it no longer needs `bFlippable`/
  `bOutputImage` because it is now an *input to* EASU, not a scanout buffer).

- **`vkCmdCopyImage` (`VkImageCopy`).** History rotation
  (`CVulkanCmdBuffer::copyImage`, line ~4518) copies base→base. Because source and
  dest are identical base extents/format, this stays a plain copy (no blit,
  `VK_FILTER` irrelevant).

- **Timeline-style completion tracking.** The design keeps the existing
  `g_device.submit()` → monotonic `uint64_t` sequence + `g_device.hasCompleted(seq)`
  / `vulkan_wait(seq, false)` bookkeeping (backed by `VkFence`/timeline semaphore in
  the device layer). Generation is submitted in its **own command buffer, after**
  the composite's `g_device.submit()` (line ~4868), so the composite's present fence
  never waits on generation. No change here — the base-layer variant submits the
  same way.

- **EASU/RCAS reuse.** `SHADER_TYPE_EASU` and `SHADER_TYPE_RCAS`
  (`ffx_fsr1.h`-derived) are invoked exactly as in the real path
  (line ~4690-4711) with `EasuPushData_t`(inputX,inputY,tempX,tempY) and
  `RcasPushData_t`. The generated base's `integerWidth()/integerHeight()` reproduce
  the same `tempX/tempY`, so `update_tmp_images()` reuses `g_output.tmpOutput`
  without reallocation.

- **Per-layer color management.** `bindColorMgmtLuts` (shaper + 3D LUT per EOTF,
  line ~4665-4666), `colorspaceMask()`, `ycbcrMask()`, and `outputTF` are consumed
  by the RCAS/BLIT pipeline key exactly as for a real frame — the generated base
  carries the real base layer's `colorspace`/`ctm`, so HDR/scRGB paths are handled
  by the existing shader variants rather than a bespoke path.

- **Optional accelerators (opt-in only).** The Tier-2 motion-compensated mode may
  optionally use `VK_KHR_shader_subgroup`/`VK_EXT_subgroup_size_control` for
  block-matching reductions and `VK_KHR_16bit_storage` + `shaderFloat16`
  (`VK_KHR_shader_float16_int8`) for fp16 luma pyramids. These are *accelerators*
  with an fp32/no-subgroup fallback; the base feature must run on a device that
  exposes only core 1.2 compute. No vendor extension (NVX/AMD proprietary optical
  flow) is ever a hard dependency.

Sync primitive summary for the generated-present composite: it is an ordinary
`vulkan_composite()` submission on the compute/graphics queue, gated only by
`hasCompleted(generatedSeqNo)` (the prediction must have finished) before the EASU
read — see Data & control flow.

---

## Integration points in gamescope

All line numbers are from the current tree; treat them as regions.

| Site | File / function | Change |
|---|---|---|
| Record hook | `rendervulkan.cpp` `vulkan_composite`, line ~4876 | Pass `frameInfo->layers[0].tex` (base) *and* a `FrameInfo_t` snapshot to `framegen_record_real_frame`, instead of `compositeImage`. Still gated on `!partial && pPipewireTexture==nullptr && pOutputOverride==nullptr`. |
| History sizing | `rendervulkan.cpp` `framegen_ensure_resources`, line ~4373-4414; `FramegenHistory_t`, line ~4210-4235 | Size `previousReal`/`currentReal` and the generated ring to the **base** width/height/format. Add snapshot fields (see below). |
| Predict dispatch | `rendervulkan.cpp` `framegen_record_real_frame`, line ~4533-4567 | Dispatch at base extent; write generated base into a `bStorage|bSampled` base-sized ring image (renamed from `framegenOutputImages`, or a parallel `framegenBaseImages`). |
| Base texture source | already available: `frameInfo->layers[0].tex` is the pre-EASU game buffer (EASU reads it at line ~4692). | No new capture pass needed. |
| Snapshot struct | `rendervulkan.hpp` near `FrameInfo_t` (line ~280-386) | New `FramegenComposeSnapshot_t` (see Data flow). |
| Consume/present | `DRMBackend.cpp` line ~3613-3648 | Replace the single-layer NEAREST synthesis with: take the snapshot template, set `layers[0].tex = pGeneratedBase`, optionally refresh cursor layer, call `vulkan_composite(&snapshotFrameInfo, …)` to produce a real output image, then `drm_prepare`/`Commit`. |
| Consume API | `rendervulkan.cpp` `vulkan_framegen_consume_generated_frame`, line ~4299-4331 | Return the generated **base** texture (+ a pointer/handle to the snapshot) rather than a scanout-ready output. Actual upscale+overlay composite now happens at present time. |
| Slot scheduler | `steamcompmgr.cpp` line ~9131-9155 | Unchanged in structure: still discards a stale generated frame on `hasRepaint`, still marks the vblank slot. The generated slot now costs one extra EASU/RCAS pass at present time (accounted below). |
| VRR / tearing gates | `steamcompmgr.cpp` line ~2566-2570 (`allowVRR`), line ~9037-9040 (tearing) | Unchanged — framegen still forces fixed-cadence, non-tearing, non-VRR while active. |
| WaylandBackend | `WaylandBackend.cpp` ~1100 (nested present) | Mirror the DRM consume change: route the generated frame through the same snapshot→`vulkan_composite` path before the nested swapchain present. |

Note on the base-layer texture format: `layers[0].tex` for a real game frame is a
DMABUF-imported client buffer (created via `vulkan_create_texture_from_dmabuf`). The
history/generated-base ring must be an **internally allocated** image of matching
extent/format that we own and can `copyImage` into and `dispatch` onto — we never
predict *into* the client's buffer.

---

## Data & control flow

### Real frame (record path — off critical path)

1. `paint_all` (`steamcompmgr.cpp` line ~2510) assembles `FrameInfo_t`: base layer
   at `layers[0]` with `scale = base.scale` (line ~2043-2044), `useFSRLayer0` set at
   line ~2639 when `needsScaling` (`layers[0].scale < 0.999`) and filter is FSR;
   overlays appended at `layers[1..]` (Steam overlay ~2690, notification ~2732,
   cursor ~2748).
2. `vulkan_composite` runs the real EASU→RCAS→overlay composite into `compositeImage`
   and submits it (`sequence = g_device.submit(...)`, line ~4868). **The real frame's
   present waits only on `sequence`.**
3. `framegen_record_real_frame(layers[0].tex, frameInfo)` runs *after* that submit
   (line ~4876). It:
   a. Rejects overlay-only repaints via base-texture pointer identity (line
      ~4434-4441) — unchanged.
   b. Ensures base-sized history (`framegen_ensure_resources`).
   c. Copies `layers[0].tex → currentReal` (base→base, ~14 MB fp32 at 1440p).
   d. **Snapshots the compose recipe** into `FramegenComposeSnapshot_t`:
      `bool useFSR/useNIS/blur; EOTF outputEncodingEOTF; bool applyOutputColorMgmt;`
      the base `Layer_t` sans `tex` (scale/offset/opacity/filter/colorspace/ctm/
      hdr_metadata_blob/blackBorder), and a copy of `layers[1..layerCount-1]` with
      their `Rc<CVulkanTexture>` refs. This is a value copy of a `FrameInfo_t`
      minus the base texture — cheap, no GPU work.
   e. In a **separate command buffer**, binds `previousReal` (slot 0) and
      `currentReal` (slot 1), dispatches the extrapolate/blend shader at base extent
      into `generatedBase = framegenBaseImages[nOutImage % N]`
      (line ~4548-4560, extents now base-sized).
   f. `generatedSeqNo = g_device.submit(cmdBuffer)`; stash `pendingGenerated =
      generatedBase` and the snapshot. Rotate history (swap prev/current).
      `force_repaint()`.

### Generated frame (present path — fills an empty vblank)

7. Scheduler (`steamcompmgr.cpp` line ~9131): if no new real content latched and it
   is a vblank, `bShouldPaint = true` for the generated slot.
8. Backend present (`DRMBackend.cpp` line ~3613): `vulkan_framegen_consume_generated_frame()`
   returns the generated **base** texture *iff* `hasCompleted(generatedSeqNo)`
   (line ~4310) — otherwise it is discarded ("generation_too_slow") and the display
   simply repeats the last real frame. **The present path never stalls on
   generation.**
9. The backend takes the stashed snapshot, sets `snapshot.layers[0].tex =
   pGeneratedBase`, and — for the **late cursor** — overwrites the cursor overlay
   layer with the pointer's *current* position/texture (re-query
   `pFocus->cursor`), so pointer latency on a generated frame is one composite, not
   half a real-frame interval.
10. It calls `vulkan_composite(&snapshotFrameInfo, …, pOutputOverride=<a generated
    output image>, increment=false)`. This runs EASU (generated base →
    `tmpOutput`) → RCAS + `bind_all_layers` (overlays) → output image, identical to
    a real composite. Crucially `framegen_record_real_frame` is **not** re-entered
    here because `pOutputOverride != nullptr` (guard at line ~4875), so a generated
    frame never poisons history.
11. `drm_prepare(&g_DRM, bAsync, &snapshotFrameInfo)` with `allowVRR=false` (fixed
    slot), then `Commit`. Generated frames create **no Wayland commit** and emit no
    `frame_done`/presentation feedback (constraint 2) — they are produced entirely
    inside the compositor from already-latched buffers.

Why it stays off the real-frame critical path:
- Prediction (step 3e) is a trailing submit; the real present fence is independent.
- The generated composite (step 10) only ever runs in a vblank the game left
  **empty** — there is no real frame competing for that slot. If a real frame does
  arrive, the scheduler discards the generated frame (line ~9140-9141) *before* any
  generated composite is issued.
- If the GPU is oversubscribed, `hasCompleted` gates both the next record
  (line ~4494) and the consume (line ~4310): the feature goes dormant instead of
  queuing work in front of a composite (constraint 1 & 4).

---

## Latency & throughput analysis

Let output = 3840×2160, base = 2560×1440 (1.5× FSR). Base/output area ratio =
`(2560·1440)/(3840·2160) = 0.444`.

**Generation-side (record path, off critical path):**

| Term | Output-space (current) | Base-space (this proposal) |
|---|---|---|
| history `copyImage` (fp32 RGBA) | 3840×2160×16 B = 33.2 MB R+W ×2 | 14.7 MB R+W ×2 → **−56%** |
| predict dispatch reads (2 inputs + 4 TAA taps, cached) | ~2× full image ≈ 66 MB | ~29 MB → **−56%** |
| predict dispatch write | 33.2 MB | 14.7 MB |
| resident ring (3 gen images + 2 history) | 5 × 33 MB ≈ 166 MB | 5 × 14.7 MB ≈ 74 MB |

With fp16 (Tier-1) halve all of the above again. On a cross-PCIe dual-GPU route the
history-copy source is the game buffer already resident on the **compositing** GPU
(gamescope imports the client DMABUF and composites locally), so base-layer
recording does **not** add PCIe traffic beyond what the composite already pulls;
it *reduces* the local VRAM bandwidth of the prediction by ~56%.

**Present-side (generated composite) — the new cost.** The base-layer approach moves
the EASU+RCAS+overlay work onto the generated slot as well. That is one extra
EASU (base→temp) + one RCAS (temp+overlays→output) dispatch per generated frame.
This is *identical in cost to the FSR portion of a real composite* and runs only in
an otherwise-empty vblank, so it does not contend with a real frame. Net added
latency to a real frame: **zero** (unchanged from the current design). Net added
work per generated frame vs. the current output-space design: +1 EASU +1 RCAS pass,
paid out of idle GPU time; offset by the −56% cheaper prediction. On a headroom-
starved GPU the `hasCompleted` gate (step 8) drops the generated frame rather than
letting the extra composite bleed into the next real frame's budget.

Throughput headroom check: the generated composite must finish within the vblank
slot it fills (e.g. 8.33 ms at 120 Hz half-cadence of a 60 fps game). EASU+RCAS at
4K is well under 1 ms on any GPU capable of running gamescope's FSR path at that
resolution in the first place, so the slot budget is comfortable; the binding
constraint remains the *prediction*, which is now cheaper.

---

## Interaction with VRR / HDR / anti-lag / tearing

- **VRR.** Unchanged: `frameInfo.allowVRR = cv_adaptive_sync &&
  !vulkan_framegen_is_enabled()` (`steamcompmgr.cpp` line ~2570). Framegen fills
  fixed vblank slots; VRR and framegen remain mutually exclusive. The generated
  composite is submitted with `allowVRR = false` in the snapshot template.
- **HDR / per-layer colorspace.** This is where base-layer generation is strictly
  *more correct* than the current design. Today the generated frame is presented
  with `applyOutputColorMgmt=false` and a single `HDR10_PQ`/`SRGB` colorspace tag
  (`DRMBackend.cpp` line ~3621-3633) — a re-interpretation of already-color-managed
  output. Under this proposal the generated base carries the real base layer's
  `colorspace` and `ctm` (scRGB→2020 matrix `s_scRGB709To2020Matrix` when
  applicable, mirroring `paint_cached_base_layer` line ~2056-2057), and the RCAS/
  BLIT pipeline applies the *same* shaper/3D LUTs and `outputTF` via the normal
  `colorspaceMask()`/`bindColorMgmtLuts` path. HDR metadata (`hdr_metadata_blob`)
  from the snapshot flows through. History is still invalidated on an EOTF change
  (`framegen_record_real_frame` line ~4450-4455) so we never predict across an
  SDR↔HDR transition. Prediction happens in the base layer's encoding
  (game buffer space), which is the correct space for motion — extrapolating in
  post-tonemap output space (current behavior) distorts motion in HDR highlights.
- **Anti-lag / Reflex / DXVK.** Unchanged and central: generated frames still emit
  no client `frame_done` or presentation feedback and create no Wayland commit
  (step 11), so they are invisible to the game, Wine/DXVK, Reflex and anti-lag
  (the critical rule). Late-cursor re-composite reads gamescope's own cursor state,
  not the client's, so it introduces no client-visible feedback.
- **Tearing.** Unchanged: tearing and framegen are exclusive
  (`steamcompmgr.cpp` line ~9037-9040). Generated frames must land on vblanks.

---

## Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Overlay stack changed between record and present (HUD appears, layer count changes) | med | History already invalidates on layer-count change (line ~4450). If the change happens *after* record but before present, discard the pending generated frame (the snapshot is stale) — reuse the `superseded_by_real_frame` path (line ~9140). |
| Base layer isn't `layers[0]` (video underlay path, `bHasVideoUnderlay`, line ~2602) | med | Only record when `layers[0]` is the base game plane (`PaintWindowFlag::BasePlane`). Gate `framegen_record_real_frame` on the same predicate paint_all uses; otherwise stay dormant. |
| No FSR active (`needsScaling==false`, native-res game) | low | Snapshot `useFSRLayer0=false`; generated composite falls through to the plain BLIT path (line ~4809) — overlays still composite correctly over the generated base at 1:1. Generation just runs at output res in that case (same as today). |
| Cursor re-fetch at present time races cursor state on another thread | med | Take the cursor snapshot under the same lock `paint_all` uses; if unavailable, fall back to the recorded cursor layer (no regression vs. today). |
| Generated base texture lifetime (client buffer recycled) | med | The generated base is an **internally owned** ring image, never the client DMABUF; `Rc` refs on overlay textures in the snapshot keep those alive until the generated frame is presented or discarded. |
| Extra EASU/RCAS pass on generated slot pushes into next real frame under load | med | `hasCompleted(generatedSeqNo)` gate (step 8) + the `gpu_oversubscribed` record gate (line ~4494) drop generation before it can contend; degrade to hardware repeat. |
| Block-matching MVs polluted by RCAS ringing (Tier-2) | low | Doing MC at **base** resolution operates on pre-sharpen game pixels, *avoiding* the RCAS ringing that would corrupt block matching on the output image — a positive side effect, not a risk. |
| scRGB/float base with `predicted.a` handling | low | Shader already pins alpha to current frame (line ~73) and only neighborhood-clamps RGB; base-space prediction preserves HDR highlights >1.0 and wide-gamut negatives (shader comment line ~68-72). |
| Two output-image pools (real swapchain vs. generated composite target) exhausting VRAM | low | Reuse the existing `framegenOutputImages` ring (line ~558, `std::array<…,3>`) now sized to **output** res for the generated composite target, while the new base ring is separate and small. Net VRAM roughly neutral vs. today. |

## Testing & validation plan

Counters/logs to add (behind `g_bFramegenDebug`, mirroring existing
`vk_log`/`xwm_log` lines like 4569-4578, 9146-9154):

- `framegen: base record id=… base=%ux%u output=%ux%u fsr=%d` — confirm base
  extents and that FSR is snapshotted.
- Per-generated-frame timing: prediction dispatch µs, generated-composite (EASU+RCAS)
  µs, and slot classification (extend the existing `slot=real|generated|repeat`
  log at line ~9151-9154).
- Counters: generated-presented, discarded-`generation_too_slow`,
  discarded-`superseded_by_real_frame`, dormant-`gpu_oversubscribed`,
  `late_cursor_refreshed` count.

What to measure / how to prove correctness:

1. **No real-frame latency regression.** Instrument the composite `submit`→present
   fence interval; assert it is statistically identical with framegen on vs. off
   (real-frame path untouched). Use `gpuvis`/`gpuvis_trace` markers already present
   in `paint_all` (line ~2527, ~3182).
2. **Cursor sharpness.** Capture generated frames (via the existing screenshot/
   pipewire path — note it bypasses framegen, so add a debug capture of the
   generated output image) while flicking the cursor; verify the cursor is a pixel-
   exact overlay with zero ghost, and that a HUD counter reads a stable value on
   generated frames (it should equal the last real frame's HUD, not a smear).
3. **Bandwidth.** Confirm predicted −56% VRAM traffic via GPU perf counters
   (e.g. `amdgpu` `GRBM`/`radeontop` or `nvidia-smi dmon`) comparing base vs.
   output-space generation at 1440p→4K.
4. **HDR correctness.** With an HDR (`EOTF_PQ`) title, verify generated frames match
   the color of neighboring real frames (no hue/brightness pop), proving the shaper/
   3D LUT + `outputTF` path is applied identically. Verify EOTF-change invalidation.
5. **Oversubscription behavior.** Cap GPU clocks / run a stress load; confirm the
   feature goes dormant (counters) and the display cleanly repeats rather than
   dropping a real frame.
6. **Validation layers.** Run with `VK_LAYER_KHRONOS_validation` + sync validation
   to prove the extra generated composite introduces no missing barriers between the
   prediction write (storage) and the EASU read (sampled) of the generated base.

## Incremental rollout / flags

- Keep the existing gate `vulkan_framegen_is_enabled()` (line ~4246) and
  `g_bExperimentalFramegen` / `g_nFramegenMultiplier` / `g_eFramegenMode`.
- New sub-flag `--framegen-base-layer` (or ConVar `cv_framegen_base_layer`,
  default **on** once stable) selecting base-layer vs. legacy output-space
  generation. Default off during bring-up so the two paths can be A/B'd on the same
  build.
- New `--framegen-late-cursor` (ConVar, default on) toggling the present-time cursor
  re-fetch; off falls back to the recorded cursor layer.
- **Fallback ladder** (all automatic, no user action): base-layer generation →
  (no FSR active) output-res generation over the base → (`gpu_oversubscribed` /
  `generation_too_slow`) dormant, hardware repeat → (feature disabled) normal
  compositing. Every rung is already reachable through existing invalidate/discard
  reasons; the new flags only choose the *top* rung.
- Reuse existing `vulkan_framegen_reset`/`invalidate_history` reasons for
  observability; add `base_layer_toggle` as a reset reason so flipping the flag at
  runtime cleanly rebuilds the (differently-sized) history/ring.

## Open questions

1. **Present-time composite ownership.** Should the generated composite run on the
   dedicated lower-priority framegen queue (Tier-1) or the main composite queue? It
   only runs in empty vblanks, but on the low-priority queue it can be preempted by
   a late-arriving real composite — arguably the safest choice for constraint 1.
2. **Snapshot vs. live overlays.** For non-cursor overlays (notification fade, HUD),
   do we re-fetch at present time (freshest, but risks a mid-flight layer-count
   change) or freeze the snapshot (simplest, one frame stale)? Proposal freezes all
   but the cursor; revisit if HUD staleness is visible at high multipliers (x3/x4).
3. **NIS path.** EASU/RCAS reuse is clean; NIS (`useNISLayer0`, line ~4713-4768)
   uses a temp image + a second BLIT with a mutated `nisFrameInfo`. The generated
   present must replicate that two-step; confirm the snapshot carries enough to
   rebuild `nisFrameInfo` (it does: it's derived from the same `frameInfo`).
4. **x3/x4 multiplier interaction.** With multiple generated frames per real
   interval (Tier-1), each needs its own EASU/RCAS pass. Does the slot budget hold at
   4K/240 Hz? Needs measurement (§3 above) before enabling >x2 with base-layer FSR.
5. **Partial-overlay output pool.** The `outputImagesPartialOverlay` path
   (line ~4661) is excluded from recording today; confirm base-layer generation
   never needs to target it, or add a parallel partial generated target.
6. **Base layer with black borders / letterboxing** (`blackBorder`, integer offset):
   confirm EASU on the generated base honors the same border mask so generated and
   real frames letterbox identically.
