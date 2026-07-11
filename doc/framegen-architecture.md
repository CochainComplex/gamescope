# Compositor-Side Frame Generation — Architecture

The definitive engineer's map of gamescope's compositor-side frame generation.

Line anchors are current-tree as of commit `1b949f0`; the code is refactored often, so
**anchor to symbols, not line numbers** when they drift. Design-document line anchors
(`doc/framegen-proposals/`) have drifted post-refactor — reference those by symbol/section.

---

## 0. Scope and the one organizing fact

Upstream fork point is **`a563226`**. The framegen work is the commit arc
`313b9af → c33daa3 → 528ee87 → 996ef05 → 992f8d9 → fd7be06 → 32a28a1 → a75bfbe → 2a3a6c0 → 1b949f0`
— **32 files, ~6,200 insertions**. `src/rendervulkan.cpp` (+2264) is the engine, 16 compute
shaders are the algorithmic surface, `src/steamcompmgr.cpp` (+114) is the present arbiter, the
backends + FROG WSI layer carry the dual-GPU/present-timing plumbing, and
`doc/framegen-proposals/` holds six roadmap docs (the #01 VRR-hybrid, #02
base-layer and #06 JIT prototypes are uncommitted working-tree changes on top
of `1b949f0`).

**Target topology (drives every design choice):** the game renders on the *strong* GPU
(NVIDIA RTX PRO 500 / AMD 7900 XT) → the frame is shared over dma-buf → the **weak AMD
compositor GPU (890M / 5700 XT) composites, generates the in-between frames, and presents.**
Generation bandwidth therefore runs on the *weak* card. The load-bearing performance truth is
that **these passes are memory-bandwidth-bound and ALU-light** — nearly every optimization below
is a fight over history-read bandwidth on that weak card.

**Four invariants the whole design upholds** (enforcement points in §5) — the default forward-extrapolation
path; the opt-in **bidir** mode (`GAMESCOPE_FRAMEGEN_BIDIR=1`, §3.3) deliberately relaxes #1 and #4:
1. Real frames are **never** delayed by generation. *(Bidir is the exception: it presents each real
   frame one interval late — the intrinsic cost of interpolating between two reals, hence opt-in.)*
2. Generated frames are **never** waited on or presented late.
3. A real frame latching **always** discards any pending generated frame (latency safety).
   *(Bidir preserves queued real frames instead — its pending queue is a presentation timeline, §3.3.)*
4. Generation is **forward extrapolation** (predict the future beyond the current real frame,
   zero added latency) — *not* interpolation. (The debug-only `blend` mode and the opt-in **bidir**
   mode are the exceptions — bidir interpolates *between* the two reals and so adds one interval of latency.)

---

## 1. Entry points (the doors in)

### CLI flags (`src/main.cpp`)
| Flag | Global set | Notes |
|---|---|---|
| `--experimental-framegen` | `g_bExperimentalFramegen` | master enable; forces the full-composite path |
| `--framegen-mode {extrapolate\|motion\|blend}` | `g_eFramegenMode` | default `extrapolate`; `blend` is debug-only |
| `--framegen-multiplier {2,3,4}` | `g_nFramegenMultiplier` | validated, `exit(1)` on bad value; also *sizes the output pool* |
| `--framegen-strength <0.0-1.0>` | `g_flFramegenStrength` | forward step; **parsed with `atof`** — garbage silently → 0.0 (no error), unlike the multiplier |
| `--framegen-debug` | `g_bFramegenDebug`, `g_uFramegenDebugEvery` | see §8 observability |

Parse sites `main.cpp:139-160, 842-880`; benchmark early-exit `main.cpp:1094-1100`.

### Environment variables
- `GAMESCOPE_FRAMEGEN_BENCHMARK` — **presence-only** (even `=0` triggers); runs the microbench then `exit(0)` *before* output creation.
- `GAMESCOPE_FRAMEGEN_SINGLE_QUEUE` — needs a **truthy int** (`=0` does *not* disable); forces the shared-queue regime.
- `GAMESCOPE_FRAMEGEN_DEBUG_EVERY` — debug log rate (default 60; rejects 0/non-uint → 60).
- `GAMESCOPE_FRAMEGEN_JIT` — truthy int; opt-in **#06 JIT display-clock pacing** prototype (`rendervulkan.cpp:4864` `framegen_jit_enabled`). Plans one slot per vblank against the KMS pageflip clock instead of a baked k/gap batch. **Also gated on the dedicated framegen queue** — a no-op on the shared-queue fallback.
- `GAMESCOPE_FRAMEGEN_VRR_HYBRID` — truthy int; opt-in **#01 VRR hybrid** prototype (`rendervulkan.cpp:4878` `vulkan_framegen_vrr_hybrid_requested`). Real frames present VRR-style, one generated frame flips mid-interval on a timer. Gated on the dedicated queue **and** only *active* while the connector is actually in VRR (`IsVRRActive()`); otherwise it falls back live to the fixed-refresh paths.
- `GAMESCOPE_FRAMEGEN_BASE` — truthy int; opt-in **#02 base-layer generation** prototype (`rendervulkan.cpp:4919` `framegen_base_layer_enabled`). Generates on the pre-upscale game layer (`layers[0].tex`), then the present-time consume runs a **late composite** (`framegen_base_present_composite`, `:5350`) against the *live* `FrameInfo_t` — generated frames go through the full FSR/color pipeline and carry fresh overlays + latest cursor. Unlike JIT/hybrid it does **not** need the dedicated queue; it self-selects per frame via `framegen_base_layer_usable` (`:4939`: layer 0 is the base plane, non-YCbCr, no ReShade, format supports sampled+storage) and falls back **live** to output-space generation for unusable scenes.

Stage-B motion-quality knobs (**default ON**; `=0` disables for A/B attribution — see §3.2):
- `GAMESCOPE_FRAMEGEN_FB` — forward-backward consistency check (`framegen_fbcheck_enabled`, `:5644`; runs the coarse-to-fine matcher a second time reverse-anchored and kills confidence where the round trip doesn't close). `=0` disables.
- `GAMESCOPE_FRAMEGEN_FB_TOL` — float, clamped `[0.05, 8.0]`, default `0.75`; the FB round-trip tolerance in low-res texels (larger = more forgiving = fewer kills).
- `GAMESCOPE_FRAMEGEN_AGREE` — per-pixel two-source agreement test in the warp (kills `conf` at full res where the two real-frame projections of the flow read different content). `=0` disables.
- `GAMESCOPE_FRAMEGEN_BIDIR` — truthy int; opt-in **B3 bidirectional interpolation** (`vulkan_framegen_bidir_active`, `:5076`). Generated frames sit *between* the two reals (warp both toward phase `t`, blend by confidence, phase-correct crossfade fallback) instead of extrapolating past the newest — removes hold-then-jump judder and handles translucency, at the cost of presenting each real frame **one interval late** (§0 invariants #1/#4 exception; see §3.3). **Requires motion mode; mutually exclusive with `_JIT`/#06, `_VRR_HYBRID`/#01 and `_BASE`/#02** (each owns its own timeline); silently ignored otherwise (logs once).
- `GAMESCOPE_FRAMEGEN_ADAPT` — **B4 self-supervised adaptation** (`framegen_adapt_enabled`; motion mode). Each real frame grades the field that predicted it (field-res stats probe, 14–25 µs per real); a same-batch apply pass folds a global field-trust factor into the field's confidence, and the next-batch CPU readback auto-calibrates the FB tolerance and agreement window (see §3.4). `=0` disables (B3-bit-exact warps, no probe). An explicit `_FB_TOL` pins the tolerance against auto-calibration.
- `GAMESCOPE_FRAMEGEN_NET` — path to a weights blob; opt-in **Stage C learned forward-field refinement** (`framegen_net_weights_path`; motion mode, no `_BIDIR` requirement). A tiny fused-conv net (`cs_framegen_motion_net.comp`, 12→16→16→4, ~4.6k params) refines the checked causal field once per real frame — bounded flow residual (±2 field texels, tanh-limited) + evidence-gated confidence recalibration — before the B4 probe and forward warp consume it (`framegen_motion_field()`). Bidir additionally refines the reverse field. Trained offline by `scripts/framegen-net-train.py`; a zero-head blob is bit-neutral (= Stage B). See §3.5.
- `GAMESCOPE_FRAMEGEN_RECORD` — directory; **Stage C dataset capture** (motion mode, no `_BIDIR` requirement). Dumps raw field-res training tensors (both lumas + both checked fields, pre-refinement/pre-trust) one `GSFD` file per real frame, up to `GAMESCOPE_FRAMEGEN_RECORD_MAX` (default 1000, ~0.6–1.2 MB each — mind the disk).
- `GAMESCOPE_FRAMEGEN_NET_ONLINE` — truthy int; **C2 in-situ learning** (see §3.5). The refiner keeps training on the framegen GPU against every real frame; works with a `_NET` blob as the prior or from a synthesized neutral init. Without `_NET_PROFILE` the model is **ephemeral — nothing is ever written to disk**. `_NET_LR` (default `3e-4`), `_NET_EVERY` (train every Nth real frame, default 1) tune it; `_NET_PROFILE=<path>` makes the learning persistent per game: loaded as the prior when present (a malformed/torn file is rejected loudly and the neutral prior used instead), checkpointed back every 1024 trained steps **off-thread** (file I/O never rides the render thread) and flushed at exit and on every framegen reset, so sessions shorter than the checkpoint interval persist too. Writes are atomic (temp + rename): a crash or a full disk can never tear an existing good profile. The served weights are health-checked every trained step; a non-finite value re-initializes the optimizer state from the prior automatically and is never persisted.

### Runtime hot path
- **`framegen_record_real_frame()`** `rendervulkan.cpp:5682` — the scheduler; called from the tail of `vulkan_composite` (`:6216`) for base-layer, non-partial, non-pipewire, non-override, non-Vulkan-swapchain commits only.
- **Present arbiter** `steamcompmgr.cpp:9143-9196` — per-vblank: real vs generated vs HW-repeat.
- **`vulkan_framegen_consume_generated_frame(pFrameInfo)`** `rendervulkan.cpp:5126` — returns a scanout-ready generated frame, called from `CDRMBackend::Present` (`DRMBackend.cpp:3613`) / `CWaylandConnector::Present` (`WaylandBackend.cpp:1101`). Takes the live present `FrameInfo_t`: in **base-layer mode (#02)** it substitutes the generated base into `layers[0]` and runs the late overlay/cursor composite (`framegen_base_present_composite`); in output-space mode the arg is unused and it returns the pre-composited generated image as before.
- **`IBackend::SupportsFramegen()`** (`backend.h`, new virtual) — default false; DRM + nested-Wayland return true; `CDeferredBackend` delegates to its child (`DeferredBackend.h:224`). Gates the feature to backends that can actually consume generated frames and pay the forced-composite/no-VRR tax.

### Shader build wiring (`src/meson.build`)
The 22 framegen `.comp` shaders are compiled to SPIR-V C-arrays (`meson.build:69-97`) and registered unconditionally by the `SHADER()` macro table (`rendervulkan.cpp:1172-1205`; the Stage-B `MOTION_PYRAMID{,_RGBA}` / `MOTION_MATCH_REFINE` / `MOTION_FBCHECK`, the B3 `MOTION_BIDIR`, the B4 `MOTION_STATS{,_APPLY}` and the Stage-C/C2 `MOTION_NET{,_TRAIN,_OPT}` slots are the newest). **Load-bearing:** when `!supportsShaderFloat16`, the FP16 enum slots are **aliased to the fp32 SPIR-V arrays** (`:1175-1183`), so the dispatcher can name an fp16 `ShaderType` unconditionally with no null-pipeline branch at dispatch time.

---

## 2. End-to-end dataflow (one real frame)

1. **Client render (donor GPU) + WSI handoff.** Client renders on the strong GPU. The FROG WSI
   layer forces `VK_PRESENT_MODE_MAILBOX_KHR` unconditionally (`layer/VkLayer_FROG_gamescope_wsi.cpp:1220`)
   and only chains `VK_EXT_swapchain_maintenance1` structs when the driver supports **both** the
   extension and the feature bit (`deviceSupportsSwapchainMaintenance1()`, `layer:52`; gate at
   `CreateDevice` `layer:698`; per-swapchain memo `layer:542`). This is the NVIDIA cross-GPU fix
   (`2a3a6c0`): the unconditional force previously failed device creation on the NVIDIA proprietary
   driver with `VK_ERROR_INITIALIZATION_FAILED`, blocking any NVIDIA-render / AMD-present setup.
2. **dma-buf import to the compositor GPU.** DRM `drm_fbid_from_dmabuf` / Wayland
   `ImportDmabufToBackend`. Only `g_bDebugDualGpuRoute` diagnostics were added here
   (`DRMBackend.cpp:1630`, `WaylandBackend.cpp:2258`), no new import logic.
3. **Force full composite.** `bNeedsFullComposite |= vulkan_framegen_is_enabled()`
   (`DRMBackend.cpp:3579`, `WaylandBackend.cpp:1077`) — every real frame goes through
   `vulkan_composite` so a full-screen image lands in the output ring to serve as zero-copy
   history. This forced-composite tax applies **even when generation is dormant**; it is the cost
   the `SupportsFramegen()` gate contains.
4. **Composite on the AMD card.** `vulkan_composite` (`rendervulkan.cpp:5927`) records into the
   composite command buffer and returns a scratch-timeline `sequence`. The real frame's present
   waits **only** on `sequence` (`:6208-6216`).
5. **Framegen decision.** `framegen_record_real_frame` (`:5682`) runs the overlay-only filter,
   history-invalidation checks, the generate/dormant gate, multiplier adaptation, and the
   degradation-ladder step, then shifts zero-copy history `previousReal ← currentReal ←
   composited image` (`:5789-5791`).
6. **Generation on the dedicated queue.** If generating, `framegen_submit_batch` (`:5314`) records
   the whole interval into **one** `markFramegen()` command buffer, brackets it with GPU timestamps,
   and submits via `submitFramegen` (`:1857`), which waits the composite scratch-timeline value,
   `QueueSubmit`s to `m_framegenQueue`, and signals `m_framegenTimeline`. `framegen_submit_batch`
   appends `PendingGenerated_t` entries and pins `genReadA/B = previousReal/currentReal`,
   `genReadSeqNo = batch seqNo` (`:5417-5419`). Then `force_repaint()` (def `steamcompmgr.cpp:7171`)
   `Nudge()`s the compositor thread to run the empty-vblank iteration at all.
7. **Ring advance.** After composite + increment (`:6218-6253`), `nOutImage` advances but **skips**
   any slot equal to `currentReal`/`previousReal` (history) or `genReadA`/`genReadB` (in-flight
   cross-queue read), bounded by `nRing`. `nLastOutImage` records the true just-composited slot
   because the skip broke the implicit `nOutImage-1`.
8. **Present decision (per vblank).** `steamcompmgr.cpp:9143-9196`: real base content (`hasRepaint`)
   → discard the entire pending batch; else empty vblank + `generated_frame_ready()` (non-blocking
   peek) → latch a generated frame; else HW repeat.
9. **Present on empty vblanks.** `vulkan_framegen_consume_generated_frame()` (`:4969`) presents the
   generated texture as a **single full-screen base layer**: `opacity 1.0`, `NEAREST`,
   `applyOutputColorMgmt=false`, `allowVRR=false`, colorspace from `outputEncodingEOTF`. The Wayland
   path **hard-nulls planes 1..7** — overlays/cursor are explicitly dropped for a generated slot.
   Consume drops the front slot if `!hasCompletedFramegen` (never stalls) and kicks
   `framegen_refill_idle` (`:5442`) when the queue drains.

**Two clocks.** Framegen paces on `nFramegenRefreshMhz = g_nNestedRefresh ? g_nNestedRefresh :
g_nOutputRefresh` → `ulVblankIntervalNs = 1e12 / mHz` (`:5755-5756`), which must equal
`CVBlankTimer::GetRefresh()`. In nested mode `g_nOutputRefresh` is later overwritten with the
*parent monitor's* refresh while slots are still placed on `g_nNestedRefresh`'s cadence — deriving
the interval from `g_nOutputRefresh` there desyncs phase/strength and was the temporal wobble. On
DRM the two are equal (no-op).

---

## 3. The three algorithmic branches (the core)

The batch loop (`:5367-5391`) picks Motion / Blend / Extrapolate per `eff.mode`. Extrapolate fuses
slots in **pairs** (`extrapolatePair(i, i+1)`, odd trailing slot → single `extrapolate`). All
shaders are `local_size 8×8`. History bound via `framegen_bind_history` (`:5088`):
`s_samplers[0]=previousReal`, `[1]=currentReal`, both `setTextureSrgb(true)` (sRGB view →
hardware linearizes on read).

### 3.1 Comparison matrix

| Variant | Math | Memory strategy | Dispatched when | Tradeoff / cost |
|---|---|---|---|---|
| **Extrapolate LDS fp32** (`cs_framegen_extrapolate.comp`) | `delta = cur−prev`; `strength = u_strength·(1−smoothstep(0.08,0.40,motion))`, `motion = max(\|Δr\|,\|Δg\|,\|Δb\|)`; `predicted = cur + delta·strength`; full 3×3 neighbor clamp; **no [0,1] clamp**; alpha = `cur.a` | 10×10 LDS apron `s_cur[100]` of CURRENT only, staged once/group; diagonals add LDS reads but no image traffic | non-NVIDIA integer targets | preserves diagonal/thin motion better; halos bounded by suppression + neighbor clamp |
| **Extrapolate direct** (`_extrapolate_direct.comp`) | **bit-identical** to LDS fp32 | no LDS / no barrier; neighbors via `texelFetch` straight from texture cache | **NVIDIA only** (`vendorID==0x10DE`, `:5200`) | identical output; **~30–37% faster** on NVIDIA (large L2 makes the apron pure overhead); also beats fp16 there |
| **Extrapolate fp16** (`_extrapolate_fp16.comp`) | same math, `f16vec3` ALU; push uploaded fp32 then cast; alpha never demoted | **LDS apron stays vec4 fp32** — only ALU is fp16, bandwidth unchanged, so upside is capped | `supportsShaderFloat16 && !floatTarget`, non-NVIDIA | fp16 range/precision bands scRGB highlights → **gated off for float targets** |
| **Extrapolate pair (+ pair_fp16)** (`_extrapolate_pair.comp`) | writes TWO slots from one dispatch; suppress term + neighbor min/max computed **once** and shared | halves the two full-res history reads across the pair | x3/x4 | same quality; halves history bandwidth at high multipliers |
| **Blend** — DEBUG ONLY (`cs_framegen_blend.comp`) | `mix(prev, cur, phase)` full vec4 **including alpha**; **INTERPOLATION**; no suppression/rectification | normalized-coord **bilinear** `texture()` (the only framegen shader using sampler filtering) | `eff.mode==Blend` only | structurally violates invariant #4 → judder; reference lever |
| **Motion** (luma → match → warp) | see §3.2 | see §3.2 | `eff.mode==Motion && dispatch.motionSupported`; **priciest rung, shed first** | smooth pans; `conf=0` ≡ bounded extrapolate (never much worse); highest cost |

**Invariant across all extrapolate variants:** alpha pinned to `current`; the *only* range bound is
the 9-tap neighbor clamp — no `[0,1]` clamp, so UNORM stays in range while scRGB keeps HDR>1.0 and
wide-gamut negatives; previous frame sampled at center only.

### 3.2 Motion mode detail

Stages 1+2 run **once per real-frame pair** (`framegen_prepare_motion` `:5231`); stage 3 runs
**per slot** (`framegen_warp_slot` `:5292`). Falls back to extrapolate for the whole batch if
intermediates can't be allocated.

- **Stage 1 — luma pyramid** (`cs_framegen_motion_luma_pair.comp` +
  `cs_framegen_motion_pyramid.comp`): **4-tap bilinear box** to 1/8-res luma, both frames in one
  dispatch (`dstPrev=binding1`, `dstCur=binding2`), then two further 2× box steps (one bilinear tap
  each, both frames per dispatch) to 1/16 and 1/32 res. The box filter is a **wobble root-cause
  fix** — a single centered tap covers ~4/64 px, so sub-block translation aliases the low-res luma
  and the integer matcher toggles its winner frame-to-frame. Samples the reals as **sRGB → linear**
  before computing Rec.709 luma. Luma targets are R16F when
  `framegen_format_supports_sampled_storage(R16F)` (¼ bandwidth, no sRGB view to mis-decode), else
  ABGR16161616F (the `_rgba` shader twins).
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
  their confidence killed so the warp falls back to bounded extrapolation there. The check tests
  the **four covering reverse texels individually** (a bilinear blend across a motion boundary
  fails spuriously), forgives `tolBase + tolSlope·(|F|+|R|)` texels of error
  (`GAMESCOPE_FRAMEGEN_FB_TOL` overrides tolBase, default 0.75), and **feathers the resulting kill
  mask with a 3×3 tent** — a raw per-texel kill is an 8×8-px tile snapping to the static fallback
  while its neighbours warp, which reads as "tearing tiles" and flickers at the real-frame rate.
  `.a` carries the raw round-trip error for future per-tile quality feedback.
- **Stage 3 — warp** (`cs_framegen_motion_warp.comp`): `predictedMotion = texture(currentReal,
  uv − mvUv·strength)` (backward gather along the flow); `predictedExtra = cur + delta·strength`
  with suppression; `predicted = mix(predictedExtra, predictedMotion, conf)`; alpha = `cur.a`.
  **Key subtlety:** the TAA 4-neighbor clamp is applied **only to `predictedExtra`** — clamping the
  motion term would cap displacement at ~1 texel/frame and defeat the warp. *Confidence, not
  clamping, is the motion term's safety*: a poor match lowers `conf` and blends back toward the
  bounded extrapolation. The warp adds a **per-pixel two-source agreement test**
  (`GAMESCOPE_FRAMEGEN_AGREE=0` disables): the same flow is also projected from the previous frame
  (`uv − mvUv·(strength+1)`, one interval further along); where the two projections read different
  content the flow is wrong *for that pixel* and `conf` is killed at full resolution — the 1/8-res
  confidence field cannot make this call, and a mid-confidence mix of two positions is a double
  exposure (ghosted edges, doubled text). The distance is normalized by local magnitude so the
  threshold stays meaningful for scRGB values above 1.0.

Because `predicted = mix(predictedExtra, predictedMotion, conf)`, **motion is a strict superset of
extrapolate** — `conf=0` reproduces the bounded extrapolate output exactly, so it can never be much
worse even where matching fails.

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
  *cheaper* than the forward warp (0.095 vs 0.136 ms @1080p int on Blackwell).
- **Scheduling** (the substantive change): the pending queue becomes the **presentation
  timeline**. A recording real frame appends `[interp(k/gap)…, realN]` (never clears — nothing is
  stale; phases interpolate the *measured, just-completed* interval, so bidir never speculates
  past the gap), and the backends flip `vulkan_framegen_bidir_flip_texture(composite)` — in steady
  state the queue front at a real paint is the PREVIOUS real frame, so real frames present exactly
  one interval late (the intrinsic interpolation latency; why it's opt-in). Interps drain on
  repeat vblanks via the normal consume path. The supersede discard is skipped (steamcompmgr), and
  discard/invalidate paths preserve queued `bReal` entries (painted content the user hasn't seen);
  ring advance pins their slots. Composites outside the timeline snap it back to live: dormant/
  prime/keep-up frames flip immediately and drop stale pending — bidir degrades to zero added
  latency when the game keeps up. Overlay-only recomposites present the queue front instead (their
  game content is still queued; the overlay rides the next real frame). The degraded ladder rung
  routes to **blend** (phase-correct crossfade), never extrapolate — an extrapolated slot would
  come from a different timeline. Excluded: base-layer #02, JIT #06, VRR-hybrid #01 (own
  timelines); the classic batch path only.

### 3.4 Self-supervised online adaptation (B4, default on in motion mode)

Every real frame is ground truth for the field that was just estimated from it. After the last
consistency check, a field-res **stats probe** (`cs_framegen_motion_stats.comp`,
`framegen_record_adapt_probe`) warps the previous frame's luma along the final checked field and
compares it against the current frame's luma per texel — a direct measurement of "would this field
have predicted the frame we actually got", which is the exact bet every generated slot in the
coming interval places. Per-texel verdicts LDS-reduce into a 16×1 `R32_UINT` counter image
(total / residual sum / mispredicting-but-surviving count / killed count / static-texel noise
floor / an 8-bin histogram of the FB round-trip error B2 already stores in `field.a`). Cost:
14–25 µs per REAL frame (1080p–2160p), zero per generated frame.

Two consumers, two latencies:

- **GPU, same batch** (`cs_framegen_motion_stats_apply.comp`): a field-res apply pass folds a
  batch-global **field trust** factor into the field's confidence channel —
  `trust = 1 − smoothstep(0.15, 0.45, mispredicting/surviving)`. The signal isolates regimes where
  the per-texel checks *lie*: texels that passed FB + feathering yet still mispredicted (lighting
  flashes, particle chaos, stroboscopic content). Low trust slides the whole frame toward the safe
  fallback (crossfade in bidir, bounded extrapolation forward) — smoothly, with zero latency, and
  self-recovering the moment the field predicts again. This is deliberately how quality-driven
  degradation ships WITHOUT touching #04's monotonic ladder: a discrete quality rung would either
  oscillate or, held monotonic, let one bad interval degrade the whole scene. Perf note: baking
  trust into the field (which the warps already fetch per pixel) is load-bearing — reading the
  stats image from the full-res warps, even once per workgroup, measured ~10% of the warp on
  NVIDIA (repeated surface loads of one texel serialize on the cache).
- **CPU, next batch** (`framegen_adapt_consume`): the counters copy into a host-mapped readback
  image; the one-batch-in-flight guarantee makes the mapped read race-free (parsed only after
  `hasCompletedFramegen`). Slow EMAs (1/8) drive two auto-calibrations: the **FB tolerance**
  loosens (up to 2.5 texels) only on *ambiguity-without-error* — round trips fail while measured
  prediction error stays low (periodic textures: fences, grilles, tiling), where the kill would
  reintroduce fizzle; and the **agreement window** widens by the measured static-scene noise floor
  (film grain, dithering) so inherent temporal noise stops mass-killing the motion term. An
  explicit `GAMESCOPE_FRAMEGEN_FB_TOL` pins the tolerance (auto-cal keeps its hands off).

`GAMESCOPE_FRAMEGEN_ADAPT=0` disables all of it (B3-bit-exact warps, no probe). With
`--framegen-debug`, consume logs
`adapt resid=… bad=…% killed=…% noise=… fbP75=… -> fbTol=… agree=…/…`.

### 3.5 Learned field refinement (Stage C, opt-in via `GAMESCOPE_FRAMEGEN_NET=<blob>`)

A tiny convolutional net (12→16→16→4 channels, 3×3 kernels, ~4.6k parameters,
`cs_framegen_motion_net.comp`) refines the checked forward motion field at field resolution once per
real frame — zero cost per generated frame, no future-frame dependency, and the full-res warp is
untouched. Bidir additionally refines the checked reverse field. It predicts
**corrections, not pixels**: a flow residual bounded to ±2 field texels (`2·tanh`) and an additive
confidence recalibration, on top of the Stage-B field. Safety is structural, three layers deep:

1. a zero final layer IS Stage B (the trainer zero-initializes the head, so the failure floor is
   the current behavior, and `--init` exports a bit-neutral blob);
2. the corrections are bounded by construction (tanh flow, clamped confidence), so the worst a bad
   checkpoint can do is limited;
3. the B4 probe grades (and trust-scales) the **refined** field — the one the warps consume — so a
   mispredicting net is clamped in the same batch and shows up in the `adapt resid=…` log lines.

It never touches image content — HDR rules unaffected by construction. Inputs per texel (12
channels; `scripts/framegen-net-train.py` mirrors them exactly and the two must stay in lockstep):
both directions' flow/conf/round-trip error, destination luma, source luma warped along the field
(the photometric evidence), the B4-style magnitude-normalized residual, and a task-conditioning
flag (reserved). Refining the reverse field is the same problem with the lumas swapped, so **one
set of weights serves both dispatches** (binding symmetry, the fbcheck trick). Weights ride a
2048-wide `R32F` texture uploaded once via a mappable staging image. The forward consistency pass
already produces both checked fields, so learned forward prediction does not require `_BIDIR`.

**Training** is the same self-supervision B4 measures: warp the source luma along the refined
flow; each texel pays either the warp error (× refined confidence) or the fallback error
(× complement), plus a small confidence-independent flow term so rejected vectors remain
trainable. Confidence can rise only when the final corrected vector reconstructs the observed
pair. Capture with `GAMESCOPE_FRAMEGEN_RECORD`, train with
`scripts/framegen-net-train.py` (numpy-only, CPU, minutes), deploy the blob, and validate with the
live A/B the adapt log provides (net vs no-net `resid`/`bad%` on the same content).

**Kernel shape** (measured on NVIDIA Blackwell; re-benchmark on the present GPU before tuning):
one fused dispatch per direction, 8×8 output tiles from 14×14 feature tiles (3-texel receptive
apron), activations and per-layer weights staged in LDS, everything vec4-packed so each MAC step
is a `dot()` (the scalar version was 2.2× slower — LDS load-instruction bound, not ALU). Two
measured dead ends: 128 threads/tile (worse; not occupancy-bound) and a position×channel-group
work split (2.3× worse; quadruples activation loads). Costs 0.59/1.05/2.29 ms per real frame at
1080p/1440p/2160p on the RTX PRO 500 — the single most expensive per-real pass, which is why it is
opt-in; a `packHalf2x16` LDS variant and a `VK_KHR_cooperative_matrix` path are the known next
steps if a bigger net earns them.

**In-situ learning (C2, `GAMESCOPE_FRAMEGEN_NET_ONLINE=1`).** The net keeps training *on the
framegen GPU while it serves*, so the correction function tracks the current scene instead of
whatever an offline blob was fit to. Per real frame, riding the same batch command buffer: two
gradient dispatches (`cs_framegen_motion_net_train.comp`; each workgroup = one hash-placed 4×4
training tile, forward + full backprop in LDS, per-tile gradient written as one row of a slice
image — **no float atomics**, `VK_EXT_shader_atomic_float` isn't universal) and one optimizer
dispatch (`cs_framegen_motion_net_opt.comp`; one thread per parameter: slice-mean, Adam,
**decay toward the prior**, and an EMA publish). ~8M MACs per step — an order of magnitude below
the inference pass; `_NET_EVERY=N` strides it for weak GPUs. Three structural properties do the
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

---

## 4. The decision state machine

`framegen_record_real_frame` (`:5682`) resolves the whole decision once per real frame. It first
filters **overlay-only repaints** by base-layer pointer identity (`:5701-5708`) — a MangoHud tick or
fading notification re-composites the same game content and must not poison the pacing measurement —
then invalidates history on scene change (§4.5), then decides.

### 4.1 THE structural fact: two control regimes selected by `hasFramegenQueue()`

This is the single biggest newcomer trap. There are **two entirely different control systems**:

- **DEDICATED queue** — `bCanSpeculate` (`:5782`, = `hasFramegenQueue && bGpuHasHeadroom &&
  prev!=0`) **ignores** both the empty-vblank test *and* the leaky bucket, so the scheduler
  **generates a full batch after every usable real frame**, even when the game is hitting refresh
  and no vblank is empty. Misses are discarded by the supersede path (no latency cost). The
  **monotonic timestamp-driven degradation ladder** (§4.4) is the *only* proactive protection.
- **SHARED queue** (single-queue family, or `GAMESCOPE_FRAMEGEN_SINGLE_QUEUE`) — the **leaky-bucket
  admission gate** (§4.2) requires a proven empty vblank + 2 stable frames; `nGenerate` is
  **hard-capped to 1** (`:5915`) to bound head-of-line work in front of the next composite; and there
  is **no ladder at all** (no timestamps → `ulCurRungCostNs` stays 0 → the degrade guard at `:5871`
  never fires). Full quality is held and the **reactive supersede-discard is the sole safety net**.

### 4.2 Per-frame generate/dormant gate (`:5778-5836`)

Signals: `bLeavesEmptyVblank` (`prev!=0 && now-prev ≥ 1.5·interval`); `bGpuHasHeadroom =
hasCompletedFramegen(lastGeneratedSeqNo)` (the oversubscription guard → **at most one batch in
flight**; `hasCompletedFramegen(0)` is trivially true so the first frame is never blocked);
`bGeneratable = bLeavesEmptyVblank && bGpuHasHeadroom`; `bCanSpeculate` (above). Final gate
`bShouldGenerate = bCanSpeculate || bReactiveReady` (`:5823`). On not-generate → dormant early-return
(`:5836`), logging `busy` / `stabilizing` / `dormant`.

**Leaky bucket** (`:5805-5822`): `nStableFrames` saturates in `[0, k_uFramegenStableFramesRequired=4]`;
generatable → `+k_uFramegenStableFramesGain=2` (clamp 4), else → `−k_uFramegenStableFramesLeak=1`
(floor 0). `bReactiveReady = bGeneratable && nStableFrames ≥ 4` → needs **2 consecutive** generatable
frames from cold; leak(1) < gain(2) so a single jittered interval never disarms an established
stream. Reset to 0 on invalidation. **Fully bypassed on the dedicated queue.**

### 4.3 Multiplier x2/x3/x4 adaptation (`:5846-5916`)

`nMeasuredGapVblanks = round((now−prev)/interval)`, min 1. `nGapVblanks = bCanSpeculate ?
max(measured, max(2, eff.multiplier)) : measured` — the dedicated queue plans ≥ the multiplier's
slots even after a fast interval. `nGenerate = min(gap−1, max(1, eff.multiplier−1))` — **one fewer
than the whole-vblank gap** (the final slot *is* the next real frame). Shared-queue caps to 1.
Per slot (`submit_batch:5335-5340`): `phase = k/nGapVblanks`; `strength = clamp(phase ·
(g_flFramegenStrength / 0.5), 0, k_flFramegenMaxForwardStrength=1.5)`. `strength 0.5` reproduces the
classic x2 half-step.

### 4.4 Monotonic degradation ladder (#04, `:5849-5896`)

Evaluated **only on frames that actually generate** (after the dormant early-return), so an
idle/dormant stretch never moves the rung.

- `deadline = vblankInterval · k_uFramegenDeadlinePercent(85) / 100`.
- `ulCurRungCostNs = framegenRungCostNs(nDegradeSteps, nCurGenForLadder)` — the live-measured GPU
  cost of the current config's batch shape (per-(rung, gen-count), 7/8 EMA; §6).
- Guard: `nMaxDegradeSteps>0 && cost!=0 && steps<max` (`:5871`).
- If `nDegradeHold>0`: decrement only (cooldown — burns down even during a dip below deadline; it is
  **wall-frames, not over-budget-frames**).
- Else if `cost > deadline`: peek `nextEff = framegen_effective_config(steps+1)` and **step only if
  it truly reduces work** (`nextEff.mode != curEff.mode || nNextGen < nCurGen`, `:5888-5891`) — a
  pure multiplier notch that doesn't lower the generated count this gap would cost quality for zero
  GPU saving. On step: `nDegradeSteps++`, `nDegradeHold = k_uFramegenDegradeHoldFrames(4)`.

**Rung ordering** (`framegen_effective_config` `:4880`): applies `n` degradations to the ceiling
`{g_eFramegenMode, max(2, g_nFramegenMultiplier)}` — sheds **Motion → Extrapolate first** (one rung),
then steps the multiplier down toward x2. `framegen_max_degrade_steps() = (mode==Motion?1:0) +
max(0, mult−2)`. There is deliberately **no "stop generating" rung** — the ladder always keeps
generating (so its GPU-time input never starves); the genuine "even x2 overruns" case is left to the
reactive pacing gate. It **never mutates** `g_eFramegenMode`/`g_nFramegenMultiplier` (those gate the
feature and size the pool); it only reads an *effective* config. **Monotonic within a scene** — never
restores mid-scene (restoring means re-probing a richer config that may not fit, then dropping it → a
visible toggle, the exact micro-stutter this feature exists to avoid). Re-probed from full only on a
scene change.

### 4.5 History invalidation and priming

`vulkan_framegen_invalidate_history` (`:4907`) drops history and resets `nStableFrames=0`,
`nDegradeSteps=0`, the learned rung costs (`framegenResetRungCosts`), and the retained ring slots on:
focus change, layer-count change, output-EOTF change (SDR↔HDR without a format change), long
frame-gap (`k_ulFramegenMaxRealFrameGapNs`, a stall/load-screen), resolution/format change. **The
first real frame after any invalidation only PRIMES** (`:5794-5801`: sets `valid=true`, returns
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
    └──(!bGpuHasHeadroom: prior batch in flight)── BUSY ──────────────────┤   (motion→extrap→x3→x2)
                                                                          │   never UP until scene change
  Dedicated queue: bCanSpeculate skips the empty-vblank test AND the      │
  leaky bucket → GENERATES EVERY FRAME (misses discarded)                 ▼
                                                             refill_idle extends 1 slot on drain
```

---

## 5. Latency & correctness invariants (with enforcement points)

1. **Real frames never delayed** *(default path; bidir §3.3 is the deliberate exception)*. Generation
   is a *separate* command buffer submitted *after* composite; the real present waits only on the
   composite `sequence` (`:6208-6216`). On a dedicated queue a slow generation on queue 1 cannot
   head-of-line-block composite on the in-order queue 0. Push constants (≤64B, `static_assert :2155`)
   instead of the shared upload arena keep framegen from racing the composite host bump allocator across
   queues. **Under bidir** this still holds at the *generation* level (generation never blocks the
   composite), but the *presentation* timeline intentionally holds each real frame one interval so the
   interpolations before it can show first — a scheduling choice, not a generation stall.
2. **Generated frames never waited on.** All completion checks are **non-blocking**
   `hasCompletedFramegen` peeks (`GetSemaphoreCounterValue ≥ seq`, `:1908`): the present-decision peek
   (`generated_frame_ready`, `steamcompmgr.cpp:9166`), the oversubscription guard (`:5780`), and the
   ring-advance read-pin (`:6240`). `consume` drops the front slot if incomplete rather than stalling
   scanout. A not-ready slot becomes a free HW repeat.
3. **A real frame latching discards the ENTIRE pending batch.**
   `vulkan_framegen_discard_generated_frame` clears the whole `pending` **vector** (`:5023`); called
   on `hasRepaint` (`steamcompmgr.cpp:9156`, `superseded_by_real_frame`) and on
   `overlay_defer_budget` / `generation_too_slow` (`:9191`). Correctness of the split (decision in
   steamcompmgr, flip in backend) hinges on discard emptying `pending` before `paint_all` (`:9210`).
4. **Forward extrapolation only.** Per-slot `strength = phase · (g_flFramegenStrength / 0.5)` is a
   forward coefficient; idle refill pushes `phase>1` up to the 1.5 cap — never interpolation. Only the
   debug `blend` shader interpolates (hence debug-only).

**Supporting invariants:** only base-layer commits count as real (overlay filter `:5701-5708`);
framegen ⊥ VRR (`allowVRR = cv_adaptive_sync && !framegen`, `steamcompmgr.cpp:2570`; generated present
`allowVRR=false`, `DRMBackend.cpp:3619`) and ⊥ tearing (`bTearing && !framegen`, `:9052`) — generated
frames need fixed vblank slots; a slot is exactly one of real / generated / HW-repeat; overlay
deferral is bounded by `k_nFramegenMaxDeferredOverlay=4` with the overlay flag preserved for the next
real composite. **One batch in flight** (`bGpuHasHeadroom`) is the single assumption making three
lockless structures safe (§6).

---

## 6. Resource architecture

- **Zero-copy history.** `previousReal`/`currentReal` are `Rc<CVulkanTexture>` **aliasing composite
  ring slots**, not copies (`:4727`, shift `:5789-5791`) — saves a full-resolution copy per real
  frame (~66 MB/frame at 4K) on the weak card. This forces the ring to grow **3 → 5 slots**
  (`k_uOutputRingSize{Framegen,Default}`, `:3872`) so 2 history + scanout + next-target never collide,
  the **pinned-slot skip** in the ring advance (`:6240`), and structural suppression of the
  partial-overlay aliases while active (`:3915`).
- **Generated-frame scanout pool** `framegenOutputImages`, sized **`2·g_nFramegenMultiplier`**
  (`:5051`), rolling cursor `% pool.size()`. **Strictly disjoint** from the composite ring — generated
  frames flip directly and outlive generation.
- **Dedicated compute queue.** `createDevice` requests `queueCount=2` from `m_queueFamily` when
  `framegen && backend-supported && queueCount≥2 && !GAMESCOPE_FRAMEGEN_SINGLE_QUEUE` (`:709`),
  retrieves index 1 → `m_framegenQueue`. Both queues share `REALTIME` global priority (Vulkan has no
  intra-family priority — this is a **decoupling** win, not a priority one). Single-family topology
  still works. Requested only when framegen is enabled, so other sessions are unaffected.
- **Own framegen timeline** `m_framegenTimeline` (monotonic `m_framegenSeqNo`); graceful downgrade to
  the shared queue if creation fails. Composite `waitIdle` explicitly drains it.
- **Per-queue descriptor ring.** `m_framegenDescriptorSets` (16) vs composite `m_descriptorSets` (24)
  → a 40-set pool; routed by `CVulkanCmdBuffer::m_bFramegen` set by `markFramegen()`. **No completion
  tracking** — a bare modulo counter, safe *only* because of the one-batch-in-flight guard.
- **Command-buffer split.** Generation into its own `CVulkanCmdBuffer` from the shared pool, submitted
  via `submitFramegen` (raw `vk.QueueSubmit` to `m_framegenQueue`), recycled after the framegen
  timeline signals (`framegenGarbageCollect`).
- **Cross-queue WAR pin.** `genReadA/B/genReadSeqNo` (`:5417`) pin history slots against ring reuse
  until `hasCompletedFramegen(genReadSeqNo)` — at most 2 slots pinned, so the ≥5 ring always has a
  free slot and the skip loop terminates.
- **Timestamp query ring.** Depth-4, created only if `timestampValidBits>0 && timestampPeriod!=0` on
  the dedicated queue; TOP/BOTTOM_OF_PIPE brackets; readback **without** `WAIT_BIT` folds into
  `m_aFramegenRungCostNs[8][4]` via a 7/8 EMA keyed by **batch shape** (so x2-gaps don't inherit
  x4-batch timings).

**The one assumption behind three lockless structures:** "at most one framegen batch in flight"
(`bGpuHasHeadroom`) is what makes the 16-slot descriptor ring, the depth-4 timestamp ring, and the
`genReadA/B` pins all safe without completion tracking. Any change allowing two concurrent batches
breaks all three.

Central state: `FramegenHistory_t g_framegenHistory` (`:4725/4801`),
`FramegenMotionResources_t g_framegenMotion`, `FramegenDispatch_t g_framegenDispatch` — all touched
only from the compositor thread → no locking.

---

## 7. Vendor / perf model

The unifying truth: **bandwidth-bound, ALU-light.** LDS apron helps only on cache-poor GPUs; fp16
helps only on integer targets and its upside is capped because the apron storage stays fp32; the
no-LDS `direct` shader wins on large-L2 / Infinity-Cache parts where the neighbor cross is already a
cache hit.

- **Dispatch selection** `framegen_dispatch_for_format` (`:5167`) — single-slot **format-keyed
  cache**, computed once per drmFormat: `useFp16 = supportsShaderFloat16 && !isFloatFormat`;
  `useR16FLuma` picks the luma format + `MOTION_LUMA_PAIR` vs `_RGBA`; `motionSupported = ABGR16161616F
  sampled+storage`. **Vendor override** (`:5198-5204`): `vendorID==0x10DE` (NVIDIA) forces
  `useFp16=false` and `extrapolate=DIRECT` — but **deliberately leaves `extrapolatePair` on LDS** (no
  measured direct-pair shader yet).
- **Microbenchmark** `vulkan_framegen_benchmark` (`:5505`) — times the *real* production helpers over
  a 3-res × 2-format sweep {1080p/1440p/2160p} × {ABGR2101010 int, ABGR16161616F float}, `nIters=200`,
  own 2-query pool, `waitIdle`-serialized; marks the production-selected variant with `(*)`. Empirical
  basis for the NVIDIA=direct choice. (Note: it checks only `timestampPeriod==0`, **not**
  `timestampValidBits`, unlike the live path — weaker guard.)
- **Live measurement → ladder** (dedicated queue only): timestamp ring → non-blocking readback in
  `framegenGarbageCollect` → 7/8 EMA per (rung, gen-count). On the shared queue there are no
  timestamps, so the ladder is inert and the reactive discard is the only protection.

---

## 8. Observability

- **`g_bFramegenDebug`** (`main.cpp:348`) + **`FramegenDebugShouldLog`** (`main.cpp:502-508`, gate
  `<=1 || counter==1 || counter%every==0`, period `g_uFramegenDebugEvery` default 60) throttle 8
  rendervulkan call sites that emit the scheduler's live state: priming (`:5797`), overlay-only
  ignored (`:5705`), the `busy/stabilizing/dormant degrade=N/M` line (`:5830-5833`), the
  `generated N frame(s) … mode=… gpu=X.XXms` line (`:5426`), plus the real/generated/repeat slot
  classification in steamcompmgr (`:9204-9206`). **This is the only way to observe the state machine
  described above at runtime.**
- **`--debug-dual-gpu-route` / `g_bDebugDualGpuRoute`** (`main.cpp:346, 839`) is a **distinct**,
  per-frame, **un-throttled** channel for the cross-GPU import path — do not confuse it with the
  rate-limited framegen debug.

---

## 9. Roadmap (`doc/framegen-proposals/`)

| # | Proposal | Status |
|---|---|---|
| **01** | VRR hybrid — present the real frame VRR-style for the full latency win, schedule the generated frame with a timer-armed mid-interval atomic flip | **PROTOTYPE IMPLEMENTED** (`GAMESCOPE_FRAMEGEN_VRR_HYBRID=1`, dedicated queue + active VRR only). The VRR-suppression gates are gone: `steamcompmgr.cpp:2574` now keeps `allowVRR` on while framegen runs iff the hybrid is requested, and `DRMBackend.cpp:3625` inherits the real frame's `allowVRR` instead of hard-coding false — so both flips carry the same VRR state and no per-frame modeset fires. Single generated frame at phase 0.5 flipped by the absolute mid-interval timer `g_FramegenMidTimer` (`steamcompmgr.cpp:8337`); planned by `framegen_vrr_hybrid_submit` (`rendervulkan.cpp:5661`). Built on #06's frametime EMA; toggle is env, not the proposed `cv_framegen_vrr_hybrid`. **Not yet validated on a VRR panel** (see the proposal's Status line). |
| **02** | Base-layer generation + late overlay/cursor composite — generate on the pre-upscale game layer, composite UI on top (~−56% BW, no HUD ghosting) | **PROTOTYPE IMPLEMENTED** (`GAMESCOPE_FRAMEGEN_BASE=1`, no dedicated-queue requirement). Generation runs on `layers[0].tex`; the present-time consume late-composites overlays/cursor over the generated base through `vulkan_composite` (`framegen_base_present_composite`, `:5350`), so UI/HUD/cursor smear is eliminated **by construction** and generated frames get the full FSR + shaper/3D-LUT pipeline. **Divergence:** the draft's "−56% BW" inverted post zero-copy-history — base mode now *adds* a base-sized history copy per real frame + one full composite per presented generated frame; the justification is the quality win, not bandwidth. Live per-frame dispatcher (`framegen_base_layer_usable`, `:4939`) falls back to output-space generation for video-underlay/YCbCr/ReShade/no-storage-format scenes. Validated nested on the dual-GPU FSR path (motion x3). |
| **03** | dGPU optical-flow donor — offload motion estimation to the render GPU's `VK_NV_optical_flow` OFA, ship a small flow field over PCIe | **Aspirational** (longest horizon). Zero OFA symbols; needs a second Vulkan device gamescope has never had; cross-vendor timeline interop rated unreliable. Motion mode is the shipped fallback. |
| **04** | Timestamp-driven adaptive degradation | **IMPLEMENTED** (`a75bfbe`) but **divergent** from the doc — shipped is *monotonic* (down-only, re-probe on scene change), rungs are mode/multiplier notches (not a pyramid table), 85% deadline (not 0.6 budget), 2D per-(rung,gen-count) 7/8-EMA (not one global EWMA), fixed 4-frame cooldown, zero CLI flags, `VK_EXT_calibrated_timestamps` unused. (The proposals `README.md` now marks this Implemented.) |
| **05** | Tile classification + `vkCmdDispatchIndirect` + SDMA static fill — generate only over moving tiles, fill static tiles on the transfer engine | **Aspirational** (deferred). No transfer-only queue discovery / classify shader exists; the doc admits it depends on transfer-queue discovery that isn't there yet. |
| **06** | JIT phase — plan one slot per vblank against the KMS pageflip clock with a slew-limited frametime EMA, instead of baking phases from a single-interval gap guess; adds the "skip when keeping up" guard | **PROTOTYPE IMPLEMENTED** (`GAMESCOPE_FRAMEGEN_JIT=1`, dedicated queue only). `framegen_jit_submit` / `vulkan_framegen_jit_tick` (`rendervulkan.cpp`), EMA `FramegenHistory_t::ulFrametimeEmaNs`, keep-up guard `k_uFramegenJitKeepUpPercent=110`. Tested with GravityMark (~21% fewer generation passes for identical present coverage); phase-accuracy/smoothness benefit still needs native DRM + human A/B. |

---

## 10. Sharp edges / WIP / open bugs

- **Speculate-every-frame WIP (classic path only).** In the *classic* dedicated-queue regime
  `bCanSpeculate` (`:6072`) still ignores `bLeavesEmptyVblank`, so it generates a full batch **every
  real frame even when the game hits refresh** — a framegen pass of wasted bandwidth per vblank on the
  weak card, worst with Motion. The flagged "skip when comfortably keeping up" to-do (`1b949f0`) is now
  **implemented in the two opt-in prototypes**: #06 JIT (`k_uFramegenJitKeepUpPercent=110`, `:6236`)
  and #01 VRR hybrid (`k_uFramegenHybridKeepUpPercent=220`, `:6211`) skip the slot when the frametime
  EMA says the game is keeping up. The classic path (neither toggle set) still has no guard.
- **NVIDIA direct-pair not implemented.** The vendor override only swaps the single-slot extrapolate;
  `extrapolatePair` stays on LDS, so NVIDIA x3/x4 batches still pay the apron and lose the ~30–37% win.
- **Vendor predicate too narrow.** Single hardcoded `vendorID==0x10DE`. **The actual weak-card targets
  (7900 XT / 5700 XT) are not special-cased** and run the un-benchmarked LDS+fp16 default; the direct
  shader's own header claims RDNA3's Infinity Cache benefits too — dispatcher and shader comment
  disagree on RDNA3. Extend the predicate once measured on those parts.
- **Dead code.** `FramegenHistory_t::previousPresentTimeNs` (`:4765`, written `:5729/:5761`, **never
  read** — confirmed) and `previousFrameId` (`:4763`) are dead. The `MOTION_LUMA` / `_RGBA`
  single-target shaders are compiled but never dispatched (only the `_PAIR` variants run); the dead
  `_rgba` variant uses a **single-tap** downsample that would reintroduce the exact aliasing wobble
  the box filter fixed if it were ever revived.
- **Wobble** — fixed in `a75bfbe` (nested-refresh clock, box luma, sub-pixel SAD, selective clamp);
  extrapolate stays structurally juddery on fast motion → prefer motion mode for pans.
- **Translucency (vkmark "penguins")** — tested: FG *preserves* opacity (0.52 vs 0.50); the earlier
  "goes opaque" claim was disproven. Residual is edge judder/ghosting, not transparency loss; needs a
  human framegen-off A/B to finalize.
- **Config inconsistencies.** `--framegen-strength` uses `atof` (garbage → 0.0 silently, disabling the
  forward step with no error) vs the multiplier's validated `exit(1)`;
  `GAMESCOPE_FRAMEGEN_BENCHMARK` is presence-only, `GAMESCOPE_FRAMEGEN_SINGLE_QUEUE` / `_JIT` /
  `_VRR_HYBRID` / `_BASE` / `_BIDIR` need a truthy int, and the Stage-B knobs `_FB` / `_AGREE` / `_ADAPT` are default-on
  (`=0` to disable) with `_FB_TOL`/`_NET_LR` floats, `_NET`/`_RECORD`/`_NET_PROFILE` paths and `_RECORD_MAX`/`_NET_EVERY` uints. All eighteen `GAMESCOPE_FRAMEGEN_*` env vars are undocumented in `--help`.
- **Backend nuances.** The ring 3↔5 size is fixed at allocation — a mid-session framegen enable/disable
  requires a full `vulkan_remake_output_images` (`waitIdle` + reset), *not* a live adjustment. The
  Wayland generated-frame present hard-nulls planes 1..7 (overlays/cursor dropped for a generated
  slot). `force_repaint`'s load-bearing effect for the generated case is the `Nudge()` (the
  `bShouldPaint` override already handles the paint decision).
- **Shared-queue path is intentionally hobbled** — `nGenerate` forced to 1, no timestamps, so the
  ladder is inert and only the reactive discard protects it.
- **Motion is all-or-nothing under pressure** — the ladder sheds Motion before any multiplier notch,
  so `Motion-x4 → Extrapolate-x4 → x3 → x2` with no Motion-x3/x2 rung.
- **Descriptor ring has no completion tracking** — safety rests entirely on the one-batch-in-flight
  guard; the 16-slot size is margin, not a fix.
- **`nMaxDegradeSteps`** is threaded into `framegen_submit_batch` but used only for a debug log.
- **Idle-refill runaway** is bounded only by the 1.5 strength clamp + `nMaxSlots` cap + the 250 ms
  `idle_frame_gap` invalidation.
- **sRGB-view vs threshold comment** — history is bound `setTextureSrgb(true)` (hardware-linearizes
  UNORM sRGB on read), yet the shader comments describe the 0.08/0.40 suppression thresholds as
  "gamma-encoded [0,1]" — a comment/behavior mismatch worth auditing (self-cancels for pure-float
  targets).
