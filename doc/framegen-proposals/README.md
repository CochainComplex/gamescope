# Frame generation — engineer's reference

The terse engineer's reference for gamescope's compositor-side frame generation.
Three things live here: the complete **flag / toggle reference**, **how the
shipped pipeline works** (and where each piece comes from in the literature), and
the forward-looking **design proposals** this directory grew out of. It's the
reference; the other framegen docs are the tutorial, the deep dive, and the
survey:

- 🕹️ [**How-To**](../framegen-howto.md) — plain-language setup: the two-card split, wiring the display, copy-paste commands per mode, current limits. **Start here to *use* it.**
- 🔧 [**Architecture**](../framegen-architecture.md) — the deep implementation map: end-to-end dataflow, the algorithmic branches, the decision state machine, resource design, and the §3.6 prior-art table.
- 🧰 [**Maintainer guide**](../framegen-maintenance.md) — invariants, ABI/cache/queue contracts, validation levels, and publication checks.
- 📚 [**Research survey**](../research-framegen.md) — the frames-only state of the art this pipeline is measured against (primary-source cross-checked).

**What's shipped vs. proposed.** Everything in the
[quick reference](#quick-reference-enabling-each-feature) below **exists in the
tree today** — the master switch, the motion-quality stack, bidirectional
interpolation, the learned refiner + in-situ training, and the #04 degradation
ladder all ship (the alternative placement modes — base-layer, VRR-hybrid, JIT —
as env-gated prototypes). The numbered [proposals](#proposals) are the roadmap and
carry their own authoritative `Status:` lines: **#04 implemented**; **#01 / #02 /
#06** env-gated prototypes; **#03 / #05** design-only; **#07** a
research-to-implementation map (E1/E2 plus bounded frames-only Gaps A/B/D are built). Each design doc
specifies motivation, the Vulkan mechanisms, concrete integration points, a
latency/throughput analysis, an adversarial risk table, and a testing plan.

## Quick reference: enabling each feature

Frame generation is off until `--experimental-framegen`. Everything else layers
on top of it.

### Command-line flags

| Flag | Values (**default**) | Purpose |
|---|---|---|
| `--experimental-framegen` | — | **Master switch** — required by every feature below. |
| `--framegen-mode` | **`extrapolate`** · `motion` · `blend` | Algorithm. `motion` unlocks the whole motion-quality stack (FB / agreement / adaptation / bidir / learned net). `blend` is interpolation for debugging only. |
| `--framegen-quality` | `low` · `medium` · **`high`** · `ultra` · `extreme` | Motion cost/quality ceiling. Low = forward match; medium = +FB/agreement; high = +adaptation and optional ML; ultra = +causal temporal acceleration; extreme = +color-guided reconstruction, bounded three-frame disocclusion recovery, and optional learned shading focus. Deadline degradation walks down these tiers. |
| `--framegen-multiplier` | **`2`** · `3` · `4` | Presented-to-real ratio (inserts up to *N*−1 generated frames per real frame); also sizes the output image pool. |
| `--framegen-strength` | `0.0`–`1.0` (**`0.5`**) | Forward extrapolation step size. |
| `--framegen-debug` | — | Per-frame logging (rate set by `GAMESCOPE_FRAMEGEN_DEBUG_EVERY`). |

### Motion-mode quality — on by default, need only `--framegen-mode motion`

| Variable | Default | Effect (set `=0` to disable) |
|---|---|---|
| `GAMESCOPE_FRAMEGEN_FB` | on | Forward-backward consistency check — kills mislock / disocclusion fizzle. |
| `GAMESCOPE_FRAMEGEN_AGREE` | on | Per-pixel two-source agreement — kills double-exposed / ghosted edges. |
| `GAMESCOPE_FRAMEGEN_ADAPT` | on | Self-supervised adaptation — per-frame field trust + CPU auto-calibration. |
| `GAMESCOPE_FRAMEGEN_RESERVOIR` | on in causal `extreme` | Three-real-frame screen-space disocclusion resolver; `=0` disables it for A/B attribution. Never scheduled below Extreme or in bidir. |
| `GAMESCOPE_FRAMEGEN_SHADING` | on in causal `extreme` with ML | Fourth-head causal shading-persistence supervision plus bounded color-trend correction; `=0` disables only these for an otherwise-identical net/queue A/B. |
| `GAMESCOPE_FRAMEGEN_FB_TOL` | `0.75` | FB round-trip tolerance (texels); **setting it pins the value** against `GAMESCOPE_FRAMEGEN_ADAPT`'s auto-calibration. |

### Opt-in modes — off by default

| Variable | Requires | Effect |
|---|---|---|
| `GAMESCOPE_FRAMEGEN_BIDIR=1` | `--framegen-mode motion`; **excludes** `GAMESCOPE_FRAMEGEN_JIT`, `GAMESCOPE_FRAMEGEN_VRR_HYBRID`, `GAMESCOPE_FRAMEGEN_BASE` | Bidirectional interpolation — smoothest motion, but real frames present **one interval late**. |
| `GAMESCOPE_FRAMEGEN_NET=<blob>` | `--framegen-mode motion --framegen-quality high|ultra|extreme` | Learned causal flow/confidence refiner; value is a `GSFR` weights blob. In bidir it defaults to confidence-veto-only: checked flow is preserved and confidence can only decrease. Empty / unreadable → disabled, not fatal. |
| `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` | motion `high`/`ultra`/`extreme`; `NET` blob optional | In-situ learning (C2): causal mode trains flow+confidence; bidir trains only the conservative confidence output row and freezes geometry/trunk. Without a `NET` blob it starts from a neutral prior; without `NET_PROFILE` the model is **ephemeral — nothing is written to disk**. |
| `GAMESCOPE_FRAMEGEN_NET_PROFILE=<path>` | `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` | Persistent per-game learning: loaded as the prior when the file exists (a malformed file is rejected loudly → neutral prior), checkpointed on an owned worker every 1024 trained steps and joined before the exit/reset flush, so short sessions persist without a detached-writer race. Atomic writes (temp + rename): a crash or full disk never tears a good profile. |
| `GAMESCOPE_FRAMEGEN_NET_LR` / `GAMESCOPE_FRAMEGEN_NET_EVERY` | `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` | Learning rate (default `3e-4`) / train every *N*th real frame (default `1` — raise on weak present GPUs). |
| `GAMESCOPE_FRAMEGEN_RECORD=<dir>` | `--framegen-mode motion` | Capture training tensors (one `GSFD` file per real frame, ≈1.2 MB at 1440p) into `<dir>`; bidir is not required. |
| `GAMESCOPE_FRAMEGEN_RECORD_MAX` | `GAMESCOPE_FRAMEGEN_RECORD` set | Cap on captured frames (default `1000`). |
| `GAMESCOPE_FRAMEGEN_JIT=1` | **dedicated framegen queue**; excludes `GAMESCOPE_FRAMEGEN_BIDIR` | #06 JIT display-clock pacing (a no-op without the dedicated queue). |
| `GAMESCOPE_FRAMEGEN_VRR_HYBRID=1` | **dedicated queue + connector actually in VRR**; excludes `GAMESCOPE_FRAMEGEN_BIDIR` | #01 VRR hybrid — real frames present VRR-style, the generated frame flips mid-interval on a timer (falls back to fixed-refresh **live** when VRR isn't active). |
| `GAMESCOPE_FRAMEGEN_BASE=1` | any mode; per-frame scene check; excludes `GAMESCOPE_FRAMEGEN_BIDIR` | #02 base-layer generation — generate pre-upscale, late-composite fresh overlays/cursor (falls back to output-space **per frame** on unsupported scenes). |

`GAMESCOPE_FRAMEGEN_BIDIR`, `GAMESCOPE_FRAMEGEN_JIT`, `GAMESCOPE_FRAMEGEN_VRR_HYBRID`
and `GAMESCOPE_FRAMEGEN_BASE` are alternative pacing/placement strategies —
`GAMESCOPE_FRAMEGEN_BIDIR` is mutually exclusive with the other three (it is
silently ignored if any of them is set); combining the remaining three is
untested, so enable one at a time.

### Development / debugging

| Variable | Effect |
|---|---|
| `GAMESCOPE_FRAMEGEN_DEBUG_EVERY` | Log every *N*th framegen event (default `60`; needs `--framegen-debug`). |
| `GAMESCOPE_FRAMEGEN_SINGLE_QUEUE=1` | Force the shared-queue regime (disables the dedicated-queue features `GAMESCOPE_FRAMEGEN_JIT` / `GAMESCOPE_FRAMEGEN_VRR_HYBRID`). |
| `GAMESCOPE_FRAMEGEN_BENCHMARK` | Run the shader microbenchmark, then exit before output creation (**presence-only** — even `=0` triggers it). |
| `GAMESCOPE_FRAMEGEN_NET_BIDIR_FLOW=1` | Restore experimental endpoint-trained flow correction/confidence raises in bidir for A/B only. Default off because it produced heavy intermediate-frame artifacts in live x4 testing. |
| `GAMESCOPE_FRAMEGEN_BIDIR_PHASE_BIAS=0…1` | Experimental low-latency bidir cadence A/B. Blends generated phases from the sharp/snappy `k/gap` baseline toward uniform multiplier spacing without changing flip timing. Default `0`; full display-grid pacing was rejected as blurrier, more edge-torn, and less responsive. |
| `GAMESCOPE_FRAMEGEN_BIDIR_OCCLUSION=0…1` | Experimental one-sided occlusion authority. When one checked direction is strong and the other is clearly rejected, smoothly retains more of the surviving warped side instead of phase-diluting it into the unwarped crossfade. Both-valid/both-killed pixels and queue timing are unchanged; default `0`. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR=<dir>` | E2 held-out full-colour validation. Requires motion+bidir, the dedicated queue, and base-layer mode off. Presents real A/B/C normally, hides B from estimation, and records paired invisible B candidates at occlusion strengths 0/0.5/1 beside exact B. Generated candidates never scan out. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR_MAX` / `_SKIP` | E2 sample cap (default `8`) / real-frame warm-up skip (default `0`). A 1440p XB30 v2 sample is about 62 MiB; synchronous file writes perturb capture cadence, so this mode is not a pacing test. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR_SPAN` / `_OFFSET` | E2 endpoint spacing (default `2`, clamped `2..16`) and hidden-reference position (default `1`, must be `< span`). The measured timestamp ratio is authoritative because uneven cadence can move it away from `OFFSET/SPAN`. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR_PHASE_TOLERANCE` | Optional E2 acceptance window around `OFFSET/SPAN` (default `1`, effectively disabled). Use `SPAN=6 OFFSET=1 PHASE_TOLERANCE=0.05` to retain only measured phases within `1/6 +/- 0.05`. |

### The learned refiner (Stage C) in four steps

1. **Capture** a representative scene (writes `GSFD` files):
   `GAMESCOPE_FRAMEGEN_BIDIR=1 GAMESCOPE_FRAMEGEN_RECORD=/tmp/fg gamescope --experimental-framegen --framegen-mode motion … `
2. **Train** (numpy only, CPU, minutes):
   `scripts/framegen-net-train.py --data /tmp/fg --out weights.bin`
   (`--init --out neutral.bin` writes an untrained blob that is bit-identical to Stage B.)
3. **Evaluate** before shipping the blob (numpy only, CPU):
   `scripts/framegen-net-eval.py --data /tmp/fg --net weights.bin`
   reports SSIM / edge-structure / `bad%` / temporal-stability for the neutral
   (Stage B) vs refined field and the deltas — the structural/temporal view the
   scalar training residual misses (proposal #07, Gap E1). Grades the *field*, at
   field-resolution luma; colour-domain LPIPS/FvVDP need the E2 capture extension.
4. **Use** it: `GAMESCOPE_FRAMEGEN_NET=weights.bin gamescope --experimental-framegen --framegen-mode motion … `
   Add `_BIDIR=1` for confidence-veto-only interpolation; the blob's learned flow
   remains causal unless the explicit `_NET_BIDIR_FLOW=1` debug A/B is set.

Or skip the offline steps entirely: `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` trains
in-situ on the framegen GPU while serving — by itself this is the fully
generic, zero-footprint mode (neutral prior, model lives and dies with the
session, no file is ever written). Add
`GAMESCOPE_FRAMEGEN_NET_PROFILE=~/.cache/fg-<game>.bin` to persist what it
learns per game. The recommended production shape combines both: an offline
blob as the safe prior, online learning tracking the scene on top. Measured
on the vkmark harness (B4 probe metrics; resid/bad lower is better, killed
lower means more of the field gets real motion vectors): no net → resid
0.0071 / bad 4.85% / killed 80.1%; online from a neutral prior → 0.0062 /
2.88% / 70.1%; offline blob + online tracking → 0.0064 / 3.85% / 77.5%.
Confidence *raises* are photometrically evidence-gated per texel (the net can
only un-kill vectors that predict the current frame pair) — without that
gate, mid-training miscalibration showed up on screen as small squares of
stale content.

GSFR v3 uses the formerly reserved fourth output as an Extreme-only
shading-persistence focus. In-situ training checks a reprojected third causal
frame before rewarding the head, and updates only its final row/bias; the
shared motion trunk cannot regress from this loss. The warp, not the net,
forms a phase-scaled RGB trend and caps it to 8% of local magnitude per real
interval. Offline v3 training currently leaves this head neutral; v1/v2 blobs
are accepted only after the undefined row is forcibly zeroed.

### Held-out full-colour validation (E2)

Create the output directory, run a motion+bidir scene with the dedicated queue,
then evaluate the resulting paired candidates:

```sh
mkdir -p /tmp/fg-color
GAMESCOPE_FRAMEGEN_BIDIR=1 \
GAMESCOPE_FRAMEGEN_RECORD_COLOR=/tmp/fg-color \
GAMESCOPE_FRAMEGEN_RECORD_COLOR_MAX=8 \
GAMESCOPE_FRAMEGEN_RECORD_COLOR_SKIP=80 \
gamescope --experimental-framegen --framegen-mode motion …
scripts/framegen-color-eval.py --data /tmp/fg-color --per-frame
```

The estimator sees A/C, never B. One shared motion field generates occlusion
strengths `0`, `0.5`, and `1` at B's measured phase, so every comparison uses
identical source frames, timing, model state, and exact full-resolution B. The
numpy-only evaluator reports normalized colour MAE, bad pixels, PSNR, luma SSIM,
edge error, paired wins, and screen-space temporal residual change; `--lpips`
adds LPIPS when torch/lpips are installed. Core metrics operate on captured code
values; EOTF is reported, not guessed into a display transform. GSCF writes are synchronous after GPU
completion and can make the displayed real-only cadence chunky. Judge live
smoothness only in a separate run without `_RECORD_COLOR`.

The default `SPAN=2 OFFSET=1` captures the measured phase of consecutive A/B/C
frames. Low-source-rate x4 also uses early phases such as `1/6`; target those
without changing the production shader by using
`GAMESCOPE_FRAMEGEN_RECORD_COLOR_SPAN=6`,
`GAMESCOPE_FRAMEGEN_RECORD_COLOR_OFFSET=1`, and
`GAMESCOPE_FRAMEGEN_RECORD_COLOR_PHASE_TOLERANCE=0.05`. Frame spacing alone is
not a timing guarantee: off-window intervals are rejected, and accepted samples
are always generated and graded at their recorded timestamp phase.

## How the shipped pipeline works

Frame generation itself (enabled with `--experimental-framegen`) is in the tree,
along with these hardening/quality passes:

- **Zero-copy history** — the last two composited output images are retained by
  reference instead of copied (the output ring grows to 5 while framegen is on).
- **fp16 + LDS-tiled extrapolate shader** — the forward-extrapolation shader
  runs in packed fp16 on capable hardware and caches the current-frame tile in
  shared memory (falls back to fp32 for scRGB/float targets).
- **Dynamic per-slot temporal placement + x2/x3/x4** — the extrapolation
  coefficient is derived from the measured real-frame gap, and up to
  `multiplier-1` frames are inserted per interval via a pending-frame FIFO.
- **Dedicated framegen queue** — generation runs on a second same-family compute
  queue with its own timeline, so it can never block the next composite on the
  realtime queue (falls back to the shared queue when unavailable).
- **Ownership-safe generated pools** — output selection observes both Vulkan
  command-buffer refs and backend framebuffer ownership. KMS/Wayland retention
  shortens or skips a batch instead of letting the compute queue rewrite an
  image still queued for scanout.
- **Motion-compensated mode** (`--framegen-mode motion`) — a low-res luma block
  matcher produces a motion field that a warp pass reprojects along, blending
  back to extrapolation where the match is unconfident. The matcher is a
  three-level coarse-to-fine pyramid: the full search runs only at the
  coarsest level (≈ ±128 full-res px of range, 4× the SAD context per tap),
  finer levels re-localize with a ±1 seeded search plus a zero-motion static
  anchor, and the finest level adds sub-texel parabolic refinement. Two
  consistency passes guard the result: a **forward-backward check** (a second,
  reverse-anchored estimation; vectors whose round trip does not close — the
  mislock/disocclusion fizzle class — lose their confidence, with the kill
  mask feathered to avoid tile-shaped fallback boundaries) and a **per-pixel
  two-source agreement test** in the warp (the flow is projected from both
  real frames; pixels where the projections read different content fall back
  individually, so edge errors degrade cleanly instead of double-exposing).
  Ultra/Extreme temporal acceleration normalizes retained displacement fields
  by their measured real-frame intervals, so source-cadence jitter is not
  interpreted as physical acceleration.
  *Prior art:* the block-matching + SAD + luma-pyramid construction is the
  classical, non-learned analog of AMD FSR 3's FidelityFX Optical Flow (GPUOpen);
  the two-source color-match arbitration mirrors FSR 3's optical-flow-vs-MV blend
  (see [research](../research-framegen.md) §4).
- **Bidirectional interpolation** (`GAMESCOPE_FRAMEGEN_BIDIR=1`, opt-in,
  motion mode) — generated frames sit *between* the two real frames: both are
  warped toward the slot's phase (the current frame along the forward field,
  the previous along the already-computed checked reverse field) and blended
  by confidence, so occlusions take content from the frame that has it and
  unmatched regions get a phase-correct crossfade instead of a static hold —
  removing extrapolation's hold-then-jump fallback judder and handling
  translucent content correctly. The real frame rides the pending queue
  behind its interpolations and presents one interval late (the intrinsic
  interpolation latency; hence opt-in), degrading to zero added latency
  whenever the game keeps up with refresh. *Prior art:* the video-frame-
  interpolation regime — bidirectional warp-and-blend with occlusion side-
  selection, as in RIFE (Huang et al., ECCV 2022) and FILM (Reda et al., ECCV
  2022); the one-interval latency is VFI's intrinsic cost (research §0–1).
- **Self-supervised online adaptation** (default on in motion mode,
  `GAMESCOPE_FRAMEGEN_ADAPT=0` disables) — every real frame is ground truth
  for the motion field just estimated from it: a microseconds-scale probe
  measures the field's actual prediction error against the frame, folds a
  global trust factor into the field's confidence in the same batch (scenes
  where vectors pass every consistency check yet mispredict — flashes,
  particle chaos — fade smoothly to the safe fallback and recover just as
  smoothly), and auto-calibrates the consistency tolerance and agreement
  window to the content on the CPU one batch later (loosening the round-trip
  kill only where ambiguity is measurably harmless, widening the agreement
  window by the measured temporal-noise floor).
- **Content scene-cut rejection** (part of adaptation at high and above) — four-bin luminance
  histograms over nine screen regions distinguish a hard content cut from coherent motion, while
  prediction residual and failed-field coverage provide independent confirmation. The verdict is
  finalized once on the GPU in the same batch; causal prediction duplicates the newest endpoint
  and bidirectional prediction chooses the nearest real endpoint, so unrelated scenes are never
  warped or dissolved together. The 1M-asteroid/camera-motion stress test produced no false cuts.
- **Learned forward-field refinement** (opt-in,
  `GAMESCOPE_FRAMEGEN_NET=<weights>`, motion mode) — a ~4.6k-parameter
  fused-conv net refines the causal checked field once per real frame at field
  resolution: a tanh-bounded flow residual plus a confidence recalibration,
  learned from the residual error
  classes the hand-written consistency checks can't express (flow-boundary
  smoothing, occlusion inpainting, confidence-vs-usefulness calibration). A
  zero-initialized head is exactly the unrefined pipeline, the corrections
  are bounded by construction, and the adaptation probe grades the refined
  field — so a bad checkpoint is clamped in the same batch. JIT/idle refills
  reuse that finalized field and weight snapshot; they do not rerun inference,
  adaptation or training on an identical pair. Trained offline,
  self-supervised, on tensors captured with `GAMESCOPE_FRAMEGEN_RECORD`
  (`scripts/framegen-net-train.py`, numpy-only). *Prior art:* the "heuristic
  motion + lightweight correction net" template of GFFE (Wu et al., ACM TOG
  2024, arXiv 2406.18551), though GFFE's correction net refines shading/color
  while ours refines the motion field (research §3).
  **Bidir uses a stricter contract:** it runs both directions but preserves the
  FB-checked vectors, forbids confidence raises, and updates only the final
  confidence row online. Endpoint photometric supervision cannot distinguish
  repeated-texture/aperture matches that reconstruct both real frames yet take
  a wrong path through intermediate time. Full learned bidir flow remains an
  explicit debug A/B, not a quality tier.

The techniques above have recognizable ancestors in the frame-generation
literature (block-matching optical flow, video frame interpolation, quadratic-
motion extrapolation, lightweight correction nets); each is cited inline, and
the full mapping — venues + DOIs/arXiv, primary-source cross-checked — is the
prior-art table in [`framegen-architecture.md`](../framegen-architecture.md) §3.6,
drawn from the [research survey](../research-framegen.md). The proposals below
build on top of that foundation.

## Proposals

1. [VRR hybrid: timed mid-interval generated flips](01-vrr-hybrid-timed-flips.md)
   — stop suppressing adaptive sync; present real frames immediately (full VRR
   latency win) and schedule the generated frame with a timer-armed mid-interval
   atomic flip, keeping the panel above its LFC floor.
   **Prototype implemented** (`GAMESCOPE_FRAMEGEN_VRR_HYBRID=1`, dedicated
   queue + active VRR only; built on #06's estimator — see the doc's
   implementation notes for divergences).
2. [Base-layer generation with late overlay/cursor composite](02-base-layer-generation-late-overlay.md)
   — generate on the pre-upscale game layer, upscale the generated frame through
   the existing FSR pass, and composite overlays/cursor on top: roughly halves
   generation bandwidth and removes UI/HUD ghosting.
   **Prototype implemented** (`GAMESCOPE_FRAMEGEN_BASE=1`; late composite uses
   the live layer stack — fresh overlays and latest cursor on every generated
   frame — with live fallback to output-space generation for unsupported
   scenes; see the doc's implementation notes for divergences).
3. [dGPU optical-flow donor via `VK_NV_optical_flow`](03-dgpu-optical-flow-donor.md)
   — offload motion estimation to the NVIDIA render GPU's fixed-function optical
   flow accelerator, shipping a tiny flow field across PCIe to feed the warp
   pass. Opt-in and vendor-specific, with the on-GPU estimator as fallback.
4. [Timestamp-driven adaptive degradation ladder](04-timestamp-adaptive-degradation.md)
   — measure generation GPU time with timestamp queries and step quality/rate
   down *before* deadlines are missed (motion → extrapolate → lower multiplier →
   dormant), instead of only reacting after a late frame. **Implemented** (as a
   monotonic degrade-once-and-hold ladder; see the doc for divergences).
5. [Tile classification + indirect dispatch + SDMA static fill](05-tile-classification-indirect-sdma.md)
   — a cheap change-detection pass drives `vkCmdDispatchIndirect` so generation
   runs only over moving tiles, with static tiles filled on the transfer (SDMA)
   engine. Collapses generation cost on mostly-static content. (This is the
   deferred Tier-2 tile-classification item; it depends on transfer-queue
   discovery that does not exist yet, so it is documented rather than built.)
6. [JIT phase: display-clock pacing for generated frames](06-jit-phase-display-clock.md)
   — stop baking slot phases from a single-interval gap guess; plan one slot at a
   time, one vblank ahead, with its phase measured against the KMS pageflip clock
   and a slew-limited frametime EMA. Fixes the phase-vs-vblank sawtooth in the
   marginal 40–59 fps regime and adds the missing "skip when keeping up" guard.
   **Prototype implemented** (`GAMESCOPE_FRAMEGEN_JIT=1`, dedicated queue only).
7. [Frames-only SOTA alignment: what we have, what's missing](07-frames-only-sota-alignment.md)
   — maps the [frame-generation research survey](../research-framegen.md) onto
   the shipped pipeline and proposals #01–#06: which SOTA ideas are already in
   the tree (extrapolation-first, quadratic acceleration, per-pixel color-match
   arbitration, UI post-composite), and the genuine frames-only gaps worth
   building — a perceptual/temporal validation harness, a frames-only
   disocclusion reservoir, and an optional color-domain shading-correction net
   head. **Design map / gap analysis**; Gap E1's structural/temporal evaluator
   (`scripts/framegen-net-eval.py`), Gap B's GPU content scene-cut guard, Gap
   A's bounded Extreme resolver, and Gap D's bounded causal shading-focus form
   are implemented. E2 full-color perceptual capture remains open.
