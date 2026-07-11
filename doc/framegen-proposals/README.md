# Frame-generation design proposals

This directory collects forward-looking design proposals for gamescope's
experimental compositor-side frame generation. Each specifies motivation, the
Vulkan mechanisms, concrete integration points in the tree, a latency/throughput
analysis, an adversarial risk table, and a testing plan. Most are **design
documents**; each proposal's `Status:` line says whether it has since been
implemented. Of the numbered proposals, #04 is implemented and #01, #02 and #06
have env-gated prototypes; the **quick reference** below lists every flag and
toggle and exactly what each one needs.

> **Just want to use it?** Start with the plain-language
> [**How-To guide**](../framegen-howto.md) — two-card setup, how to connect the
> display, copy-paste commands per mode, and current limitations. The tables
> below are the terse engineer's reference.

## Quick reference: enabling each feature

Frame generation is off until `--experimental-framegen`. Everything else layers
on top of it.

### Command-line flags

| Flag | Values (**default**) | Purpose |
|---|---|---|
| `--experimental-framegen` | — | **Master switch** — required by every feature below. |
| `--framegen-mode` | **`extrapolate`** · `motion` · `blend` | Algorithm. `motion` unlocks the whole motion-quality stack (FB / agreement / adaptation / bidir / learned net). `blend` is interpolation for debugging only. |
| `--framegen-multiplier` | **`2`** · `3` · `4` | Presented-to-real ratio (inserts up to *N*−1 generated frames per real frame); also sizes the output image pool. |
| `--framegen-strength` | `0.0`–`1.0` (**`0.5`**) | Forward extrapolation step size. |
| `--framegen-debug` | — | Per-frame logging (rate set by `GAMESCOPE_FRAMEGEN_DEBUG_EVERY`). |

### Motion-mode quality — on by default, need only `--framegen-mode motion`

| Variable | Default | Effect (set `=0` to disable) |
|---|---|---|
| `GAMESCOPE_FRAMEGEN_FB` | on | Forward-backward consistency check — kills mislock / disocclusion fizzle. |
| `GAMESCOPE_FRAMEGEN_AGREE` | on | Per-pixel two-source agreement — kills double-exposed / ghosted edges. |
| `GAMESCOPE_FRAMEGEN_ADAPT` | on | Self-supervised adaptation — per-frame field trust + CPU auto-calibration. |
| `GAMESCOPE_FRAMEGEN_FB_TOL` | `0.75` | FB round-trip tolerance (texels); **setting it pins the value** against `GAMESCOPE_FRAMEGEN_ADAPT`'s auto-calibration. |

### Opt-in modes — off by default

| Variable | Requires | Effect |
|---|---|---|
| `GAMESCOPE_FRAMEGEN_BIDIR=1` | `--framegen-mode motion`; **excludes** `GAMESCOPE_FRAMEGEN_JIT`, `GAMESCOPE_FRAMEGEN_VRR_HYBRID`, `GAMESCOPE_FRAMEGEN_BASE` | Bidirectional interpolation — smoothest motion, but real frames present **one interval late**. |
| `GAMESCOPE_FRAMEGEN_NET=<blob>` | `--framegen-mode motion` | Learned causal forward-field refiner; value is the path to a `GSFR` weights blob (see below). Bidir is optional. Empty / unreadable → disabled, not fatal. |
| `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` | `--framegen-mode motion`; `NET` blob optional | In-situ learning (C2): the forward refiner keeps training on the framegen GPU against every real frame, tracking the current scene. Without a `NET` blob it starts from a neutral prior; without `NET_PROFILE` the model is **ephemeral — nothing is written to disk**. |
| `GAMESCOPE_FRAMEGEN_NET_PROFILE=<path>` | `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` | Persistent per-game learning: loaded as the prior when the file exists (a malformed file is rejected loudly → neutral prior), checkpointed off-thread every 1024 trained steps and flushed at exit/reset, so short sessions persist too. Atomic writes (temp + rename): a crash or full disk never tears a good profile. |
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

### The learned refiner (Stage C) in three steps

1. **Capture** a representative scene (writes `GSFD` files):
   `GAMESCOPE_FRAMEGEN_BIDIR=1 GAMESCOPE_FRAMEGEN_RECORD=/tmp/fg gamescope --experimental-framegen --framegen-mode motion … `
2. **Train** (numpy only, CPU, minutes):
   `scripts/framegen-net-train.py --data /tmp/fg --out weights.bin`
   (`--init --out neutral.bin` writes an untrained blob that is bit-identical to Stage B.)
3. **Use** it: `GAMESCOPE_FRAMEGEN_NET=weights.bin GAMESCOPE_FRAMEGEN_BIDIR=1 gamescope --experimental-framegen --framegen-mode motion … `

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

## Context: how the shipped features work

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
  whenever the game keeps up with refresh.
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
- **Learned forward-field refinement** (opt-in,
  `GAMESCOPE_FRAMEGEN_NET=<weights>`, motion mode) — a ~4.6k-parameter
  fused-conv net refines the causal checked field once per real frame at field
  resolution (and the reverse field only for opt-in bidir): a tanh-bounded flow
  residual plus a confidence recalibration, learned from the residual error
  classes the hand-written consistency checks can't express (flow-boundary
  smoothing, occlusion inpainting, confidence-vs-usefulness calibration). A
  zero-initialized head is exactly the unrefined pipeline, the corrections
  are bounded by construction, and the adaptation probe grades the refined
  field — so a bad checkpoint is clamped in the same batch. Trained offline,
  self-supervised, on tensors captured with `GAMESCOPE_FRAMEGEN_RECORD`
  (`scripts/framegen-net-train.py`, numpy-only).

The proposals below build on top of that foundation.

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
