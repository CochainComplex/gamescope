# Frame-generation design proposals

This directory collects forward-looking design proposals for gamescope's
experimental compositor-side frame generation. They are **design documents**,
not implemented features — each specifies motivation, the Vulkan mechanisms,
concrete integration points in the tree, a latency/throughput analysis, an
adversarial risk table, and a testing plan.

## Context: what is already implemented

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
  back to extrapolation where the match is unconfident.

The proposals below build on top of that foundation.

## Proposals

1. [VRR hybrid: timed mid-interval generated flips](01-vrr-hybrid-timed-flips.md)
   — stop suppressing adaptive sync; present real frames immediately (full VRR
   latency win) and schedule the generated frame with a timer-armed mid-interval
   atomic flip, keeping the panel above its LFC floor.
2. [Base-layer generation with late overlay/cursor composite](02-base-layer-generation-late-overlay.md)
   — generate on the pre-upscale game layer, upscale the generated frame through
   the existing FSR pass, and composite overlays/cursor on top: roughly halves
   generation bandwidth and removes UI/HUD ghosting.
3. [dGPU optical-flow donor via `VK_NV_optical_flow`](03-dgpu-optical-flow-donor.md)
   — offload motion estimation to the NVIDIA render GPU's fixed-function optical
   flow accelerator, shipping a tiny flow field across PCIe to feed the warp
   pass. Opt-in and vendor-specific, with the on-GPU estimator as fallback.
4. [Timestamp-driven adaptive degradation ladder](04-timestamp-adaptive-degradation.md)
   — measure generation GPU time with timestamp queries and step quality/rate
   down *before* deadlines are missed (motion → extrapolate → lower multiplier →
   dormant), instead of only reacting after a late frame.
5. [Tile classification + indirect dispatch + SDMA static fill](05-tile-classification-indirect-sdma.md)
   — a cheap change-detection pass drives `vkCmdDispatchIndirect` so generation
   runs only over moving tiles, with static tiles filled on the transfer (SDMA)
   engine. Collapses generation cost on mostly-static content. (This is the
   deferred Tier-2 tile-classification item; it depends on transfer-queue
   discovery that does not exist yet, so it is documented rather than built.)
