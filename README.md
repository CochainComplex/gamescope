# gamescope-gameslop

*A compositor-side **frame-generation** fork of [gamescope](https://github.com/ValveSoftware/gamescope). The name is tongue-in-cheek: yes, most of the pixels you see are machine-invented in-between frames — "slop" — but they're honestly labelled (the game never sees them) and produced by a real motion-estimation pipeline that **predicts *or* interpolates**, not a dumb frame average.*

### What it is

A **poor man's DLSS 4.5 / FSR 4.1 frame-generation surrogate** that runs on *any* Vulkan GPU — no RTX 50, no RDNA 4, no tensor cores, no vendor optical-flow or AI block, no driver lock-in. Those premium stacks generate frames *inside* the game from engine motion vectors and only on the newest silicon (DLSS 4.5's 6× multi-frame gen is RTX-50-exclusive; the ML-based FSR 4.1 stack is built for RDNA 4, only now trickling down to older Radeons). Gameslop chases the same outcome — more frames in the gaps — one layer down in the compositor, from the *finished frames alone*, tied to no vendor, no engine, and no per-game integration. It's a stopgap for the ongoing GPU/VRAM price crunch: it wrings smooth high-fps *motion* out of hardware you already own. It doesn't lower latency and it can't show detail the game never rendered — it buys smoothness, nothing else.

### The one principle to understand first: this is a **two-GPU** technique

Generating a frame costs real GPU work. If you do that on the **same** card that's rendering the game, you are stealing performance *from the game* to fake frames on top of it — usually a wash or a net loss (you'd get more by just rendering real frames, or by using the game's own FSR/DLSS). So doing this on one GPU is, in general, **not worth it**.

Gameslop's whole reason to exist is to put the generation on a **second, cheaper / older / otherwise-idle GPU**:

- your **strong** card renders the game (nothing taken away from it);
- a **weak "present" card — ⚠️ this must be the GPU your display is physically connected to** — generates the in-between frames *and* drives the display;
- the two are bridged over dma-buf.

The generation is essentially **free** because it runs on silicon that would otherwise sit idle. Normally that present card does almost nothing — it just composites and flips *finished* frames straight to the screen, which barely touches it. Gameslop spends that leftover budget doing **more than passthrough**: instead of forwarding a frame and waiting for the next real one, the present card uses its idle time to estimate motion and **synthesise extra frames** in the gap, while its consistency checks try to keep the artifacts down. That second card can be a laptop iGPU (e.g. an AMD 890M), an old GPU you never threw out (a 5700 XT), or the second card in a desktop — cheap hardware doing the work a new flagship would charge you a fortune for.

**When a single GPU *is* worth it (the exception):** only when you have GPU headroom to spare and no better option — typically an **older game that can't use FSR/DLSS frame generation** and doesn't fully load your (good) card. Then spending that idle headroom on generated frames is a real win. For a modern game that already pegs your GPU, single-GPU framegen just robs Peter to pay Paul — use the game's built-in frame generation instead.

**Where the trick lives:** frame generation happens in the **compositor**, downstream of the game — it never reaches *back* into the game's rendering. The generated frames are gamescope's own; the game, Wine/DXVK, Reflex and anti-lag never see them, and nothing is pulled *off* the render card, so the game's real frame rate and input latency are left completely untouched. The second card only ever **polishes what comes out** — a post-processor, not a middleman in the render path. Plug your monitor into the weak "present" GPU, render the game on the strong one, and let the two bridge over dma-buf.

### It's not a *dumb* blend — extrapolate, or truly interpolate

"In-between frames" makes people picture crude frame-averaging. Gameslop can do **genuine interpolation** — its opt-in **bidirectional** mode warps *both* neighbouring real frames to the in-between phase and blends them by confidence. The catch is latency: interpolation holds the newer real frame back, so this mode always runs **~1 frame behind**. The primary path instead **predicts forward** from the last two real frames, adds zero algorithmic latency, and can apply the learned refiner directly to that causal forward field. A plain frame-blend (`blend`) exists only as a debug aid.

**And it works blind — on purpose.** FSR and DLSS Frame Generation are handed the scene's *true* motion by the game engine — per-pixel **motion vectors**, depth, and — on DLSS — a dedicated **flow accelerator** (fixed-function hardware on DLSS 3, a learned network on DLSS 4). Gameslop takes **none** of that, and that is a deliberate design choice, not just a missing feature. It *could* try to pull motion vectors, depth or the framebuffer off the render card — but ferrying that data across to the second GPU every frame would **bottleneck the whole system** and add latency to the very render path this design exists to leave alone. So the rule is **non-intervention**: never touch the game's rendering, its GPU, or its input latency. The second card sits **downstream as a pure post-processor** — it sees only the **finished frames** and works out the motion itself from the pixels alone. That's a harder, more artifact-prone problem, which is exactly why most of the pipeline below is not "generate a frame" but "estimate the motion, then decide how much of it to trust":

Under the hood it's a staged, self-correcting motion pipeline:

- **Motion estimation** — a 3-level coarse-to-fine luma **pyramid block matcher** (full search only at the coarsest level, ≈±128 px reach), **sub-pixel parabolic** refinement, seeded ±1 search down the levels, and vector-median seeding to kill tear-like mislocks.
- **Artifact control** — a **forward-backward consistency check** (round-trip a vector; if it doesn't close, drop its confidence — kills disocclusion/mislock fizzle), a **per-pixel two-source agreement test** that stops ghosted double-exposures at edges, TAA-style neighbourhood clamping, and an Extreme-tier three-real-frame reservoir that validates adjacent background motion before filling a newly revealed boundary.
- **Extrapolate *or* interpolate** — default **forward extrapolation** (zero added latency), or true **bidirectional interpolation** (warp *both* real frames to the in-between phase, confidence-blend, phase-correct crossfade in the gaps) for the smoothest motion at the cost of ~1 frame of latency.
- **It learns** — a small (~4.6k-param) **convolutional refiner net** cleans up the motion fields (bounded flow residual + confidence recalibration), trainable offline from captured frames *or* **learning in-situ on the GPU while you play**, with per-game persistence. In Extreme, its zero-neutral fourth head learns whether aligned lighting/color trends persist across three causal real frames; the warp applies only a tightly bounded analytic correction. On top of that a **self-supervised loop** grades every real frame against the prediction that targeted it and auto-tunes its own thresholds.
- **Pacing & display** — **display-clock (KMS pageflip) JIT pacing** places each generated frame against the real vblank cadence; a **VRR/adaptive-sync-compatible** hybrid; and a **base-layer** path that composites HUD/cursor *after* generation so UI text stays crisp.
- **Engineering** — generation runs on a **dedicated async-compute queue** so it can never stall the real frame; **zero-copy** history; **fp16** + vendor-aware shader dispatch; and a **deadline-driven degradation ladder** that measures its own GPU time and sheds quality *before* it misses a vblank.

### How it maps to the state of the art

Working from finished frames only is the *hard* frontier of frame generation, and the design deliberately tracks the published frames-only research rather than reinventing it. Each piece has a named ancestor — an **analog**, since we have no engine motion vectors, depth, or G-buffers to lean on:

- the pyramid block-matcher is the classical construction behind **AMD FSR 3**'s FidelityFX Optical Flow;
- the causal-acceleration tier follows **Mob-FGSR**'s quadratic (uniform-acceleration) motion model;
- the heuristic-motion-plus-lightweight-correction-net shape is the **GFFE** (G-buffer-Free Frame Extrapolation) template;
- forward-vs-bidirectional is the extrapolation/interpolation split the literature draws, with the same latency trade-off.

The full mapping — what's already here, what the engine-integrated methods (DLSS, FSR, Mob-FGSR) do that a frames-only compositor *can't*, and the concrete gaps worth closing — is written against a primary-source-verified survey in **[the research doc](doc/research-framegen.md)** and **[SOTA-alignment proposal #07](doc/framegen-proposals/07-frames-only-sota-alignment.md)**.

### Reality check

It's **experimental**. Expect shimmer on fine detail, ghost trails on fast motion (use base-layer mode for HUDs), and the odd crash. Bidirectional mode adds ~1 frame of lag (skip it for competitive shooters). VRR mode only does something on an actual FreeSync/G-Sync display. The generated frames only fill vblanks the game left empty — if you're already at refresh, there's nothing to do. And the whole economic pitch — a spare/old card doing the heavy lifting — is also its main requirement.

### Start here

- 🕹️ **[How-To guide](doc/framegen-howto.md)** — plain-language setup: the two-card split, how to wire the display, copy-paste commands per mode, and current limits.
- 🎛️ **[Engineer's reference](doc/framegen-proposals/README.md)** — every flag and toggle, how the shipped pipeline works, and the design-proposal roadmap.
- 🔧 **[Architecture](doc/framegen-architecture.md)** — how it works under the hood, end to end.
- 🧰 **[Maintainer guide](doc/framegen-maintenance.md)** — contracts that must survive refactors and the publication/validation checklist.
- 📚 **[Research survey](doc/research-framegen.md)** — the frames-only state of the art this pipeline is measured against (primary-source cross-checked).

> The build target and binary are still `gamescope` (drop-in compatible with existing scripts, Steam launch options, and packaging); "gamescope-gameslop" is the fork's identity, not a rename of the executable.

---

## gamescope: the micro-compositor formerly known as steamcompmgr

In an embedded session usecase, gamescope does the same thing as steamcompmgr, but with less extra copies and latency:

 - It's getting game frames through Wayland by way of Xwayland, so there's no copy within X itself before it gets the frame.
 - It can use DRM/KMS to directly flip game frames to the screen, even when stretching or when notifications are up, removing another copy.
 - When it does need to composite with the GPU, it does so with async Vulkan compute, meaning you get to see your frame quick even if the game already has the GPU busy with the next frame.

It also runs on top of a regular desktop, the 'nested' usecase steamcompmgr didn't support.

 - Because the game is running in its own personal Xwayland sandbox desktop, it can't interfere with your desktop and your desktop can't interfere with it.
 - You can spoof a virtual screen with a desired resolution and refresh rate as the only thing the game sees, and control/resize the output as needed. This can be useful in exotic display configurations like ultrawide or multi-monitor setups that involve rotation.

It runs on Mesa + AMD or Intel, and could be made to run on other Mesa/DRM drivers with minimal work. AMD requires Mesa 20.3+, Intel requires Mesa 21.2+. For NVIDIA's proprietary driver, version 515.43.04+ is required (make sure the `nvidia-drm.modeset=1` kernel parameter is set).

If running RadeonSI clients with older cards (GFX8 and below), currently have to set `R600_DEBUG=nodcc`, or corruption will be observed until the stack picks up DRM modifiers support.

## Building

```
git submodule update --init
meson setup build/
ninja -C build/
build/src/gamescope -- <game>
```

Install with:

```
meson install -C build/ --skip-subprojects
```

## Keyboard shortcuts

* **Super + F** : Toggle fullscreen
* **Super + N** : Toggle nearest neighbour filtering
* **Super + U** : Toggle FSR upscaling
* **Super + Y** : Toggle NIS upscaling
* **Super + I** : Increase FSR sharpness by 1
* **Super + O** : Decrease FSR sharpness by 1
* **Super + S** : Take screenshot (currently goes to `/tmp/gamescope_$DATE.png`)
* **Super + G** : Toggle keyboard grab

## Examples

On any X11 or Wayland desktop, you can set the Steam launch arguments of your game as follows:

```sh
# Upscale a 720p game to 1440p with integer scaling
gamescope -h 720 -H 1440 -S integer -- %command%

# Limit a vsynced game to 30 FPS
gamescope -r 30 -- %command%

# Run the game at 1080p, but scale output to a fullscreen 3440×1440 pillarboxed ultrawide window
gamescope -w 1920 -h 1080 -W 3440 -H 1440 -b -- %command%
```

## Experimental frame generation

*The conceptual overview is at the [top of this README](#gamescope-gameslop) and in the [How-To](doc/framegen-howto.md); the [engineer's reference](doc/framegen-proposals/README.md) lists every flag. This section is the in-tree CLI reference — behaviour, timing, HDR/scene-change handling, and the dual-GPU example commands.*

`--experimental-framegen` enables compositor-side frame generation: for every
real frame gamescope composites, it generates one or more additional frames and
presents them on the empty vblanks the game left behind, multiplying the
perceived frame rate. `--framegen-multiplier` selects x2 (default), x3 or x4;
the number of frames actually inserted adapts to the measured frame gap.

Frame generation is designed for a dual-GPU split where a second GPU (e.g. an
AMD iGPU) composites and generates while the game renders on the primary GPU,
but it also works on a single GPU. Generated frames are the compositor's own —
they create no client commits, so they are invisible to the game, Wine/DXVK,
Reflex and anti-lag, and never produce fake presentation feedback.

This is a prototype. Keep the following in mind:

* **Causal generated frames only display in slots without newer real content.**
  A causal candidate is dropped if a new game frame needs that display slot;
  overlay-only updates are either boundedly deferred or late-composited in base
  mode. Bidirectional mode is the explicit exception: its interpolations and
  real endpoints form one delayed, content-ordered queue. On a
  dedicated framegen queue, gamescope also prepares speculative generated
  candidates after usable real frames so sudden missed vblanks already have
  work in flight on the second GPU. If a slow game exhausts that first batch,
  the dedicated queue can refill one prediction at a time while the real-frame
  history is still recent, bounded by the stall/scene-change timeout. If the
  game keeps up, those candidates are discarded before they can add latency. On
  the single/shared-queue path,
  admission stays conservative and framegen waits for a stable empty-vblank
  cadence before generating. For the most deterministic cadence, cap the game
  near an integer fraction of refresh (for example `-r 72` on a 144 Hz display).
* **It forces the composite path** (implies `--force-composite`), so direct
  scan-out is disabled while it is active. This costs a little latency and power
  versus a plain direct-scan-out frame. The default fixed-slot path suppresses
  adaptive sync (VRR), and all framegen paths suppress tearing flips. The opt-in
  `GAMESCOPE_FRAMEGEN_VRR_HYBRID=1` prototype is the VRR exception: on a
  dedicated framegen queue with an actively VRR-capable connector it schedules
  one generated flip inside the measured interval. If you prefer native VRR
  with no generated artifacts, run without `--experimental-framegen`.
* **Timing / latency.** In causal modes the real frame is presented immediately;
  bidirectional mode deliberately delays it by one real-frame interval so its
  interpolations can precede it. All
  framegen GPU work (history copy + generation) is submitted in a separate
  command buffer *after* the real frame's composite, so the real frame's page
  flip never waits on it. When the compositor's GPU exposes a second compute
  queue, all generation runs on a **dedicated
  queue** with its own timeline, so a slow generation can never sit in front of
  the next composite on the realtime queue — the composite is protected in
  hardware, not just by scheduling. (Set `GAMESCOPE_FRAMEGEN_SINGLE_QUEUE=1` to
  force the single-queue path.) A generated frame whose GPU work has not
  finished by its vblank is skipped (the display repeats the last frame) rather
  than waited on, and if the compositing GPU can't keep up at all, framegen goes
  dormant instead of queueing work in front of real frames. History is kept
  **zero-copy** by retaining references to the composited output images instead
  of copying them, and the generated-frame shader runs fp16 on capable hardware,
  so per-frame overhead stays low. The generated-frame algorithm is selected
  with `--framegen-mode`: `extrapolate` (default) predicts motion forward from
  the last two real frames so displayed motion stays monotonic and smooth (real
  N → generated N+⅓ → N+⅔ → real N+1 for x3); `motion` additionally estimates
  per-block motion (a low-res luma block matcher) and reprojects along it,
  falling back to extrapolation where the match is unconfident — higher quality
  on panning/scrolling at a higher compute cost; `blend` averages the last two
  real frames and is kept as a debug aid only. For `motion`,
  `--framegen-quality` selects a real cost ceiling: `low` runs forward matching
  only; `medium` adds reverse consistency and per-pixel agreement; `high`
  (default, compatible with the previous behavior) adds self-supervised
  adaptation and permits the optional learned refiner; `ultra` additionally
  retains and reprojects the preceding checked field to make a bounded causal
  acceleration prediction; `extreme` reconstructs a per-pixel motion-layer
  verdict from nearby field hypotheses using full-resolution color
  correspondence before applying that acceleration, then uses two-interval-old
  luma to validate adjacent background motion at locally diagnosed
  disocclusions. The displayed color still comes from the newest real frame,
  not history. The timestamp ladder
  sheds these tiers one at a time before falling back to plain extrapolation
  or reducing the multiplier.
* **Scene changes.** Prediction history is dropped automatically on focus
  change, overlay/notification appearance or disappearance, SDR↔HDR/EOTF
  changes, resolution or format changes, and long frame gaps, so stale content
  is never smeared across a scene transition. High and higher motion tiers also
  compare nine regional luminance histograms plus motion-compensated prediction
  error in the same GPU batch. A content cut with steady frametime duplicates
  the newest real frame instead of extrapolating across unrelated images (or
  selects the nearest real endpoint in bidirectional mode). Overlay-only
  repaints (e.g. a MangoHud update) never count as game
  frames and always win over a pending generated frame. Screenshots and
  pipewire streaming captures run separate passes and don't interact with
  frame generation.
* **HDR** is supported: with an HDR10 (PQ) output the prediction runs on
  PQ-encoded values (perceptually well-behaved), and with scRGB float output
  highlights above 1.0 and wide-gamut values survive generation. Generated
  frames bypass output color management because the history already stores
  color-managed output.
* **Ghosting.** Without motion vectors, `extrapolate` can ghost/halo around
  fast-moving edges. Two safeguards bound this: the prediction is faded out
  where the frame-to-frame change is large, and it is clamped to the local
  neighborhood of the current frame (TAA-style rectification), so it can never
  invent colors that aren't already visible around a pixel. If you still see
  ghosting, lower `--framegen-strength` (e.g. `0.35` or `0.25`). `0.0` disables
  the forward step entirely (the generated frame becomes a copy of the last real
  frame); `0.5` (default) gives the smoothest motion.

The GPU selector notation used below is `vendor:device`, written as lowercase
hexadecimal PCI IDs. You can find the IDs with:

```sh
lspci -nn | grep -Ei 'vga|3d|display'
vulkaninfo --summary
```

For example, an `lspci` line ending in `[1002:5678]` is written as
`1002:5678`, and a line ending in `[10de:1234]` is written as `10de:1234`.
Pass the compositor GPU to gamescope with `--prefer-vk-device vendor:device`.
For the game process after the `--` separator, use
`MESA_VK_DEVICE_SELECT='vendor:device!'` to request only that Vulkan device.
On NVIDIA Optimus systems, the child process may also need
`__NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only`.

```sh
# Replace these example IDs with values reported by lspci on your system.
PRESENT_DEV=1002:5678
RENDER_DEV=10de:1234!

# Dual-GPU: composite on the present GPU, render vkcube on the render GPU,
# upscale 1440p -> 2160p with FSR, and use x2 frame generation.
gamescope \
    --prefer-vk-device "$PRESENT_DEV" \
    --experimental-framegen \
    -w 2560 -h 1440 -W 3840 -H 2160 \
    -F fsr -f -- \
    env MESA_VK_DEVICE_SELECT="$RENDER_DEV" vkcube

# Same, with the low-latency default made explicit and diagnostics enabled.
# --debug-dual-gpu-route logs GPU/buffer/frame-path routing; --framegen-debug
# logs framegen history, dispatch, and present cadence.
gamescope \
    --debug-dual-gpu-route \
    --prefer-vk-device "$PRESENT_DEV" \
    --experimental-framegen --framegen-mode extrapolate --framegen-debug \
    -w 2560 -h 1440 -W 3840 -H 2160 \
    -F fsr -f -- \
    env MESA_VK_DEVICE_SELECT="$RENDER_DEV" vkcube
```

### Dual-GPU test commands

Export the render and present IDs once, using the values for your system:

```sh
export RENDER_DEV=10de:1234!
export PRESENT_DEV=1002:5678
```

Common framegen settings used in the tests:

```sh
--prefer-vk-device "$PRESENT_DEV"     # gamescope/compositor/framegen device
--experimental-framegen
--framegen-mode motion                # motion-compensated generation
--framegen-multiplier 3               # x3 target cadence
--framegen-strength 0.5
--framegen-debug
-W 2560 -H 1440 -r 120
```

If this checkout was built against locally compiled Wayland/WSI dependencies,
run through the local environment wrapper:

```sh
GAMESCOPE_BUILD_DIR=build-perf ./env-gamescope-local.sh gamescope ...
```

For Vulkan clients, pin the game/render GPU with Mesa's Vulkan selector after
the `--` separator:

```sh
GAMESCOPE_BUILD_DIR=build-perf ./env-gamescope-local.sh \
gamescope \
    --prefer-vk-device "$PRESENT_DEV" \
    -W 2560 -H 1440 -r 120 \
    --expose-wayland \
    --experimental-framegen \
    --framegen-mode motion \
    --framegen-multiplier 3 \
    --framegen-strength 0.5 \
    --framegen-debug -- \
    env MESA_VK_DEVICE_SELECT="$RENDER_DEV" vkmark --size 2560x1440
```

SuperTuxKart 1.4 is an OpenGL client here, so `MESA_VK_DEVICE_SELECT` does not
select its render GPU. Force STK through X11/GLX and use NVIDIA's GL offload
variables for the child process instead:

```sh
GAMESCOPE_BUILD_DIR=build-perf ./env-gamescope-local.sh \
gamescope \
    --prefer-vk-device "$PRESENT_DEV" \
    -W 2560 -H 1440 -r 120 \
    --expose-wayland \
    --experimental-framegen \
    --framegen-mode motion \
    --framegen-multiplier 3 \
    --framegen-strength 0.5 \
    --framegen-debug -- \
    env IRR_DEVICE_TYPE=x11 \
        __NV_PRIME_RENDER_OFFLOAD=1 \
        __GLX_VENDOR_LIBRARY_NAME=nvidia \
        supertuxkart \
            --race-now \
            --track=xr591 \
            --numkarts=8 \
            --laps=3 \
            --difficulty=2 \
            --screensize=2560x1440 \
            --windowed \
            --render-driver=gl \
            --unlock-all \
            --disable-polling \
            --profile-time=18
```

Before trusting an OpenGL game, verify the split with a quick probe:

```sh
GAMESCOPE_BUILD_DIR=build-perf ./env-gamescope-local.sh \
gamescope --prefer-vk-device "$PRESENT_DEV" -W 1280 -H 720 -r 120 \
    --expose-wayland -- \
    env IRR_DEVICE_TYPE=x11 \
        __NV_PRIME_RENDER_OFFLOAD=1 \
        __GLX_VENDOR_LIBRARY_NAME=nvidia \
        glxinfo -B
```

The gamescope log should report `$PRESENT_DEV`, while `glxinfo` or
SuperTuxKart should report the NVIDIA OpenGL renderer. Nested runs like the
commands above validate routing and shader execution, but native scanout pacing
still needs a real DRM session from a text VT, for example through
`./run-framegen-native.sh` with the same framegen mode/multiplier/strength.

`--experimental-framegen` already implies `--force-composite`, so you don't need
to pass `--force-composite` separately.

## Options

See `gamescope --help` for a full list of options.

* `-W`, `-H`: set the resolution used by gamescope. Resizing the gamescope window will update these settings. Ignored in embedded mode. If `-H` is specified but `-W` isn't, a 16:9 aspect ratio is assumed. Defaults to 1280×720.
* `-w`, `-h`: set the resolution used by the game. If `-h` is specified but `-w` isn't, a 16:9 aspect ratio is assumed. Defaults to the values specified in `-W` and `-H`.
* `-r`: set a frame-rate limit for the game. Specified in frames per second. Defaults to unlimited.
* `-o`: set a frame-rate limit for the game when unfocused. Specified in frames per second. Defaults to unlimited.
* `-F fsr`: use AMD FidelityFX™ Super Resolution 1.0 for upscaling
* `-F nis`: use NVIDIA Image Scaling v1.0.3 for upscaling
* `-S integer`: use integer scaling.
* `-S stretch`: use stretch scaling, the game will fill the window. (e.g. 4:3 to 16:9)
* `-b`: create a border-less window.
* `-f`: create a full-screen window.
* `--experimental-framegen`: enable experimental compositor-side frame generation (x2–x4). Implies `--force-composite`; the default fixed-slot path disables adaptive sync, and tearing remains disabled. `GAMESCOPE_FRAMEGEN_VRR_HYBRID=1` is the opt-in VRR exception. See [Experimental frame generation](#experimental-frame-generation).
* `--framegen-mode`: generated-frame algorithm, `extrapolate` (default, low latency), `motion` (motion-compensated, higher quality/cost) or `blend` (debug).
* `--framegen-quality`: motion quality/cost ceiling: `low`, `medium`, `high` (default), `ultra`, or `extreme`. Lower tiers skip whole passes; `ultra` adds causal temporal acceleration and `extreme` adds full-resolution color-guided motion reconstruction, a bounded three-frame disocclusion resolver (`GAMESCOPE_FRAMEGEN_RESERVOIR=0`), and with ML a causal shading-persistence correction (`GAMESCOPE_FRAMEGEN_SHADING=0`).
* `--framegen-strength`: forward-extrapolation step for `extrapolate`/`motion` modes, `0.0`–`1.0` (default `0.5`). Lower values reduce ghosting.
* `--framegen-multiplier`: generated-frame multiplier, `2` (default), `3` or `4`. The number of frames actually displayed adapts to empty vblanks; on a dedicated framegen queue gamescope may speculatively prepare candidates that are later dropped if real content arrives first.
* `--framegen-debug`: log framegen history, dispatch, and present cadence.

## Reshade support

Gamescope supports a subset of Reshade effects/shaders using the `--reshade-effect [path]` and `--reshade-technique-idx [idx]` command line parameters.

This provides an easy way to do shader effects (ie. CRT shader, film grain, debugging HDR with histograms, etc) on top of whatever is being displayed in Gamescope without having to hook into the underlying process.

Uniform/shader options can be modified programmatically via the `gamescope-reshade` wayland interface. Otherwise, they will just use their initializer values.

Using Reshade effects will increase latency as there will be work performed on the general gfx + compute queue as opposed to only using the realtime async compute queue which can run in tandem with the game's gfx work.

Using Reshade effects is **highly discouraged** for doing simple transformations which can be achieved with LUTs/CTMs which are possible to do in the DC (Display Core) on AMDGPU at scanout time, or with the current regular async compute composite path.
The looks system where you can specify your own 3D LUTs would be a better alternative for such transformations.

Pull requests for improving Reshade compatibility support are appreciated.

## Status of Gamescope Packages

[![Packaging status](https://repology.org/badge/vertical-allrepos/gamescope.svg?exclude_unsupported=1)](https://repology.org/project/gamescope/versions)
