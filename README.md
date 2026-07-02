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

`--experimental-framegen` enables compositor-side x2 frame generation: for every
real frame gamescope composites, it generates one additional frame and presents
it on the next vblank, doubling the perceived frame rate.

This is a prototype. Keep the following in mind:

* **It only helps when the game runs at or below half your display's refresh
  rate.** Frame generation fills vblanks the game left empty. Real frames always
  take priority: a generated frame is dropped, never presented, if a new real
  frame is ready for the same vblank, and generation is skipped automatically
  while the game outpaces ~⅔ of the refresh rate. Still, cap the game with `-r`
  to half the refresh rate (e.g. `-r 72` on a 144 Hz display) so the cadence is
  stable instead of oscillating.
* **It forces the composite path** (implies `--force-composite`), so direct
  scan-out is disabled while it is active. This costs a little latency and power
  versus a plain direct-scan-out frame.
* **Timing / latency.** The real frame is always presented immediately — frame
  generation adds no extra latency to the frames the game actually rendered. All
  framegen GPU work (history copy + generation) is submitted in a separate
  command buffer *after* the real frame's composite, so the real frame's page
  flip never waits on it. The default `extrapolate` mode predicts motion
  forward, so displayed motion stays monotonic and smooth (real N → generated
  N+½ → real N+1 → …). The alternative `blend` mode averages the last two real
  frames; it looks softer but places the generated frame temporally *between*
  older frames, which can read as judder.
* **Ghosting.** Without motion vectors, `extrapolate` can ghost/halo around
  fast-moving edges. Two safeguards bound this: the prediction is faded out
  where the frame-to-frame change is large, and it is clamped to the local
  neighborhood of the current frame (TAA-style rectification), so it can never
  invent colors that aren't already visible around a pixel. If you still see
  ghosting, lower `--framegen-strength` (e.g. `0.35` or `0.25`). `0.0` disables
  the forward step entirely (the generated frame becomes a copy of the last real
  frame); `0.5` (default) gives the smoothest motion.

```sh
# Dual-GPU: composite on the AMD iGPU (1002:150e), render vkcube on the NVIDIA
# dGPU (10de:2db9), upscale 1440p -> 2160p with FSR, x2 frame generation.
gamescope \
    --prefer-vk-device 1002:150e \
    --experimental-framegen \
    -w 2560 -h 1440 -W 3840 -H 2160 \
    -F fsr -f -- \
    env MESA_VK_DEVICE_SELECT='10de:2db9!' vkcube

# Same, with the low-latency default made explicit and diagnostics enabled.
# --debug-dual-gpu-route logs GPU/buffer/frame-path routing; --framegen-debug
# logs framegen history, dispatch, and present cadence.
gamescope \
    --debug-dual-gpu-route \
    --prefer-vk-device 1002:150e \
    --experimental-framegen --framegen-mode extrapolate --framegen-debug \
    -w 2560 -h 1440 -W 3840 -H 2160 \
    -F fsr -f -- \
    env MESA_VK_DEVICE_SELECT='10de:2db9!' vkcube
```

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
* `--experimental-framegen`: enable experimental x2 compositor-side frame generation. Implies `--force-composite`. See [Experimental frame generation](#experimental-frame-generation).
* `--framegen-mode`: generated-frame algorithm, `extrapolate` (default, low latency) or `blend`.
* `--framegen-strength`: forward-extrapolation step for `extrapolate` mode, `0.0`–`1.0` (default `0.5`). Lower values reduce ghosting.
* `--framegen-multiplier`: generated-frame multiplier. Only `2` is supported for now.
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
