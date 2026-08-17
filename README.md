# gamescope-gameslop

gamescope-gameslop is a fork of [Valve's gamescope](https://github.com/ValveSoftware/gamescope) with frame generation in the compositor. It works from finished frames only. It does not depend on a GPU vendor, game engine, or per-game integration. The name says what most displayed pixels are: generated in-between frames. The game never sees them and the HUD counts them separately. The binary is still named `gamescope`.

## Two GPUs, one display

The intended setup uses a second, cheaper GPU as a poor man's DLSS/FSR frame generator. The stronger GPU renders the game without frame-generation work. An iGPU or older card estimates motion from the completed frames, generates frames for the gaps, composites the result, and drives the display.

```text
 CARD 1 — render GPU                         CARD 2 — present GPU
 ┌──────────────────────────┐                ┌──────────────────────────────┐
 │ renders the game         │   finished     │ estimates motion, generates  │
 │ with no frame-generation │   frames       │ in-between frames, composites│
 │ work                     │ ──────────────▶│ and drives the display       │
 └──────────────────────────┘   dma-buf      └──────────────────────────────┘
```

Running generation on the render GPU takes resources from the game. A second GPU makes that work close to free from the game's point of view. In a matched GravityMark measurement, bare performance was 100%, gamescope with frame generation disabled was 95.8%, and generation enabled was 95.3%. Compositing cost about 4%; enabling generation cost a further 0.5%. Single-GPU operation works, but is usually not worth it unless the game leaves substantial GPU headroom.

## Reverse VRR

VRR bends the display clock to the game's frame times. gamescope-gameslop bends
the content stream onto the display's fixed refresh grid. Each generated frame is
planned against a real vblank deadline.

To keep latency low, the default path extrapolates: each generated frame is
predicted forward from the last real frame and its motion, so no real frame is
held back to interpolate between two. The causal scheduler uses deadline-based,
just-in-time slot planning. In the default forward path, a real frame is never
queued behind generated work. On native KMS a real frame that arrives inside an
already-committed generated slot lands at the next vblank. Late
generation is dropped instead of waited for, so the display repeats the last
frame for that slot. Bidirectional interpolation is also available. It uses both
real endpoints and deliberately adds one real-frame interval of latency.

## In-situ learning

An optional small refiner network trains while the game runs and can persist a
profile per game. Its output is bounded: it can veto estimated motion, but cannot
invent scene detail. Offline training and evaluation tools are also included.

## Real FPS and displayed FPS

Generated frames contain no new input. They improve motion smoothness, not
reaction time. The game's FPS counter therefore continues to report the real
render rate. The gamescope HUD and metrics report the real, generated, and
repeated frames delivered to the display.

## Requirements

- Two Vulkan GPUs visible to the compositor are recommended. Any pair that can share dma-bufs should work. Validated: NVIDIA (proprietary) render to AMD (RADV) present. Intel, and NVIDIA as the present GPU, are untested.
- The display must be connected to the present GPU. That GPU runs gamescope, frame generation, composition, and scanout.
- Native operation requires Linux DRM/KMS. A nested window in an existing X11 or Wayland session is sufficient for initial testing.
- Xwayland and Proton games can be assigned to the render GPU with `MESA_VK_DEVICE_SELECT`.
- NVIDIA rendering with the proprietary driver requires its dma-buf support. As in upstream gamescope, use driver 515.43.04 or newer and enable `nvidia-drm.modeset=1`.

## Build

The build uses Meson and Ninja like upstream. Frame generation adds no external
dependencies. See [`meson_options.txt`](meson_options.txt) for build options.

```sh
git submodule update --init
meson setup build/
ninja -C build/
build/src/gamescope --help
```

Install with:

```sh
meson install -C build/ --skip-subprojects
```

## Quick start

Find the PCI `vendor:device` IDs for both GPUs:

```sh
lspci -nn | grep -Ei 'vga|3d|display'
vulkaninfo --summary
```

Set the present and render devices, then start a nested x2 motion/checked run.
The example client is `vkcube`; replace it with the game command.

```sh
PRESENT_DEV=1002:5678
RENDER_DEV='10de:1234!'
GAMESCOPE_FRAMEGEN_HUD=2 gamescope \
  --prefer-vk-device "$PRESENT_DEV" \
  --experimental-framegen \
  --framegen-mode motion \
  --framegen-multiplier 2 \
  --framegen-pipeline checked -- \
  env MESA_VK_DEVICE_SELECT="$RENDER_DEV" vkcube
```

`--prefer-vk-device` selects the present GPU. The command after `--` selects the
render GPU. On NVIDIA Optimus, the client may also need
`__NV_PRIME_RENDER_OFFLOAD=1` and `__VK_LAYER_NV_optimus=NVIDIA_only`.

Set `GAMESCOPE_FRAMEGEN_HUD=1` for the compact HUD or `2` for the pacing and
learning rows. The `present … render … buffers xGPU` row verifies that gamescope
runs on the present card while client buffers arrive from the render card.
`buffers local` means the client is rendering on the present GPU instead.

For native DRM/KMS, run the preflight first and follow the native launch section
in the [user guide](doc/framegen-howto.md):

```sh
./scripts/framegen-doctor.sh
```

The supplied [`scripts/run-framegen-native.sh`](scripts/run-framegen-native.sh)
helper can then launch from a text VT.

## Modes and variants

`--framegen-mode` accepts exactly three values. The other rows below are
environment-controlled presentation variants.

| Selection | Setting | Behavior |
|---|---|---|
| Extrapolate | `--framegen-mode extrapolate` | Default forward prediction; lowest generation cost |
| Motion | `--framegen-mode motion` | Motion-compensated forward prediction; recommended |
| Blend | `--framegen-mode blend` | Frame average retained as a debug aid |
| Bidirectional interpolation | `GAMESCOPE_FRAMEGEN_BIDIR=1` with motion | Uses two real endpoints; adds one frame interval of latency |
| Base layer | `GAMESCOPE_FRAMEGEN_BASE=1` | Generates below compositor overlays and the gamescope cursor, which are composited late; in-game UI is part of the game frame and is not separated; cannot combine with bidirectional mode |
| VRR hybrid | `GAMESCOPE_FRAMEGEN_VRR_HYBRID=1` with `--adaptive-sync` | Experimental causal path for a VRR display; not yet validated on a VRR panel |

The multiplier is x2, x3, or x4. x2 and x3 are the useful starting points. x4
is the ceiling and only fills the grid when the present GPU and source cadence
leave enough time.

## Motion pipelines

Each motion pipeline adds work to the preceding one.

| Pipeline | Work added | Status |
|---|---|---|
| `warp` | Forward motion match | Cheapest; default |
| `checked` | Reverse consistency and edge agreement | Recommended step up |
| `learned` | Self-supervised adaptation and optional refiner | Experimental; can smear |
| `predict` | Bounded acceleration from the previous checked field | Experimental; can smear |
| `guided` | Color-guided reconstruction and three-frame disocclusion handling | Full pass set; experimental; can smear |

Use `warp` if the present GPU cannot sustain `checked`. The `learned`,
`predict`, and `guided` pipelines add passes and should be treated as
experiments.

The deadline-driven degradation ladder steps down automatically when measured
GPU time no longer fits a slot. It probes upward again after sustained headroom.

## Recommended settings (start with these)

Start with motion, `checked`, and x2. Try x3 after the HUD shows stable deadline
hits. This combination was validated natively on 2026-08-16 (build 1533e44) with a Proton game
rendering on an NVIDIA laptop GPU and generation on the AMD 890M at 2560x1440@120, x3
checked, 15 minutes: dl_hit 0.998, fill 0.96–0.97 in steady windows (0.94 including
loading screens), source-ready-to-flip p50 8.75 ms / p95 12.25 ms, clean shutdown.

```sh
PRESENT_DEV=1002:5678
RENDER_DEV='10de:1234!'
GAMESCOPE_FRAMEGEN_HUD=2 \
GAMESCOPE_FRAMEGEN_METRICS=1 \
gamescope --expose-wayland --backend wayland \
  --prefer-vk-device "$PRESENT_DEV" \
  -W 2560 -H 1440 -r 120 -f \
  --experimental-framegen \
  --framegen-mode motion \
  --framegen-multiplier 2 \
  --framegen-pipeline checked -- \
  env MESA_VK_DEVICE_SELECT="$RENDER_DEV" vkcube
```

For an optional `learned`, `predict`, or `guided` run, set the online-learning
environment before launching and change the pipeline in the command:

```sh
export GAMESCOPE_FRAMEGEN_NET_ONLINE=1
export GAMESCOPE_FRAMEGEN_NET_EVERY=2
export GAMESCOPE_FRAMEGEN_NET_PROFILE="$HOME/.cache/gamescope-fg-example.bin"
```

Keep `NET_EVERY=2` on an iGPU-class present GPU; use `1` on a desktop present
GPU. Give each game its own `NET_PROFILE`. Nested mode is suitable for routing
and visual checks. Native DRM/KMS is the relevant test for scanout pacing.

## Measuring the output

`GAMESCOPE_FRAMEGEN_METRICS=1` emits a summary every five seconds. Key fields are:

| Field | Meaning |
|---|---|
| `real` / `gen` / `rep` | Real frames, generated frames, and repeated refresh slots |
| `dl_hit` | Fraction of generated and delayed-real presents that met their planned deadline. It says nothing about how many slots were filled; read `rep` and the HUD `rates` row for that |
| `bias_ms` | Display-chain timing bias learned by the scheduler |
| `lead_viable` | Learned minimum viable native KMS commit lead, in milliseconds |

HUD level 2 adds `rates`, `ladder`, and `pace` rows. They show source/generated
rates, the active degradation rung and recovery state, and deadline hit rate plus
jitter. The device row is the routing check; the rates row is the output-rate
check.

## Current limits

- Frame generation does not lower input latency and generated frames add no new
  input samples.
- Disocclusions, transparency, particles, fast edges, and UI can produce
  ghosting, shimmer, or smearing. Base-layer mode helps with compositor overlays
  and the cursor, but cannot separate arbitrary in-game UI.
- Bidirectional mode is laggy by design because it waits for the next real frame.
- Single-GPU use competes with the game and is not recommended for a saturated
  render GPU.
- The VRR hybrid path exists but has not been validated on a VRR panel.
- Cursor motion does not advance on generated frames in the default output-space
  path: the cursor moves with real composites. Base-layer mode composites the
  cursor late, so it does advance there.
- An overlay-only repaint can be held back up to four vblanks so a ready
  generated frame can fill the slot instead.
- A late generated frame is dropped. Overload appears as repeated display slots,
  not a delayed real frame in the default causal path.
- The feature is experimental. Native DRM/KMS, HDR, unusual dma-buf modifier
  combinations, and mixed-vendor systems need broader testing.

## Documentation

- [`doc/framegen-howto.md`](doc/framegen-howto.md) — user guide and native setup
- [`doc/framegen-architecture.md`](doc/framegen-architecture.md) — implementation and scheduling design
- [`doc/framegen-proposals/`](doc/framegen-proposals/) — design notes 01–07 and the engineer's flag reference
- [`doc/framegen-maintenance.md`](doc/framegen-maintenance.md) — invariants, validation, and publication checks
- [`doc/research-framegen.md`](doc/research-framegen.md) — frames-only frame-generation research
- [`scripts/`](scripts/) — doctor, nested/native launch helpers, and network training/evaluation tools

## Upstream gamescope

gamescope-gameslop retains upstream gamescope and its command-line interface.
Valve's work remains the foundation of this fork.

### gamescope: the micro-compositor formerly known as steamcompmgr

In an embedded session use case, gamescope performs the role of steamcompmgr with
fewer extra copies and less latency:

- It receives game frames through Wayland by way of Xwayland, avoiding a copy
  inside X before gamescope receives the frame.
- It can use DRM/KMS to flip game frames directly to the screen, including when
  stretching or showing notifications.
- GPU composition uses asynchronous Vulkan compute.

It can also run nested on a regular desktop. Each game gets its own Xwayland
sandbox, and gamescope can expose a virtual resolution and refresh rate before
scaling the output.

Upstream supports Mesa on AMD and Intel. AMD requires Mesa 20.3 or newer; Intel
requires Mesa 21.2 or newer. NVIDIA's proprietary driver requires 515.43.04 or
newer with `nvidia-drm.modeset=1`. RadeonSI clients on GFX8 and older hardware
may require `R600_DEBUG=nodcc` until the stack provides DRM modifier support.

### Building

The [build commands above](#build) build both gamescope and the fork's frame
generation. Upstream installation uses the same Meson target.

### Keyboard shortcuts

- **Super + F**: toggle fullscreen
- **Super + N**: toggle nearest-neighbour filtering
- **Super + U**: toggle FSR upscaling
- **Super + Y**: toggle NIS upscaling
- **Super + I**: increase FSR sharpness by 1
- **Super + O**: decrease FSR sharpness by 1
- **Super + S**: take a screenshot in `/tmp/gamescope_$DATE.png`
- **Super + G**: toggle keyboard grab

### Examples

Set a Steam game's launch options to one of these commands:

```sh
# Upscale a 720p game to 1440p with integer scaling.
gamescope -h 720 -H 1440 -S integer -- %command%

# Limit a vsynced game to 30 FPS.
gamescope -r 30 -- %command%

# Render at 1080p in a fullscreen 3440x1440 pillarboxed window.
gamescope -w 1920 -h 1080 -W 3440 -H 1440 -b -- %command%
```

### Options

Run `gamescope --help` for the complete list.

- `-W`, `-H`: set the gamescope output resolution. Defaults to 1280x720.
- `-w`, `-h`: set the game resolution. Defaults to the output resolution.
- `-r`: set the game frame-rate limit. Defaults to unlimited.
- `-o`: set the unfocused frame-rate limit. Defaults to unlimited.
- `-F fsr`: use AMD FidelityFX Super Resolution 1.0 for upscaling.
- `-F nis`: use NVIDIA Image Scaling v1.0.3 for upscaling.
- `-S integer`: use integer scaling.
- `-S stretch`: stretch the game to fill the window.
- `-b`: create a borderless window.
- `-f`: create a fullscreen window.
- `--experimental-framegen`: enable the fork's compositor-side frame generation.
  See [modes and variants](#modes-and-variants) for the related options.

### ReShade support

Gamescope supports a subset of ReShade effects with `--reshade-effect [path]`
and `--reshade-technique-idx [idx]`. Uniform and shader options can be changed
through the `gamescope-reshade` Wayland interface; otherwise their initializer
values are used.

ReShade work runs on the general graphics and compute queue and can add latency.
For simple transforms, prefer LUTs or color transformation matrices that can be
applied at scanout. Contributions that improve ReShade compatibility are welcome.

### Packages

[![Packaging status](https://repology.org/badge/vertical-allrepos/gamescope.svg?exclude_unsupported=1)](https://repology.org/project/gamescope/versions)

## License and credits

gamescope-gameslop is based on Valve's gamescope. See [`LICENSE`](LICENSE) for
the BSD 2-Clause license, Valve copyright, and licenses for incorporated code.
