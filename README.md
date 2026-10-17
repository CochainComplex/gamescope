# gamescope-gameslop

gamescope-gameslop is a fork of [Valve's gamescope](https://github.com/ValveSoftware/gamescope) that adds frame generation inside the compositor (the part of gamescope that puts finished game frames on your screen). It works from the finished frames only, so it needs no support from the GPU vendor, the game engine, or the game. The binary is still named `gamescope`, and the feature is **experimental**.

## In plain words

- Your game renders at, say, 60 FPS. Gamescope invents extra pictures between the real ones, so a 120 Hz screen looks smoother.
- It does **not** make the controls more responsive. Generated frames contain no new input.
- Cost: about 2 ms of extra delay in the default modes. The optional bidirectional mode adds about 20 to 45 ms more than the default modes (more at low frame rates).
- For best results, use a second GPU (a laptop iGPU or an older card) to do the generating. It also works on one GPU.
- It only fills gaps: your game must run below your monitor's refresh rate (for example 60 FPS on a 120 Hz screen). If the game already reaches the refresh rate, nothing is generated.
- You need to build this fork from source; the distribution's gamescope does not have the feature.

| Setup | Smoothness | Added delay (framegen alone) | GPU load |
|---|---|---|---|
| Default (`extrapolate`, or `motion` with `warp`/`checked`) | Better than no frame generation | About 2 ms | Low |
| `learned` pipeline (recommended for real games) | Cleaner moving edges than `warp` or `checked` | About 2 ms | Medium (more work per generated frame) |
| Bidirectional (`GAMESCOPE_FRAMEGEN_BIDIR=1`, opt-in) | Smoothest | About 2 ms plus 20 to 45 ms extra for bidir (more at low frame rates); about 55 ms total from game frame to screen at ~40 fps, against ~17 ms in forward mode | Medium to high (a second motion field; bidir ignores the extra `guided` passes) |

The numbers come from our own tests on one test rig; yours will differ. Several features are not validated yet (see [Known limits](#known-limits)).

New here? Read [Requirements](#requirements), then [Install](#install), then [Quick start](#quick-start). Something wrong? See [Troubleshooting](doc/framegen-howto.md#7-troubleshooting). The full player guide is [`doc/framegen-howto.md`](doc/framegen-howto.md); every document is indexed in [`doc/README.md`](doc/README.md).

## How it works

The intended setup uses a second, cheaper GPU as the frame generator. The stronger GPU (the render GPU) renders the game and does no frame-generation work. The second GPU (the present GPU, for example an iGPU or an older card) looks at the finished frames, works out how things moved, creates the in-between frames, combines them with overlays into the final picture, and drives the display. Your monitor must be plugged into the present GPU.

```text
 CARD 1 — render GPU                         CARD 2 — present GPU
 ┌──────────────────────────┐                ┌──────────────────────────────┐
 │ renders the game         │   finished     │ estimates motion, generates  │
 │ with no frame-generation │   frames       │ in-between frames, composites│
 │ work                     │ ──────────────▶│ and drives the display       │
 └──────────────────────────┘  (shared GPU   └──────────────────────────────┘
                                 buffers)
```

Words used above:

- **Motion** (the "motion field"): an estimate of where each part of the picture moved between two real frames.
- **Composite**: combining the game picture with overlays (cursor, HUD) into the final frame.
- **Shared GPU buffers**: memory that both GPUs can read, so the frames move between cards without a trip through the CPU.

Running generation on the render GPU takes resources from the game; a second GPU makes that work close to free from the game's point of view. Single-GPU operation works, but is usually not worth it unless the game leaves substantial GPU headroom. In one older GravityMark measurement (topology not recorded; it is not in the [testing doc](doc/framegen-testing.md), so treat it as indicative only), bare performance was 100%, gamescope with generation disabled was 95.8%, and generation enabled was 95.3%.

What to expect while it runs:

- The game's FPS counter keeps reporting the real render rate. The gamescope HUD reports real, generated, and repeated frames separately.
- If a generated frame would be late for the display, it is dropped and the last frame is repeated. A real frame is never held up by generated work in the default path.
- The default `extrapolate` and `motion` modes hold no real frame back. The composite that every real frame needs while frame generation is on costs about 2 ms.
- Optional bidirectional mode deliberately shows real frames later (about 20 to 45 ms more than the default modes in our tests, more at lower frame rates) in exchange for smoother motion.

How the scheduler, the bidirectional delay controller, and the optional in-game learning work is described in the [Architecture](doc/framegen-architecture.md) document.

## Requirements

- Test tools: `vkcube` and `vulkaninfo` (package `vulkan-tools` on most distributions). `vkmark` is optional.
- The doctor requires `wayland-server` >= 1.23.1 and `pixman-1` >= 0.43.0. These are known-good versions; the Meson files accept older ones (see [Install](#install)).
- Two Vulkan GPUs visible to the compositor are recommended; one GPU also works (see the single-GPU command in [Quick start](#quick-start)). Any pair that can share GPU buffers should work. Validated: NVIDIA (proprietary) render to AMD (RADV) present. Intel, and NVIDIA as the present GPU, are untested.
- The display must be connected to the present GPU. That GPU runs gamescope, frame generation, composition, and scanout.
- Native operation requires Linux DRM/KMS. A nested window (gamescope opening a window inside your existing X11 or Wayland desktop) is enough for first tests.
- NVIDIA rendering with the proprietary driver requires its dma-buf support. As in upstream gamescope, use driver 515.43.04 or newer and enable `nvidia-drm.modeset=1`.
- Vulkan games, including games run under Xwayland or Proton (DXVK), can be assigned to the render GPU with `MESA_VK_DEVICE_SELECT`. OpenGL games need other variables (see the note after [Quick start](#quick-start) step 3).

## Install

The build uses Meson and Ninja like upstream. Frame generation adds no external dependencies beyond upstream gamescope's, so install the same build dependencies you would for upstream gamescope (for example `apt build-dep gamescope` or `dnf builddep gamescope`). After building, `./scripts/framegen-doctor.sh` (Quick start step 1) checks your setup. See [`meson_options.txt`](meson_options.txt) for build options.

### 1. Build

```sh
git submodule update --init
meson setup build/ -Denable_tests=false
ninja -C build/
build/src/gamescope --help
```

`--help` should list `--experimental-framegen`. A distribution-packaged `gamescope` does not have it and fails with an unrecognized option, so the Quick start needs this freshly built binary.

Build fails? Run the preflight, which checks the build tools and libraries and prints a fix for the first failure:

```sh
./scripts/framegen-doctor.sh
```

- It checks `wayland-server` 1.23.1 or newer and `pixman-1` 0.43.0 or newer. If your distribution's versions are older, build them into one prefix and add `-Dpkg_config_path=PREFIX/lib/pkgconfig:PREFIX/share/pkgconfig` to `meson setup`.
- `catch2-with-main` is optional. Without it, keep `-Denable_tests=false` as above.

### 2. Choose how to run it

| Option | How | Catch |
|---|---|---|
| A. Run from the build tree (recommended for a first test) | Source the environment script, see below. | Nothing on your system changes. Remove it by deleting the build directory. |
| B. Install | `meson install -C build/ --skip-subprojects` | The new binary goes where a distribution-packaged gamescope would live, so it overlaps with that package. Remove or hold the distribution package first if your package manager objects. |

**Option A, simplest:** in the terminal you will use, run:

```sh
cd /path/to/this/repo && source ./scripts/env-gamescope-local.sh
```

After that, plain `gamescope` in this terminal is your build (it puts `build/src` on `PATH` and sets the WSI layer path), so the Quick start commands work as written. Steam launch options do not inherit this. From an uninstalled build, the absolute binary path alone runs without the fork's WSI layer, so wrap the command: `/path/to/this/repo/scripts/env-gamescope-local.sh /path/to/this/repo/build/src/gamescope ... -- %command%`. Or install (option B).

**Hybrid launcher from a build tree:** [`scripts/gamescope-hybrid.sh`](scripts/gamescope-hybrid.sh) finds `build/src/gamescope` itself, but it does not set the WSI layer path. Run it through the environment script, or install the build (option B) first:

```sh
./scripts/env-gamescope-local.sh /path/to/this/repo/scripts/gamescope-hybrid.sh --steer-child ...
```

## Quick start

The examples below are all one command with different pieces filled in. Read the template first.

```text
[ENV VARS] gamescope [DISPLAY] [GPUS] [FRAMEGEN] -- [env MESA_VK_DEVICE_SELECT=RENDER_GPU] game

ENV VARS : GAMESCOPE_FRAMEGEN_HUD=2  (shows the on-screen HUD), NET_ONLINE, NET_PROFILE, ...
DISPLAY  : -W <width> -H <height> -r <monitor Hz> -f   (-f = fullscreen)
GPUS     : --prefer-vk-device <present GPU>            (two GPUs only)
FRAMEGEN : --experimental-framegen --framegen-mode motion
           --framegen-multiplier 2 --framegen-pipeline checked
```

Every later example is this template with a different pipeline, environment variable, or launcher. Defaults if you leave a piece out: mode `extrapolate`, pipeline `warp`, multiplier 2, window 1280x720 (the nested test window is small on purpose; add `-W 1920 -H 1080` for a bigger one).

### Step 0. Which setup are you?

- **Laptop with an iGPU and a dGPU:** run `./scripts/framegen-doctor.sh` (step 1), then use the [hybrid launcher command](#a-good-start-for-most-games) (it detects the GPUs for you). The one-line laptop test is in step 2.
- **Desktop with two GPUs:** run `./scripts/framegen-doctor.sh` and copy its two `export` lines (step 1).
- **One GPU:** use the one-GPU command in step 2. It works, but a second GPU is what makes frame generation cheap for the game.

### Step 1. Find your GPUs

The present GPU is the one your monitor cable is plugged into. The render GPU is the other one. Let the doctor script find them:

```sh
./scripts/framegen-doctor.sh
```

On a two-GPU machine it prints two lines like these. Copy them into your terminal as printed:

```sh
export RENDER_DEV=10de:1234!
export PRESENT_DEV=1002:5678
```

Typed in an interactive shell, the unquoted `!` works. In a script or in Steam launch options, put quotes around it (`'10de:1234!'`).

The IDs are PCI `vendor:device` pairs: `10de` is NVIDIA, `1002` is AMD, `8086` is Intel. The trailing `!` on the render GPU makes the choice exclusive. To find the IDs by hand instead:

```sh
lspci -nn | grep -Ei 'vga|3d|display'
vulkaninfo --summary
```

### Step 2. Run a nested test

Nested means gamescope opens a window inside your running desktop. It needs no text console.

Frame generation only fills gaps. If the test program or game already runs at or above your refresh rate, nothing is generated. `vkcube` normally runs far faster than any monitor, so the commands below cap it with `--framerate-limit 30` (on a 120 Hz output, gamescope releases every fourth frame). Change `-r 120` to your monitor's refresh rate. A game capped below the refresh rate (for example with `DXVK_FRAME_RATE=60`) works the same way.

30 fps on 120 Hz is below half the refresh rate, where the rule is multiplier 4. These test commands use 2 or the default anyway: in the default forward mode the multiplier does not change what you see (see [Names you will see](#names-you-will-see)).

**One GPU (copy and paste as is; the multiplier defaults to 2):**

```sh
GAMESCOPE_FRAMEGEN_HUD=2 gamescope \
  -r 120 --framerate-limit 30 \
  --experimental-framegen \
  --framegen-mode motion \
  --framegen-pipeline checked -- vkcube
```

**Hybrid laptop** (the launcher detects the GPUs; no exports needed; from a build tree, prefix it with `./scripts/env-gamescope-local.sh`, see [Install](#install)):

```sh
GAMESCOPE_FRAMEGEN_HUD=2 ./scripts/gamescope-hybrid.sh --steer-child -r 120 --framerate-limit 30 --experimental-framegen --framegen-mode motion --framegen-pipeline checked -- vkcube
```

**Two GPUs** (needs the `PRESENT_DEV` and `RENDER_DEV` variables from step 1):

```sh
GAMESCOPE_FRAMEGEN_HUD=2 gamescope \
  --prefer-vk-device "$PRESENT_DEV" \
  -r 120 --framerate-limit 30 \
  --experimental-framegen \
  --framegen-mode motion \
  --framegen-multiplier 2 \
  --framegen-pipeline checked -- \
  env MESA_VK_DEVICE_SELECT="$RENDER_DEV" vkcube
```

`--prefer-vk-device` picks the present GPU (the monitor GPU). `MESA_VK_DEVICE_SELECT` picks the GPU the game renders on. If you run from the build tree, see [Install](#install) option A for the binary path and the `env-gamescope-local.sh` prefix. On NVIDIA Optimus, the client may also need `__NV_PRIME_RENDER_OFFLOAD=1` and `__VK_LAYER_NV_optimus=NVIDIA_only`.

### Step 3. Confirm it works

The HUD is a small block of text, by default at the bottom centre of the gamescope window (`GAMESCOPE_FRAMEGEN_HUD_CORNER` moves it). `GAMESCOPE_FRAMEGEN_HUD=1` shows these rows: `gameslop`, `present`, `modes`, `rates` and `ladder`. `GAMESCOPE_FRAMEGEN_HUD=2` adds `pace` (how many frames met their deadline, and timing jitter) and `net` (the learned refiner, when active). The first tests use `2`.

| Look at this | Healthy | Not healthy: what to do |
|---|---|---|
| `rates` row | `source 30fps ... gen 90fps ... fill` near 100% (the generated rate fills the gap up to your refresh rate) | `gen 0fps`: the game already runs at the refresh rate. Cap it below (see the warning in step 2). |
| `present ... render ... buffers` | `buffers xGPU` with two GPUs: the game renders on one card and gamescope runs on the other | `buffers local` with two GPUs: the game renders on the present GPU. Fix `MESA_VK_DEVICE_SELECT`. (`buffers local` is expected with one GPU.) |
| repeat count in `rates` | Under about 5% of the display rate (under 6 fps at 120 Hz) | Many repeats: the present GPU is too slow for this setting. See [If something feels off](#if-something-feels-off). |

The `ladder` row: `rung 0/N full` is what you want. A higher rung means gamescope lowered quality because the generating GPU missed its deadline. It steps back up by itself (not in bidirectional mode or VRR hybrid, where it stays down). If it stays down, use `--framegen-pipeline checked` or `warp`, or a lower resolution. Add `--framegen-debug` to log `framegen: generated N frame(s)` lines in the terminal.

MESA_VK_DEVICE_SELECT only steers Vulkan games, including DXVK and Proton. For OpenGL games on a desktop with two cards, use `DRI_PRIME=...` (AMD or Intel, Mesa) or `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` (NVIDIA). The hybrid launcher in step 4 handles this on laptops; see [`scripts/README.md`](scripts/README.md).

### Step 4. Use it with a real game

**Laptop? Use the hybrid command below. Desktop with two cards? Use the Steam line.**

On a hybrid laptop, [`scripts/gamescope-hybrid.sh`](scripts/gamescope-hybrid.sh) detects the GPUs and steers only the game to the dGPU; see [`scripts/README.md`](scripts/README.md) and the hybrid command under [A good start for most games](#a-good-start-for-most-games).

On a desktop, set the game's Steam launch options. This is the template from above in one line:

```sh
GAMESCOPE_FRAMEGEN_HUD=1 gamescope --prefer-vk-device <PRESENT_ID> -W 2560 -H 1440 -r 120 -f --experimental-framegen --framegen-mode motion --framegen-multiplier 2 --framegen-pipeline checked -- env MESA_VK_DEVICE_SELECT='<RENDER_ID>!' %command%
```

- `--prefer-vk-device` is your monitor GPU (the present GPU). `MESA_VK_DEVICE_SELECT` is your game GPU, and the `!` makes it exclusive.
- Replace `<PRESENT_ID>` and `<RENDER_ID>` with the full `vendor:device` values from the doctor's export lines (for example `1002:73bf` and `10de:2684`), without the quotes. Keep the `!` and the quotes in the command.
- Keep `HUD=1` for play; use `2` only while testing.
- Match `-W`, `-H` and `-r` to your display.
- On a single GPU, drop `--prefer-vk-device ...` and the `env MESA_VK_DEVICE_SELECT=... ` part.
- If you did not install, start it through the environment script so the WSI layer is found: `/path/to/this/repo/scripts/env-gamescope-local.sh /path/to/this/repo/build/src/gamescope ...` (see [Install](#install)).

Expected: the game opens fullscreen with a small text HUD at the bottom centre. Frame generation only has an effect when the game runs below the display refresh rate (see [Choosing settings](#choosing-settings)). The command above is a safe start. For the recommended real-game command, which uses the `learned` pipeline, go to [A good start for most games](#a-good-start-for-most-games).

### Step 5 (optional). Native mode, for judging smoothness

Native mode is for testing, not for daily play. Gamescope takes over the screen from a text console, the way a console game runs, so your desktop cannot disturb the timing of frames reaching the display. Use it when you want to judge smoothness. Steam games are not supported this way yet.

First run `./scripts/framegen-doctor.sh` in your desktop terminal and write down the two IDs; variables exported there do not exist on the text console. Then switch to a free text console (for example Ctrl+Alt+F3), log in, and `cd` into your checkout. To return to your desktop later, press the key combination for your desktop's console, usually Ctrl+Alt+F1 or F2 depending on the distribution. Then run, with your own IDs:

```sh
GAMESCOPE_FRAMEGEN_HUD=2 RENDER_DEV='10de:1234!' PRESENT_DEV=1002:5678 \
  ./scripts/run-framegen-native.sh motion 2 600 2560x1440 120
```

The launcher arguments are `[mode] [multiplier] [windows] [WxH] [refresh] [-- client [args...]]`; replace the placeholder IDs. With no client after `--`, the launcher runs a `vkmark` desktop with 600 windows, which is GPU-bound and leaves free display slots to fill. The launcher has no frame cap option, so a client you pass after `--` (such as `vkcube`) must already run below the refresh rate, or nothing is generated. The launcher always uses the default `warp` pipeline; to judge `learned` natively, run your own `gamescope --backend drm ...` command from the console. The full procedure and the argument list are in the [native mode section of the User Guide](doc/framegen-howto.md#native-mode-from-a-text-console).

`run-framegen-native.sh` runs one command after `--`, so it does not cover a Steam game. `gamescope-hybrid.sh` forwards gamescope options such as `--backend` to gamescope, but a native Steam setup is not validated; see [Limits](doc/framegen-howto.md#8-limits-and-what-does-not-work-yet).

## Choosing settings

This section is a summary; the [User Guide](doc/framegen-howto.md) is the authoritative copy of settings advice and troubleshooting. There is no single best setting. What feels best depends on the game (how it looks and moves) and on your hardware (how fast the GPU that drives the display is). Expect to try two or three settings for a few minutes each; that is cheap, and dropping `--experimental-framegen` gives you plain gamescope again.

### Names you will see

| Name | Choices | Meaning |
|---|---|---|
| Mode (`--framegen-mode`) | `extrapolate`, `motion` | How generated frames are made. Neither holds back a real frame (only the forced composite, about 2 ms, is added). `motion` is the recommended everyday mode. |
| Pipeline (`--framegen-pipeline`) | `warp` < `checked` < `learned` < `predict` < `guided` | How careful the motion estimate is. Each includes the one before; further right means more GPU work per frame. `predict` felt choppy in our tests; `guided` is the costliest and a close second to `learned`. The automatic quality step-down walks this same order. |
| Multiplier (`--framegen-multiplier`) | 2, 3 or 4 (default 2) | In the default forward modes, on a GPU with a dedicated framegen queue (the startup log ends in `(dedicated queue)`; see [Reference](doc/framegen-reference.md#multipliers)), gamescope fills every free display slot whatever the number (it only sizes buffers). It adds no latency. Bidirectional mode makes at most N-1 frames per real pair, so it needs a higher number at low frame rates. Rule: 2 if the game runs at half the refresh rate or more, otherwise 4. Read the game's real frame rate from the HUD `rates` row (`source`). |
| Bidirectional (`GAMESCOPE_FRAMEGEN_BIDIR=1`) | on or off (off) | Waits for the next real frame and blends between two real frames. Smoothest, but adds about 20 to 45 ms more than the default modes. |
| Learned refiner (`GAMESCOPE_FRAMEGEN_NET_ONLINE=1`) | on or off | Trains a small model on your GPU while you play, to improve the `learned` pipeline. |

Details for every value are in [User Guide section 5](doc/framegen-howto.md#5-modes-and-presets).

### First, give it gaps to fill

Frame generation only adds frames when the game runs below the display refresh rate. If your game already reaches the refresh rate, nothing is generated: cap it below, for example at 60 FPS on a 120 Hz display, with the game's own frame limiter or, for DXVK and Proton games, `DXVK_FRAME_RATE=60`. Put `DXVK_FRAME_RATE=60` in front of `gamescope` in the launch options, like the other variables. Pick a cap the game holds even in busy scenes: frame generation smooths motion but cannot hide uneven game pacing, and in our tests a cap the game kept missing made motion less even, not more. A game that already runs well below the refresh rate needs no cap; in our tests Far Cry New Dawn at about 68 FPS on a 120 Hz display worked well uncapped.

### A good start for most games

In our tests, this felt best in a realistic modern game: motion mode (forward, not bidirectional), multiplier 2, and the `learned` pipeline with in-game learning on and one profile file per game, so it remembers what it learned. This is the recommended real-game command. Set it as a Steam launch option, in the same form as [Quick start](#quick-start) step 4:

```sh
GAMESCOPE_FRAMEGEN_NET_ONLINE=1 GAMESCOPE_FRAMEGEN_NET_PROFILE=$HOME/fg-mygame.bin GAMESCOPE_FRAMEGEN_HUD=1 gamescope --prefer-vk-device <PRESENT_ID> -W 2560 -H 1440 -r 120 -f --experimental-framegen --framegen-mode motion --framegen-multiplier 2 --framegen-pipeline learned -- env MESA_VK_DEVICE_SELECT='<RENDER_ID>!' %command%
```

As in step 4, match `-W`, `-H`, and `-r` to your display, replace the `<PRESENT_ID>` and `<RENDER_ID>` IDs as in step 4, and on a single GPU drop the `--prefer-vk-device` option and the `env MESA_VK_DEVICE_SELECT=...` part. Use a different profile file name for each game. The `learned` pipeline learns while you play, so give it a minute.

On a hybrid laptop, let [`scripts/gamescope-hybrid.sh`](scripts/gamescope-hybrid.sh) pick the GPUs: keep the variables in front, and put the same gamescope options after `--steer-child`, without `--prefer-vk-device` and `MESA_VK_DEVICE_SELECT` (use the absolute path of your checkout):

```sh
GAMESCOPE_FRAMEGEN_NET_ONLINE=1 GAMESCOPE_FRAMEGEN_NET_PROFILE=$HOME/fg-mygame.bin GAMESCOPE_FRAMEGEN_HUD=1 /path/to/gamescope/scripts/gamescope-hybrid.sh --steer-child -W 2560 -H 1600 -r 120 -f --experimental-framegen --framegen-mode motion --framegen-multiplier 2 --framegen-pipeline learned -- %command%
```

If your present GPU is a laptop iGPU, also put `GAMESCOPE_FRAMEGEN_NET_EVERY=2` in front of either command (it halves the training cost). On a desktop card, leave it out. From a build tree, start the hybrid command through `./scripts/env-gamescope-local.sh` (see [Install](#install)).

Quick rule: if your game's FPS times 2 is below your refresh rate (for example 40 FPS on a 120 Hz display), read [Low frame rates](#low-frame-rates-below-half-the-refresh-rate) first.

### Low frame rates (below half the refresh rate)

Quick rule: if your game's FPS x 2 < your refresh rate (for example 40 FPS on 120 Hz), use `--framegen-multiplier 4`. In forward mode the number does not change how many frames are made; bidirectional mode needs it.

What we saw in Far Cry New Dawn with HD textures at about 40 FPS, dipping into the 30s, on a 120 Hz display:

- Set `--framegen-multiplier 4`. It adds no latency. Forward mode fills every free slot at any multiplier, but bidirectional mode needs a multiplier of at least the refresh rate divided by the game's FPS: at 40 FPS with multiplier 2 it left almost a third of the refresh slots empty.
- `learned` and `guided` both worked. In one test, with the game-side low-latency option below turned on, `guided` felt at least as good as `learned`, so try both. `warp` showed many artifacts at this frame rate.
- Bidirectional mode gave the cleanest image, with almost no artifacts at multiplier 4, but it has a stutter in its motion. It holds back one whole game frame: about 55 ms in total from the game's frame to the screen (63 ms with the DXVK anti-lag setting below), against about 17 ms in forward mode, so roughly 38 ms extra. It suits slow or cinematic games, not shooters.
- Tip, not a default (the quick rule above does not depend on it): a low frame rate also adds delay inside the game, in its own queue of frames, which gamescope can neither measure nor remove. For DXVK and Proton games, putting `DXVK_CONFIG="dxgi.maxFrameLatency = 1; dxvk.latencySleep = True"` in front of `gamescope` shortens that queue. It cost no FPS in our test, so judge it by feel.

### How to try settings

- Change one thing at a time.
- Play the same scene for about a minute.
- Judge by feel (smoothness, sharpness, how direct the controls feel), not by the game's FPS counter, which keeps showing only real frames.
- If a change does not clearly help, go back.

### If something feels off

The most common cases. Artifacts, blur, a black screen and more are in [User Guide section 7](doc/framegen-howto.md#7-troubleshooting).

| If it ... | Try |
|---|---|
| feels laggy, or the controls feel heavy | Do not use bidirectional mode (`GAMESCOPE_FRAMEGEN_BIDIR=1`). It looks the cleanest and smoothest, but it adds a lot of latency. The multiplier adds none. At low frame rates, also shorten the game's own frame queue (see [Low frame rates](#low-frame-rates-below-half-the-refresh-rate)). |
| looks choppy although the frame rate is high | Use `--framegen-pipeline learned` with `GAMESCOPE_FRAMEGEN_NET_ONLINE=1`. In our tests, the `checked` and `predict` pipelines felt choppy. Also keep the game's frame rate steady by capping it at a rate it holds. |
| few generated frames (`gen` near 0) | Check that the startup log ends in `(dedicated queue)` (gamescope prints it to the terminal it was started from; for Steam launch options, start Steam from a terminal; see [Saving the log](doc/framegen-howto.md#the-metrics-log)); without it, at most one frame is made per gap. Also check that the game runs below the refresh rate (see [First, give it gaps to fill](#first-give-it-gaps-to-fill)). Details: [Reference](doc/framegen-reference.md#multipliers). |
| repeats many frames, or the HUD `ladder` row keeps stepping down | In bidirectional mode below half the refresh rate, raise the multiplier first (see [Low frame rates](#low-frame-rates-below-half-the-refresh-rate)). Otherwise the GPU that drives the display is too slow for this setting; gamescope already steps down on its own, but a cheaper fixed choice is steadier: use `checked` or `warp`, or lower the output resolution. On an integrated present GPU, also try `GAMESCOPE_FRAMEGEN_NET_EVERY=2`. The `ladder` row is shown at `GAMESCOPE_FRAMEGEN_HUD=1` and above. |

### Starting points by kind of game

The first three entries are what worked in our tests, all with one viewer. Treat all four as a starting point, not a rule.

| Game type (tested example) | Try first | Avoid or note |
|---|---|---|
| Realistic modern games (Far Cry Primal, Far Cry New Dawn, Far Cry 3 Blood Dragon; about 55 to 70 FPS) | `learned` with in-game learning felt best in all three. `guided` was a close second but costs the most GPU time. | `warp` was snappy but showed artifacts. `checked` and `predict` felt choppy. Bidirectional is the smoothest but added about 20 to 45 ms more than the default modes (more at lower frame rates): it ranked last in Primal and Blood Dragon and tied with `guided` in New Dawn. In New Dawn, frame generation off felt clearly worst. At lower frame rates, see [Low frame rates](#low-frame-rates-below-half-the-refresh-rate). |
| Older, sharp-edged shooters with clear contrast (Xonotic) | The sharp default; it was preferred over frame generation off. `GAMESCOPE_FRAMEGEN_FIELD_FINE=1` (finer motion detail; works with the `warp` and `checked` pipelines only) looked crisper there. | These tests used `checked`; `learned` was not tried on this game yet. Sharpness and low lag matter most. |
| Particle swarms and high-contrast space scenes (GravityMark) | The default motion detail. Expect some artifacts. | `GAMESCOPE_FRAMEGEN_FIELD_FINE=1` looked blocky there. This is the hardest case for any frame generation. |
| Competitive or very fast games (general advice, not tested) | Leave frame generation off if the game already reaches your refresh rate. | Avoid bidirectional mode. |

### Hardware and other options

Heavier pipelines need more from the GPU that drives the display, so weaker GPUs or higher resolutions may need `warp` or `checked`; per-GPU numbers are in the [User Guide](doc/framegen-howto.md).

- `GAMESCOPE_FRAMEGEN_BASE=1` for crisper compositor overlays and the gamescope cursor. It does not separate the game's own UI.
- `GAMESCOPE_FRAMEGEN_VRR_HYBRID=1` with `--adaptive-sync` for FreeSync and G-Sync displays. Normally frame generation disables adaptive sync while it is active; this option keeps it. It is experimental, not validated on a VRR panel, and needs a dedicated framegen queue. See [User Guide section e](doc/framegen-howto.md#e-vrr--freesync--g-sync-monitors).

What every mode, pipeline, and multiplier does is in [User Guide section 5](doc/framegen-howto.md#5-modes-and-presets). Every flag, environment variable, metrics field, and HUD row is in the [Configuration Reference](doc/framegen-reference.md).

## Known limits

- Frame generation does not lower input latency; generated frames add no input samples.
- Disocclusions (areas newly uncovered by motion), transparency, particles, fast edges, and UI can produce ghosting, shimmer, or smearing.
- Bidirectional mode is laggy by design, and single-GPU use competes with the game.
- The VRR hybrid path has not been validated on a VRR panel; native DRM/KMS, HDR, and mixed-vendor systems need broader testing.

The full list is in [User Guide section 8](doc/framegen-howto.md#8-limits-and-what-does-not-work-yet).

## Documentation

All documents (user guide, configuration reference, architecture, maintenance, testing, proposals, research, scripts) are indexed by goal in [`doc/README.md`](doc/README.md). Start with the [User Guide](doc/framegen-howto.md) to set things up and the [Configuration Reference](doc/framegen-reference.md) to look up a flag.

## Upstream gamescope

This fork retains upstream gamescope and its command-line interface. Valve's work
remains the foundation of this fork. The rest of this section is upstream's
documentation, lightly tidied.

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

# Run the output at 30 Hz, so a vsynced game runs at 30 FPS.
gamescope -r 30 -- %command%

# Render at 1080p in a fullscreen 3440x1440 pillarboxed window.
gamescope -w 1920 -h 1080 -W 3440 -H 1440 -b -- %command%
```

### Options

Run `gamescope --help` for the complete list.

- `-W`, `-H`: set the gamescope output resolution. Defaults to 1280x720.
- `-w`, `-h`: set the game resolution. Defaults to the output resolution.
- `-r`, `--nested-refresh`: refresh rate of the gamescope output (for frame generation, your monitor's Hz, for example 120). Games are paced to it. It is not a frame cap; to cap the game, use `--framerate-limit` or the game's own limiter.
- `-o`, `--nested-unfocused-refresh`: refresh rate used while the gamescope window is unfocused.
- `-F fsr`: use AMD FidelityFX Super Resolution 1.0 for upscaling.
- `-F nis`: use NVIDIA Image Scaling v1.0.3 for upscaling.
- `-S integer`: use integer scaling.
- `-S stretch`: stretch the game to fill the window.
- `-b`: create a borderless window.
- `-f`: create a fullscreen window.
- `--experimental-framegen`: enable the fork's compositor-side frame generation.
  See [Choosing settings](#choosing-settings) for the related options.

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

gamescope-gameslop is based on Valve's gamescope. See [`LICENSE`](LICENSE) for the BSD 2-Clause license, Valve copyright, and licenses for incorporated code.
