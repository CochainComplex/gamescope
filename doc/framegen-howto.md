# gamescope-gameslop User Guide

gamescope-gameslop is a fork of gamescope that adds compositor-side frame
generation: it inserts extra frames between the ones a game renders, so motion
looks smoother. It works with any game and any GPU because it only looks at the
finished picture. This guide explains how to set it up and run it.

> **Experimental.** Frame generation is specific to this fork and is not in
> upstream gamescope. Expect visual glitches and occasional crashes. When a
> workload is unstable, retry with a cheaper pipeline or a simpler mode.

## Quick start

Three steps. The build takes the longest; the other two take a few minutes. You
need a Linux desktop and a game that runs below your monitor's refresh rate (see
[section 1](#1-what-frame-generation-does)). On a first run, prove the setup in
a test window first: [README Quick start](../README.md#quick-start), steps 0 to 3.

1. **Build it** (the full steps are in the
   [top-level README](../README.md#install)). Running it from the build tree is
   fine; installing is optional (see README Install step 2). A distribution-packaged
   `gamescope` does not have `--experimental-framegen`. If you built it but did
   not install it, load the build first (see
   [Set up the terminal](#set-up-the-terminal-once-per-terminal)); or use
   `scripts/gamescope-hybrid.sh`, which finds `build/src/gamescope` of its own
   checkout (or the binary in `GAMESCOPE_BIN`).
2. **Run the doctor.** It checks your build and GPUs and, when a display-attached
   GPU is found, prints the two lines you need. You do not have to look up any
   IDs by hand:

   ```bash
   ./scripts/framegen-doctor.sh
   ```

   Copy the `export PRESENT_DEV=...` and `export RENDER_DEV=...!` lines it
   prints into your terminal. `PRESENT_DEV` is the card your monitor is plugged
   into. `RENDER_DEV` is the card that runs the game. (How to read the
   doctor's output: [Reading framegen-doctor.sh](#reading-framegen-doctorsh).)
   You should see: the verdict lines `can build: yes` and `can run ...: yes`,
   with exit status 0, or a `WARN` that names what is reduced.
3. **Run your game through gamescope.** This is the start we recommend for most
   games (the `learned` pipeline, with in-game learning on):

   ```bash
   GAMESCOPE_FRAMEGEN_HUD=1 \
   GAMESCOPE_FRAMEGEN_NET_ONLINE=1 \
   GAMESCOPE_FRAMEGEN_NET_PROFILE=$HOME/fg-mygame.bin \
   gamescope --prefer-vk-device "$PRESENT_DEV" -W 2560 -H 1440 -r 120 \
     --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
     --framegen-pipeline learned \
     -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME-HERE
   ```

   Replace the resolution and refresh rate with your monitor's, and
   `YOUR-GAME-HERE` with the command that starts your game. Only one GPU? Drop
   `--prefer-vk-device "$PRESENT_DEV"` and `env MESA_VK_DEVICE_SELECT=...` (see
   [Can I use just one card?](#can-i-use-just-one-card)). If your present GPU is
   a laptop's integrated GPU, also put `GAMESCOPE_FRAMEGEN_NET_EVERY=2` in front
   (it halves the training cost); with a separate present card, leave it out.
   Multiplier: 2 if the game runs at half the refresh rate or more, otherwise 4
   (see [Multiplier: which number?](#multiplier-which-number)).
   If `gen` in the HUD stays 0, the game is already at the refresh rate. Cap it
   below the refresh rate first (see
   [Frame-rate limit](#frame-rate-limit-optional)).
   Run from a normal desktop terminal, this opens gamescope as a window in your
   desktop ("nested"). That is fine for a first try; pacing can look uneven
   there (see [Nested vs. native](#8-limits-and-what-does-not-work-yet)).

   **The GPU IDs in examples elsewhere in this guide (`1002:5678`, `10de:1234`)
   are placeholders.** Always use your own from the doctor.

**What you should see.** Your game appears in a gamescope window or full screen,
with a small status block (the HUD) at the bottom. Its `rates` row shows `gen`
above zero, and `present ... buffers xGPU` on a two-GPU setup. Details:
[How to tell it works](#how-to-tell-it-works).

**Something is off?** Run `./scripts/framegen-doctor.sh` first, then go to
[Troubleshooting](#7-troubleshooting).

**Who this is for.** This guide is for people who want to play games with it.
The [Configuration Reference](framegen-reference.md) lists every flag, variable
and log field. Developers should read
[`framegen-architecture.md`](framegen-architecture.md),
[`framegen-maintenance.md`](framegen-maintenance.md) and
[`framegen-testing.md`](framegen-testing.md).

Contents:

1. [What frame generation does](#1-what-frame-generation-does)
2. [Hardware setup](#2-hardware-setup)
3. [First run](#3-first-run)
4. [Playing games](#4-playing-games)
5. [Modes and presets](#5-modes-and-presets)
6. [The HUD and the metrics log](#6-the-hud-and-the-metrics-log)
7. [Troubleshooting](#7-troubleshooting)
8. [Limits and what does not work yet](#8-limits-and-what-does-not-work-yet)

---

## 1. What frame generation does

Games draw a certain number of frames per second (fps). Frame generation
**invents extra frames in between the real ones** so the motion on screen looks
smoother. A game running at 40 fps can be shown at 80 or 120.

**Worked example.** Your monitor runs at 120 Hz, so it shows 120 pictures per
second. Your game renders 40. Without frame generation, each real frame is shown
three times in a row. With it, gamescope shows 40 real frames and fills the other
80 display slots with generated ones. Each picture is a little different from the
last, so motion looks smoother. This is the intended case: a game running below
the refresh rate. A game already at the refresh rate leaves no empty slots, so
nothing is generated.

**The name.** The fork is called gameslop (`gamescope-gameslop`); the program is
still named `gamescope`. gamescope is a *compositor*: the program that draws all
windows to your screen. While your game runs inside gamescope, gamescope draws
the screen for it, and that is the place where the extra frames are added.

What it does not do:

- **It does not reduce input lag.** The extra frames are for smoothness. They
  contain no new input, so a 20 fps game still samples your input 20 times a
  second. One mode, bidirectional, adds delay on purpose (see
  [section 5](#c-bidirectional-the-smoothest-motion)).
- **It does not change the game's own FPS counter.** The counter keeps showing
  the real render rate. The gamescope HUD and metrics report the real,
  generated and repeated frames that reach the display.

### What it costs

| Cost | Default modes (`extrapolate`, `motion`) | Bidirectional (opt-in) |
|---|---|---|
| Added input latency | About 2 ms per real frame (gamescope composites every real frame while frame generation is on). Generation itself adds at most about 0.5 ms. | About 20 to 45 ms more than the default modes (on top of their ~2 ms), more at lower frame rates |
| GPU work (on the present GPU) | Measured on a Radeon 890M at 2560x1600: about 1.1 to 3.8 ms per generated frame, depending on pipeline and game. A frame at 120 Hz lasts 8.3 ms. | Same order, plus a second motion field |
| Image quality | Occasional ghosting or fraying on fast edges | Cleanest image in our tests, but see the delay |
| What you need | Game fps below the display refresh rate | Same, and a game that tolerates the delay |

If the GPU cannot keep up, gamescope steps down to a cheaper pipeline by itself
(see [Automatic degradation](#automatic-degradation-the-ladder)).

"Added" means extra delay compared with the same game in gamescope without frame
generation. Totals from the game's frame to the screen: about 16 to 17 ms in
forward mode; in bidirectional mode about 46 ms (Far Cry Primal), and about
55 ms at about 40 fps (63 ms with the DXVK anti-lag tip).

### What artifacts look like

Frame generation guesses; sometimes it guesses wrong. This guide and the
troubleshooting table use these words:

- **Ghosting:** a faint trail behind a moving object.
- **Smearing or blur:** edges soften while they move.
- **Judder:** motion that steps unevenly instead of flowing.
- **Shimmer or fraying:** edges that crawl on fine detail such as particles or
  foliage.

**How it differs from the game's own FSR or DLSS frame generation.** Those get
the scene's real motion data from the game. gameslop sits outside the game and
sees only the finished pictures, so it has to work out the motion itself.
Expect results a little less clean than in-game DLSS or FSR, but it works on any
game and any GPU, and the game runs as if gameslop were not there.

### Words used in this guide

| Word | Meaning |
|---|---|
| Real frame | A frame the game rendered. |
| Generated frame | A frame gamescope invented between two real ones. |
| Render GPU | The card that runs the game. |
| Present GPU | The card your monitor is plugged into; it also generates frames. |
| Pipeline | How much work goes into each generated frame (`warp`, `checked`, `learned`, `predict`, `guided`). |
| Multiplier | `--framegen-multiplier` (2, 3 or 4). Forward mode fills every free display slot whatever the number; bidirectional mode makes at most one frame fewer than the number per real pair. |
| Nested | Gamescope running as a window inside your normal desktop. |
| Native | Gamescope started from a text console, owning the screen. |
| Net | A very small neural network that cleans up the motion and learns your game as you play. |
| Profile | The file where the net saves what it learned. |
| Ladder, rung | When the present GPU is too slow, gamescope steps down to a cheaper setting. The steps are the rungs of the ladder. |

---

## 2. Hardware setup

**Frame generation is designed as a two-GPU technique.** Making an extra frame
costs real GPU work. Doing it on the same card that runs the game takes power
from the game, and you would usually do better to let the game render more real
frames. So each card gets one job:

| Role | Which card | What it does |
|------|-----------|--------------|
| **Render GPU** | The stronger one (for example an NVIDIA RTX card or an AMD 7900 XT) | Runs the game. |
| **Present GPU** | The card your **monitor is plugged into** (for example a laptop's integrated AMD GPU, or an AMD 5700 XT) | Composites, generates the extra frames and sends the picture to the screen. |

The second card can be a laptop's built-in GPU or an old card you kept. It does
work that would otherwise sit idle.

### The most important rule

**Your monitor must be connected to the present GPU, not the render GPU.**
Gamescope always draws to the card the display cable is physically plugged
into. If the screen is on the wrong card, frame generation cannot run.

- **On a laptop**, the built-in screen is wired to one GPU, usually the
  integrated one. Use that GPU as the present device. The firmware wiring, not
  the vendor name, decides which GPU it is.
- **On a desktop with two cards**, plug the monitor into the weaker, present
  card (for example the 5700 XT), not the powerful one (the 7900 XT).

Not sure which card your screen is on? `./scripts/framegen-doctor.sh` marks it
with `display: ... (must be PRESENT)`. Without the doctor:

```bash
grep -l '^connected' /sys/class/drm/card*-*/status
```

The `cardN` in each path printed is a card with a connected monitor.

### Hybrid laptops

On a laptop with an integrated GPU and a discrete GPU, the integrated GPU
presents (it composites, generates and drives the screen) and the discrete GPU
renders the game. AMD and Intel integrated GPUs are detected the same way;
Intel as the present GPU is still untested (see
[Requirements](#requirements)). The helper
`scripts/gamescope-hybrid.sh` detects the pairing and steers only the game to
the discrete GPU; see [Hybrid laptops](#hybrid-laptops-launcher) in section 4.

### Find your card IDs

The doctor (`./scripts/framegen-doctor.sh`) prints your IDs for you. This is the
fallback if you want to look them up by hand. Each card has an ID of the form
`vendor:device`, four hexadecimal digits on each side of the colon. List yours:

```bash
lspci -nn | grep -iE "vga|3d|display"
```

Example output from a system with an NVIDIA render card and an AMD present card:

```
01:00.0 3D controller [10de:1234]      <- NVIDIA  (render card: runs the game)
05:00.0 Display controller [1002:5678] <- AMD     (present card: screen is here)
```

`10de` is NVIDIA, `1002` is AMD and `8086` is Intel. `vulkaninfo --summary`
lists the same devices as Vulkan sees them. The numbers above are examples;
yours will differ.

### Can I use just one card?

Yes, it works, but it is not recommended for demanding games. Generating frames
costs GPU work, and on one card that work comes out of the game's share. It is a
real win for an older game that has no FSR or DLSS frame generation of its own
and does not fully use your card. For a modern game that already maxes out your
GPU, use the game's own frame-generation setting instead.

A simple test: run the game once without gamescope and note its fps and GPU
usage. If the GPU is already close to 100%, frame generation will take frames
away from the game; skip it. If there is clear headroom, try it, and compare the
HUD's `source` fps with your baseline. If `source` drops a lot, the game lost
more than the generated frames gave back.

With one GPU, use the basic command without `--prefer-vk-device` and without
`MESA_VK_DEVICE_SELECT`:

```bash
gamescope -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
  --framegen-pipeline checked \
  -- YOUR-GAME-HERE
```

### Requirements

A checklist, with a way to check each item:

- **Two Vulkan GPUs that can share graphics buffers (dma-bufs)** is the
  recommended setup; one GPU also works. Check what Vulkan sees with
  `vulkaninfo --summary`. Validated: NVIDIA (proprietary) render to AMD (RADV)
  present. Intel, and NVIDIA as the present GPU, are untested.
- **NVIDIA rendering with the proprietary driver** needs its dma-buf support:
  driver 515.43.04 or newer (check with `nvidia-smi`) and
  `nvidia-drm.modeset=1` (`cat /sys/module/nvidia_drm/parameters/modeset`
  should print `Y`). If this is missing you may get a black screen, or the
  game's GPU stays at 0% and no real frame arrives.
- **Native (console-style) operation** needs Linux DRM/KMS. A window inside your
  existing desktop is enough for first tests.

---

## 3. First run

### Which setup am I?

- **Laptop (integrated + discrete GPU):** use the launcher in
  [section 4](#hybrid-laptops-launcher); it finds the cards for you.
- **Desktop with two GPUs:** follow this section with your own IDs.
- **One GPU:** use the one-GPU command in
  [Can I use just one card?](#can-i-use-just-one-card).

### Set up the terminal (once per terminal)

Do these in order.

**1. Load the build environment.** Open a terminal in this checkout (build it
first; see the [top-level README](../README.md#install)) and load the build
environment. This also puts the `gamescope` you built on your `PATH` for this
terminal. Replace the example path with yours:

```bash
cd /path/to/gamescope-gameslop
export GAMESCOPE_BUILD_DIR=build   # the directory you built into (the default)
source ./scripts/env-gamescope-local.sh
```

**2. Run the doctor and copy its export lines:**

```bash
./scripts/framegen-doctor.sh
```

It prints a preflight report for the build, Vulkan devices and GPU topology
(including the hybrid-laptop split) and suggests exact `export RENDER_DEV=...!`
and `export PRESENT_DEV=...` lines. Copy them as printed, including the `!` on
the render card:

```bash
export RENDER_DEV=10de:1234!   # example: strong card that runs the game (note the "!")
export PRESENT_DEV=1002:5678   # example: card your monitor is plugged into
```

The commands below and the supplied scripts reuse `$RENDER_DEV` and
`$PRESENT_DEV`. The `!` makes the choice exclusive: the game may use only that
card.

#### Reading framegen-doctor.sh

The doctor checks both the build and the runtime GPU roles. Each line starts
with a label:

| Label | Meaning |
|---|---|
| `PASS` | Fine. |
| `WARN` | Works, but reduced (for example no second GPU, so single-GPU only). |
| `FAIL` | Fix this first. The last `fix first:` line names the first step. |

Look at the bottom of the report: `can build`, `can run dual-GPU` and `can run
single-GPU` say what your machine can do. `display: ... (must be PRESENT)` marks
the card your screen is on. The script exits with status 0 when all is well,
1 with warnings and 2 on a hard failure.

In a healthy two-GPU report the verdict block shows `PASS  can build: yes`,
`PASS  can run dual-GPU: yes` and `PASS  can run single-GPU: yes`, then
`PASS  fix first: nothing required`. On a one-GPU
machine it shows `WARN  can run dual-GPU: no` and prints export lines that name
the same card twice: that is normal; use the one-GPU command and skip
`PRESENT_DEV` and `RENDER_DEV`. If no display-attached GPU is found, it prints
`FAIL  no display-attached GPU satisfies the PRESENT role` and no export lines;
fix that first.

### The basic command

Every command has the same shape:

```bash
GAMESCOPE_FRAMEGEN_NET_ONLINE=1 \
GAMESCOPE_FRAMEGEN_NET_PROFILE=$HOME/fg-mygame.bin \
gamescope --prefer-vk-device "$PRESENT_DEV" -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
  --framegen-pipeline learned \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME-HERE
```

Reading it left to right:

- `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` and `GAMESCOPE_FRAMEGEN_NET_PROFILE=...`:
  switch on in-game learning and name the file that keeps what was learned (see
  [f](#f-ai-refiner-in-game-learning)). Variables go in front of `gamescope`.
- `--prefer-vk-device "$PRESENT_DEV"`: composite and generate on the present card.
- `-W 2560 -H 1440 -r 120`: screen width, height and refresh rate. Match your
  monitor.
- `--experimental-framegen`: turns frame generation on. Required.
- `--framegen-mode motion`: the algorithm (see [section 5](#5-modes-and-presets)).
- `--framegen-multiplier 2`: use 2 when the game runs at half the refresh rate
  or more, otherwise 4. Forward mode fills every free display slot whatever the
  number; bidirectional mode needs it (see
  [Multiplier](#multiplier-which-number)).
- `--framegen-pipeline learned`: how much work goes into each generated frame.
  The program default is `warp`, the cheapest one; we recommend `learned`. For
  a quick test without a game, `checked` is cheaper.
- Everything after `--` is your game. `MESA_VK_DEVICE_SELECT="$RENDER_DEV"` pins
  the game to the strong card.

Replace `YOUR-GAME-HERE` with the command you normally use to start the game.

**Running it in a window on your desktop ("nested").** Fine for a first try:

```bash
GAMESCOPE_FRAMEGEN_HUD=2 gamescope --backend wayland \
  --prefer-vk-device "$PRESENT_DEV" -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
  --framegen-pipeline checked \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME-HERE
```

- `--backend wayland` runs gamescope as a window inside your Wayland desktop. It
  is optional: omit it, as in the README, and the default (`auto`) picks a
  backend that fits the desktop you run it in (also on X11).
- `-W 2560 -H 1440` is the window size; use a smaller size if it does not fit
  your screen.
- `-f` starts gamescope full screen. Add it if you want that.
- `--expose-wayland` lets games that speak Wayland directly connect. Games
  running through Steam Proton do not need it.
- `GAMESCOPE_FRAMEGEN_HUD=2` shows the HUD with the pacing rows.

**What nested looks like.** In a window on your desktop you may see uneven
pacing, and the `pace` row dropping low, even when everything is set up
correctly. Heavy scenes make the desktop compositor compete for the present
GPU. That is not a bug in frame generation. Judge real smoothness only in
[native mode](#native-mode-from-a-text-console).

### A ready-made test (no game needed)

If you have the GravityMark benchmark, this is a known-good test:

```bash
export GRAVITYMARK_DIR=/path/to/GravityMark/bin
(
  cd "$GRAVITYMARK_DIR"
  gamescope --prefer-vk-device "$PRESENT_DEV" -W 2560 -H 1440 -r 120 \
    --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
    --framegen-pipeline checked --framegen-debug \
    -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" \
    ./GravityMark.x64 -vk -temporal 0 -benchmark 0 -fps 1 \
      -asteroids 200000 -width 2560 -height 1440 -vsync 0
)
```

`--framegen-debug` makes gamescope print what it is doing. Debug lines are
sampled, so they appear only now and then.

`vkcube` or `vkmark` also work as simple test clients.

### How to tell it works

Start with the on-screen HUD: put `GAMESCOPE_FRAMEGEN_HUD=1` in front of
`gamescope`. The rows that matter look like this (illustrative; the numbers
will differ):

```text
present  <your present GPU>  render <your game GPU>  buffers xGPU   <- 2
rates    source 40fps[200] gen 78fps[390] repeat 2fps fill 98%      <- 1
```

1. **`rates`:** `source` (the game's real fps) is lower than the display rate,
   `gen` fills the gap and `fill` is near 100%. If you only look at two
   numbers: `gen` should be above 0 and `repeat` near 0.
2. **`present`:** shows your present GPU and `render ... buffers xGPU`, meaning
   frames arrive from the other card. `buffers local` means the game is
   rendering on the present GPU instead; fix `MESA_VK_DEVICE_SELECT`.
3. **Look at it.** Run the same scene twice, once with and once without
   `--experimental-framegen`. If it works, panning the camera looks smoother with
   it on. The game's own FPS counter will not change (see
   [section 1](#1-what-frame-generation-does)).
4. **Optional:** with `--framegen-debug` the terminal prints
   `framegen: generated N frame(s) ...`. These lines are sampled (about every
   60th), so a few lines per second at most is normal.

Not working? Go to [Troubleshooting](#7-troubleshooting).

---

## 4. Playing games

### Hybrid laptops (launcher)

`scripts/gamescope-hybrid.sh` runs gamescope on the GPU that drives the display
and steers only games to the other GPU, so you do not have to look up IDs. It
detects the pairing automatically (AMD or Intel integrated GPU with an NVIDIA or
AMD discrete GPU, or a desktop with two cards given explicit IDs). It runs the
gamescope in `GAMESCOPE_BIN` if set, else this checkout's `build/src/gamescope`
if it exists, else `gamescope` from your `PATH`. Below,
`/path/to/gamescope` means the absolute path of your checkout.

**Which mode?**

- Run one game from a terminal: use `--steer-child`.
- Run Steam games: start Steam through the launcher once (host mode), then set
  each game to `--child`.
- Not sure what it would do on your machine: run `--report` first (see below).

One game from a terminal (starts gamescope and runs the game on the other GPU):

```bash
/path/to/gamescope/scripts/gamescope-hybrid.sh --steer-child \
  --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
  --framegen-pipeline checked \
  -- YOUR-GAME-HERE
```

**Steam, in two steps.** Two terms first. *Host mode* means one gamescope runs
all of Steam; the launcher starts it when you give it no `--child` and no
`--steer-child`. `--child` means "steer only this game to the game GPU". It does
not start gamescope, so a gamescope host must already be running.

From now on, start Steam this way: a Steam you start separately from the desktop
does not use this gamescope. Never use `--steer-child` on the Steam client
itself: it would render the whole Steam client on the game GPU and keep it awake
(the launcher warns about this). Using it on a single game's `%command%` is fine.

1. Start the host from a terminal (or your session script):

   ```bash
   /path/to/gamescope/scripts/gamescope-hybrid.sh \
     --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
     -- steam -gamepadui
   ```

   You should see Steam's Big Picture interface open inside gamescope.

2. In each game's Launch Options, use an absolute path:

   ```
   /path/to/gamescope/scripts/gamescope-hybrid.sh --child -- %command%
   ```

   When you start the game, it appears inside the same gamescope, and the HUD
   (if you set `GAMESCOPE_FRAMEGEN_HUD=1` on the host) shows `buffers xGPU`.

| You want to run | Mode | Command shape |
|---|---|---|
| One game from a terminal | `--steer-child` | `gamescope-hybrid.sh --steer-child [gamescope options] -- game` |
| Steam | host (the default) for Steam, then `--child` per game | `gamescope-hybrid.sh [gamescope options] -- steam -gamepadui`, and `gamescope-hybrid.sh --child -- %command%` as each game's Launch Options |
| Lutris, Heroic | `--steer-child` as the per-game prefix (tested), or host + `--child` (not validated) | `gamescope-hybrid.sh --steer-child [gamescope options] --` |

Run `scripts/gamescope-hybrid.sh --report -- vkcube` for a dry run that prints
the detection, the decision and the command it would run. The full description
(host mode, `--steer-child`, Flatpak Steam) is in
[`scripts/README.md`](../scripts/README.md).

### Steam (without the launcher)

**Which Steam recipe?**

- **Hybrid laptop:** host + `--child` (above).
- **Desktop with two GPUs:** plain `gamescope` in the game's launch options (this
  section).
- **One GPU:** the same as the desktop case, without the GPU flags.

This form calls plain `gamescope`, so it needs a gamescope on your `PATH` that
has `--experimental-framegen`: an installed build, or a terminal where you ran
`source ./scripts/env-gamescope-local.sh` (Steam started from a desktop icon does
not inherit that). If your build is not installed, use the launcher above, which
resolves the binary itself. Put the gamescope command in the game's launch
options (Properties, then Launch Options). Everything before `--` configures
gamescope; `%command%` is the game:

```
GAMESCOPE_FRAMEGEN_HUD=1 gamescope --prefer-vk-device 1002:5678 -W 2560 -H 1440 -r 120 -f --experimental-framegen --framegen-mode motion --framegen-multiplier 2 --framegen-pipeline checked -- env MESA_VK_DEVICE_SELECT='10de:1234!' %command%
```

Use your own IDs (the ones shown are placeholders), resolution and refresh rate.
Alternatively, start the whole session from a terminal with
`gamescope ... -- steam steam://rungameid/<id>`. Environment variables go in
front of `gamescope` in launch options, as in the example.

If the game does not start at all, see
[Steam game does not start](#steam-game-does-not-start) in Troubleshooting.

### Lutris, Heroic and other launchers

The only setup tested with a launcher is the per-game one: in Lutris, set the
game's Command prefix (in Heroic, the Wrapper) to the line below. Each game then
gets its own gamescope, and the launcher itself runs outside it:

```text
/path/to/gamescope/scripts/gamescope-hybrid.sh --steer-child --experimental-framegen --framegen-mode motion --framegen-multiplier 2 --
```

Use the absolute path of your checkout: launchers run with their own working
directory and `PATH`. Starting the launcher itself as the host, with
`gamescope-hybrid.sh [options] -- lutris`, and setting `--child --` as the
per-game prefix, follows the Steam recipe but is not validated. More in
[`scripts/README.md`](../scripts/README.md).

Tested example: Far Cry Primal through Ubisoft Connect in Lutris on a hybrid
laptop, with the game on the NVIDIA GPU and gamescope presenting on the iGPU
(through `--steer-child`). Forward (non-bidirectional) x2 with the learned
pipeline was the best-rated configuration; bidirectional measured the smoothest but felt worst because of
its added latency. Cap the game below the output rate (for example
`DXVK_FRAME_RATE=60` at 120 Hz), or there are no gaps to fill. Measurements are
in [framegen-testing.md](framegen-testing.md#real-game-ab-far-cry-primal).

### Native mode (from a text console)

Starting gamescope from a text console takes over the screen like a game console
and gives the smoothest result, because the desktop compositor is no longer in
the way. Native mode is also the real test for scanout pacing, input latency and
VRR behavior.

1. In your desktop terminal, run `./scripts/framegen-doctor.sh` and write down
   your two IDs. Variables you exported in the desktop terminal do not exist on
   the text console.
2. Switch to a free text console (for example Ctrl+Alt+F3), log in, and `cd`
   into your checkout of this repository (the commands below use `./scripts/`).
3. Run the supplied launcher, with your own IDs:

```bash
RENDER_DEV='10de:1234!' PRESENT_DEV=1002:5678 \
  ./scripts/run-framegen-native.sh [mode] [multiplier] [windows] [WxH] [refresh] [-- client [args...]]
```

Defaults: mode `motion`, multiplier `2`, a vkmark desktop workload with 600
windows, 2560x1440 at 120 Hz. Positional arguments must come before `--`, so to
run your own program give all five first. For example:

```bash
RENDER_DEV='10de:1234!' PRESENT_DEV=1002:5678 \
  ./scripts/run-framegen-native.sh motion 2 600 2560x1440 120 -- vkcube
```

The launcher always uses the default `warp` pipeline (plus `--framegen-debug`
and `--framegen-strength 0.5`), so you cannot judge the recommended `learned`
setup with it. For that, run your own `gamescope --backend drm ...` command
from the text console. `scripts/watch-framegen.sh` behaves the same way and
defaults to multiplier 3, 1000 windows and 3840x2160. The launcher writes its
log to `/tmp/framegen-native.log` (override with `LOGFILE`).

`windows` only matters for the default vkmark workload; any value works with
your own program. The mode can be `motion`, `extrapolate` or `blend`. Running a
Steam game natively is not covered by this launcher, and no native Steam recipe
has been validated. `gamescope-hybrid.sh` forwards `--backend` to gamescope, so a
TTY run of host mode might work, but treat it as untested. The script refuses to
run while a graphical session is active (`DISPLAY` or `WAYLAND_DISPLAY` set)
unless `FORCE=1` is given. `scripts/watch-framegen.sh` shows frame generation in
a nested desktop window instead.

**What you should see.** With the default workload, the vkmark desktop of 600
windows fills the screen. Add `GAMESCOPE_FRAMEGEN_HUD=2` in front of the command
to get the HUD. Quit with Ctrl+C. To get back to your desktop, switch to its
console with Ctrl+Alt+F1 or F2 (which one depends on your system). If the screen
stays black, press Ctrl+C blind, run the doctor, and check that `PRESENT_DEV` is
the card your monitor is on. With the HUD on, `present` shows `buffers xGPU`.

---

## 5. Modes and presets

Letters in parentheses, such as (f), refer to the sub-sections a) to g) below.

**Defaults.** If you give only `--experimental-framegen`, you get mode
`extrapolate` and pipeline `warp`. `--framegen-pipeline` is ignored unless
`--framegen-mode motion` is also given, so setting only the pipeline changes
nothing.

**There is no single best setting.** What feels best depends on the game (how
it looks and moves) and on how fast the GPU that drives the display is, so try
two or three settings for about a minute each. The steps for a fair try-out are
in [Choosing settings](../README.md#choosing-settings) in the README.

**Recommended start: row 1 of the first table** (motion, `learned`, in-game
learning on). The snippets in this section show `checked` because it is cheap to
test. For real play, swap in `--framegen-pipeline learned` plus the two `NET`
variables from [f](#f-ai-refiner-in-game-learning). Put variables in front of
`gamescope` and flags before `--`; each snippet is the
[basic command](#the-basic-command) with the flags or variables shown added or
swapped in.

### Which one should I use?

#### Pick one base setup

| You want | Use | Add to the basic command |
|-----------|-----|-----|
| **Recommended:** a good start for most games | **Motion `learned`** (a small neural net that learns your game while you play), x2, forward mode (not bidirectional); see (f) | `--framegen-pipeline learned`, `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` and `GAMESCOPE_FRAMEGEN_NET_PROFILE=$HOME/fg-<game>.bin` (one file per game) |
| A cheap first test | **Motion `checked`**, x2 (b) | `--framegen-pipeline checked` |
| Weak display/framegen GPU | **Motion `warp`**, x2 | `--framegen-pipeline warp` |
| Competitive or fast shooter (cheapest on the GPU) | **Simple** (a), not bidirectional | `--framegen-mode extrapolate` |
| The most complete pipeline, if the GPU has room | **Motion `guided`** with the net (f); a close second to `learned` in our tests, at the highest GPU cost | `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` and `--framegen-pipeline guided` |
| The smoothest motion, if about 20 to 45 ms more than the default modes is fine (more at low frame rates) | **Bidirectional** (c), experimental; suits some games and players, not others | `GAMESCOPE_FRAMEGEN_BIDIR=1`, and `--framegen-multiplier 4` if the game runs below half the refresh rate (see Multiplier) |
| The smoothest motion with in-game learning, if the same delay is fine | **Bidirectional + net** (c), experimental; suits some games and players, not others | `GAMESCOPE_FRAMEGEN_BIDIR=1`, `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` and `--framegen-pipeline learned`, plus `--framegen-multiplier 4` if the game runs below half the refresh rate (see Multiplier) |

The first row is what felt best in our real-game tests (three Far Cry games at
about 55 to 70 FPS, one viewer, one hybrid laptop). Treat it as a starting
point, not a rule.

#### Optional add-ons

| You want | Use | Add |
|-----------|-----|-----|
| Crisp compositor overlays and cursor | **Base-layer** (d) | `GAMESCOPE_FRAMEGEN_BASE=1` |
| You own a FreeSync or G-Sync monitor | **VRR** (e), experimental | `GAMESCOPE_FRAMEGEN_VRR_HYBRID=1` and `--adaptive-sync` |

#### Multiplier: which number?

Rule: use 2 when the game runs at half the refresh rate or more, otherwise 4.
If unsure, use 4. It adds no latency.

Why the number matters little in forward mode: in forward motion mode with a dedicated framegen
queue (the usual case; the startup log says `(dedicated queue)`), gamescope
fills every free display slot until the next real frame by itself. The
multiplier then only sizes buffers, and 2, 3 and 4 give the same result.

| Mode | What the number does |
|---|---|
| Forward (the default and recommended) | No effect on how many frames are made, and no extra GPU work. |
| Bidirectional | It makes at most the multiplier minus 1 frames per pair of real frames, so use at least refresh rate divided by the game's fps (40 fps on 120 Hz: 3 or 4). At about 40 FPS on 120 Hz, x2 filled 71% of the display slots, x3 98% and x4 100%, at the same latency. |

Details are in the [Configuration Reference](framegen-reference.md#multipliers).

#### Low frame rates

Below half the refresh rate (for example 40 FPS on 120 Hz), what we saw in Far
Cry New Dawn with HD textures at about 40 FPS on a 120 Hz display:

- Use multiplier 4 (bidirectional mode needs it; forward mode does not care).
- `learned` and `guided` both worked well. `warp` showed many artifacts.
- Bidirectional mode gave the cleanest image but its motion had a stutter, and
  it holds back one whole game frame (see the numbers in
  [c)](#c-bidirectional-the-smoothest-motion)).
- Tip, not a default: a low frame rate also adds delay in the game's own frame
  queue, which gamescope cannot measure or remove. For DXVK and Proton games,
  `DXVK_CONFIG="dxgi.maxFrameLatency = 1; dxvk.latencySleep = True"` in front of
  `gamescope` shortens it. It cost no FPS in our test; judge it by feel.

The README has the same summary in
[Low frame rates](../README.md#low-frame-rates-below-half-the-refresh-rate).

### a) Simple (extrapolate, the default): cheapest, most compatible

```bash
gamescope --prefer-vk-device "$PRESENT_DEV" -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode extrapolate --framegen-multiplier 2 \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME-HERE
```

`extrapolate` is the default, so leaving out `--framegen-mode` gives the same
result. (A third mode, `blend`, exists but is not recommended.) It predicts the
next frame. It has the lowest GPU cost and works everywhere. The added delay is
the same as `motion` (about 2 ms), but fast motion can look a little juddery.

### b) Motion (forward): the base for everything below

```bash
gamescope --prefer-vk-device "$PRESENT_DEV" -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode motion --framegen-pipeline checked \
  --framegen-multiplier 2 \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME-HERE
```

Motion mode tracks how things move, for sharper and cleaner extra frames. It
holds no real frame back. The `--framegen-pipeline` flag sets how much work goes
into each generated frame. The snippet uses the cheap `checked` pipeline; for
real games, `learned` with in-game learning felt best in our tests (see
[f](#f-ai-refiner-in-game-learning)).

Choose the cost explicitly for your display/framegen GPU:

| Pipeline | What it does (plain words) | Cost | Best fit |
|---------|--------------|------|----------|
| `warp` | moves pixels along the detected motion | low | the cheapest, for older or bandwidth-limited GPUs; snappy, but showed artifacts in a real game |
| `checked` | also throws away motion it does not trust (cleaner edges, can feel choppy) | medium | cheaper than `learned`, but felt choppy in a real game |
| `learned` | also learns from your game which motion to trust; with the net on, it gets better as you play | medium | the best real-game start, with in-game learning on (f) |
| `predict` | also predicts accelerating motion | medium | felt choppy in a real game |
| `guided` | also fills in background uncovered by moving objects, using colour and three real frames | high | a close second to `learned`, at the highest GPU cost |

The real-game notes in the last column come from three Far Cry games (Primal, New
Dawn, Blood Dragon).

The pipelines are cumulative, in the order of the table: each one does
everything the ones above it do, and more. More is not automatically better.
Layers that reject or change the motion can feel choppy: `checked` throws away
motion that fails its checks, and `predict` adds acceleration. With in-game
learning, `learned` takes back rejected motion where it predicts the real frames
better than showing no motion, which helped both smoothness and image quality.

**Will my GPU cope?** On a Radeon 890M at 2560x1600, one generated frame took
about 1.1 to 1.3 ms with `warp`, 1.2 to 3.1 ms with `learned` and 1.2 to 3.8 ms
with `guided`, depending on the game (three Far Cry games); a frame at 120 Hz
lasts 8.3 ms. Weaker GPUs or higher resolutions may need `warp` or `checked`. The
simple rule: if the HUD `ladder` row leaves `full`, go one pipeline cheaper.

#### Automatic degradation (the ladder)

If measured GPU time misses the display deadline,
gamescope walks down these pipelines on its own, and steps back up one adjacent
level after sustained measured headroom. What you notice: the HUD `ladder` row
shows a rung above 0 (for example `rung 1/4`) and the image may look a little
softer or simpler. You do not need to act. One isolated deadline miss only skips
a slot; repeated misses, or a cost that cannot fit the slot, degrade. Loading
hitches do not degrade. `GAMESCOPE_FRAMEGEN_RECOVER=0` turns recovery off.
Recovery does not run in bidirectional mode or VRR hybrid: there a stepped-down
rung stays down, so if it bothers you, pick a cheaper pipeline yourself.

**Status.** Experimental; validated in nested and dual-GPU test runs. Measured
results from real-game runs are in [framegen-testing.md](framegen-testing.md).

### c) Bidirectional: the smoothest motion

**In short:** bidirectional mode waits for the next real frame before showing
the current one, so everything on screen is one game frame late (about 20 to
45 ms in our tests, more at low frame rates). Use it for slow, cinematic or
single-player games only. Avoid it for fast or competitive games.

```bash
# Use --framegen-multiplier 4 instead if the game runs below half the refresh rate.
GAMESCOPE_FRAMEGEN_BIDIR=1 gamescope --prefer-vk-device "$PRESENT_DEV" \
  -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
  --framegen-pipeline checked \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME-HERE
```

This blends between two real frames instead of guessing ahead. It gives the
smoothest result and handles see-through effects (smoke, glass) best.

**Trade-off: it adds lag.** Gamescope shows each real frame later than live,
within a bound it tracks itself (the "bidirectional delay controller"; the rules
are in the
[architecture doc](framegen-architecture.md#bidirectional-delay-controller)).
The current delay is reported as `bidir_delay_ms` in the metrics log. Below half
the refresh rate, this mode needs a higher multiplier (see
[Multiplier](#multiplier-which-number)).

What we measured, in real games:

Totals are from the game's frame to the screen.

| Game | Total, forward | Total, bidirectional | Extra for bidirectional | Verdict |
|---|---|---|---|---|
| Far Cry Primal | about 16 ms | about 46 ms | about 30 ms | Felt worst, despite the steadiest frame pacing |
| Far Cry New Dawn | not measured | not measured | about 21 ms | Tolerated; tied with `guided` for second place |
| Far Cry New Dawn, HD textures, about 40 FPS, multiplier 4 | about 17 ms | about 55 ms (63 ms with DXVK's low-latency option) | about 38 ms | Cleanest image of all modes, but a sluggish stutter inside the picture, still being investigated |

Advanced notes (details in the [reference](framegen-reference.md#values)):

- The extra `guided`-pipeline passes do not run in this mode; it uses two
  checked motion fields instead.
- `GAMESCOPE_FRAMEGEN_BIDIR_OCCLUSION` and `GAMESCOPE_FRAMEGEN_BIDIR_TRACE` are
  separate experimental edge and reconstruction settings, off by default.
- Bidirectional mode and VRR hybrid cannot run together. On a dedicated framegen
  queue, requesting both keeps VRR hybrid and ignores bidirectional mode; on a
  shared queue VRR hybrid is ignored (logged) and bidirectional mode runs.

**Status.** Experimental and opt-in. Not validated: native DRM/KMS, HDR output
and compositor overlays. Test details are in
[framegen-testing.md](framegen-testing.md).

### d) Base-layer: keeps compositor overlays and the cursor crisp

```bash
GAMESCOPE_FRAMEGEN_BASE=1 gamescope --prefer-vk-device "$PRESENT_DEV" \
  -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
  --framegen-pipeline checked \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME-HERE
```

Generates on the game layer before gamescope composites its own overlays and
cursor on top, so those stay crisp instead of smearing. It cannot help with a
game's own HUD or menus: the game draws them into the frame gamescope receives,
so they are generated along with everything else. With
`GAMESCOPE_FRAMEGEN_BIDIR=1`, scenes that take base-layer generation use causal
generation and the other scenes use bidirectional interpolation. The terminal
logs the first decision; later scene changes are logged with `--framegen-debug`.

### e) VRR / FreeSync / G-Sync monitors

```bash
GAMESCOPE_FRAMEGEN_VRR_HYBRID=1 gamescope --prefer-vk-device "$PRESENT_DEV" \
  --adaptive-sync -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
  --framegen-pipeline checked \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME-HERE
```

Keeps your monitor's adaptive sync working together with frame generation. It
does something only on a real VRR monitor that is actually in VRR mode. On an
ordinary monitor (and most laptop screens) it quietly does nothing and falls
back to normal. It needs a dedicated framegen queue; on a shared queue it is
ignored and the terminal says so. It uses the causal timeline and cannot be
combined with bidirectional interpolation. It has not been validated on a VRR
panel (the main test machine has no VRR display).

### f) AI refiner (in-game learning)

```bash
GAMESCOPE_FRAMEGEN_NET_ONLINE=1 \
GAMESCOPE_FRAMEGEN_NET_PROFILE=$HOME/fg-mygame.bin \
gamescope --prefer-vk-device "$PRESENT_DEV" \
  -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode motion --framegen-pipeline learned \
  --framegen-multiplier 2 \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME-HERE
```

This is the start we suggest for most games: forward motion mode, multiplier 2,
the `learned` pipeline, in-game learning on and one profile file per game. In
our real-game tests (three Far Cry games: Primal, New Dawn, Blood Dragon) it felt best in all three.

A tiny neural net cleans up the motion and learns your game as you play. It
works with the normal forward mode, which does not hold real frames back. At
`guided` it also learns to correct small shadow and reflection colour changes
(`GAMESCOPE_FRAMEGEN_SHADING=0` turns that part off). In bidirectional mode the
refiner is deliberately conservative and can only lower confidence toward a
plain crossfade.

- `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` turns the learning on: the net trains while
  you play, so give it a minute. **If you forget it, nothing warns you:** the
  `learned` pipeline then runs without the net, as `checked` plus
  self-calibration only (unless you also give a `GAMESCOPE_FRAMEGEN_NET` weights
  file). The HUD `modes` row shows `net:online` while it is learning, and
  `net:off` when it is not.
- `GAMESCOPE_FRAMEGEN_NET_PROFILE=$HOME/fg-<gamename>.bin` is the file where the
  net keeps what it learned. Gamescope loads it at the start if it exists, so
  the net does not begin from scratch every session. With
  `GAMESCOPE_FRAMEGEN_HUD=2` the HUD also shows whether a profile was loaded.
- `GAMESCOPE_FRAMEGEN_NET_EVERY=2` trains every second real frame. Rule: if your
  **present GPU is a laptop's integrated GPU**, set it (pacing stays equal and
  it played best in the real-game test). If your present GPU is **a separate
  graphics card**, leave it unset (the default is `1`).
- It needs `--framegen-mode motion` and `--framegen-pipeline learned` or above.

**Profile files.**

- You do not need to create the file. Gamescope writes it as it learns (a
  checkpoint every 1024 training steps) and again when the session ends. If it
  does not exist yet, the net starts from a neutral prior.
- It is small: the weights are about 18.6 kB.
- To reset learning, delete the file.
- Use one file per game, and per resolution to be safe: what the net learns
  depends on how the game looks and moves.
- The learning is deliberately temporary. The weights are pulled back toward the
  safe starting point continually, so the net only stays away from it as long as
  the current scene keeps earning it (a memory of roughly a minute).

If anything looks worse, compare by leaving out
`GAMESCOPE_FRAMEGEN_NET_ONLINE=1`.

Power users can also train the net offline on a captured scene and measure it
before use (`scripts/framegen-net-train.py` then `scripts/framegen-net-eval.py`);
see [Offline refiner and colour evaluators](framegen-testing.md#offline-refiner-and-colour-evaluators); the design is in [architecture 3.5](framegen-architecture.md#35-learned-field-refinement-stage-c-opt-in-via-gamescope_framegen_netblob).

### g) Fine-tuning switches

Three switches change details of how generated frames look and when they are
made. The defaults are the safe choice; change one only to compare. Put the
variable in front of `gamescope`, like the others.

- `GAMESCOPE_FRAMEGEN_SHARP_FALLBACK` is on by default. Where the motion is hard
  to tell, a generated frame moves the real frame only as far as the motion can
  be trusted. The older crossfade blended two guesses instead and showed double
  edges and blur where the confidence changed. The sharp version shows only real
  pixels, so HDR is unaffected, and it costs no more GPU time than the
  crossfade (about 10% less at 1440p and 2160p on a Radeon 890M). It applies to
  the `warp`, `checked` and `learned` pipelines; `guided` and bidirectional mode
  use their own shaders and are not affected. In blind tests on `checked`, it
  beat frame generation off on Xonotic, where the crossfade lost to off, and it
  was clearly smoother than off on GravityMark. Set
  `GAMESCOPE_FRAMEGEN_SHARP_FALLBACK=0` only to compare with the older, softer
  crossfade. Known limit: it can fray the edges of dense particle swarms. We
  have not tested it blind on `learned` yet.
- `GAMESCOPE_FRAMEGEN_FIELD_FINE=1` is off by default. It follows the motion in
  finer detail, with four times as many motion samples, so it costs more GPU
  time. It works with the `warp` and `checked` pipelines only; the others ignore
  it. In blind tests on `checked`, it looked crisper on Xonotic (an older,
  sharp-edged shooter) in both rounds and tied on the vkmark desktop, but it
  looked blocky on GravityMark's particle swarm in both rounds. Try it on a
  sharp-edged game, and leave it off for particle swarms.
- `GAMESCOPE_FRAMEGEN_BATCH_REPAINT=1` is off by default. It makes gamescope
  repaint after each batch of generated frames, instead of only waking its
  compositor. On a game limited by the GPU (GravityMark with a million
  asteroids, 2048x1280 at 120 Hz), real frames rose from 34 to 44 FPS with
  steadier frame timing. But one real frame in twenty took longer than about 25
  ms to reach the screen, up from about 11 ms, and it felt laggier in a blind
  test. That is why it is off. It works in forward mode only (not bidirectional,
  not VRR hybrid).

### Frame-rate limit (optional)

Frame generation only fills gaps, so a game that already reaches the refresh
rate needs a cap (see [First, give it gaps to fill](../README.md#first-give-it-gaps-to-fill)
in the README). The game's own limiter or `DXVK_FRAME_RATE` is usually the
simplest way to cap it. Pick a cap the game holds even in busy scenes; a cap it
keeps missing made motion less even in our tests.

gamescope also has its own cap. `--framerate-limit 30` is a divisor of the
refresh rate and rounds down, so at 120 Hz it releases the game's frames on
every fourth display refresh (30 fps). At 120 Hz and 30 fps each real frame is
followed by three generated frames, whatever the multiplier in forward mode
(on a GPU with a dedicated framegen queue; otherwise at most one per gap).
That gives even pacing, but it deliberately caps the game and adds input
latency if the game could otherwise run faster. `0` turns it off.

Benchmark presets (such as the GravityMark x4 baseline) and the scheduler
internals are developer material; see [framegen-testing.md](framegen-testing.md)
and [framegen-architecture.md](framegen-architecture.md).

---

## 6. The HUD and the metrics log

### The on-screen HUD

`GAMESCOPE_FRAMEGEN_HUD=1` draws a compact status block on screen; `=2` adds
pacing and learning rows. `GAMESCOPE_FRAMEGEN_HUD_SCALE=1..6` sizes it (`0`, the default, picks 2 to 4 automatically from the output height) and
`GAMESCOPE_FRAMEGEN_HUD_CORNER=bc|tl|tr|bl|br` places it (default bottom-center,
`bc`).

A full HUD at level 2 looks like this (illustrative; your numbers and GPU names
will differ):

```text
gameslop <version>   motion x2 . learned . 120Hz fixed
present  <present GPU>  render <game GPU>  buffers xGPU
modes    bidir:off  base:off  net:online  adapt:on
rates    source 40fps[200] gen 78fps[390] repeat 2fps fill 98%
ladder   learned/motion rung 0/4  full
pace     hit 99%  ...  jitter +/-0.7ms  ...
net      online . 3.2k steps . profile loaded
```

**If you only look at one number:** `repeat` (should be near 0) and `gen`
(should be above 0). In the example, 40 real + 78 generated + 2 repeated = 120
display slots, so `fill` is 98%.

#### The four rows that matter

| Row | Looks healthy | Worrying, and what to do |
|---|---|---|
| `present`/`render`/`buffers` | `buffers xGPU` on a two-GPU setup: the setup is correct. | `buffers local`: the game runs on the present GPU. Check `MESA_VK_DEVICE_SELECT` (see [wrong GPU](#wrong-gpu-in-the-hud)). |
| `rates` | `gen` is about the display rate minus `source`, and `fill` is high. | Low `fill`: either the game is too close to the refresh rate (nothing to fill) or the present GPU is too slow. See [Lots of little stutters](#lots-of-little-stutters). |
| `ladder` | `rung 0/N` and `full`: gamescope runs the setting you asked for. | A rung above 0, or `hold`: gamescope lowered quality because the present GPU missed its deadline. Use a cheaper pipeline (`checked` or `warp`), or a lower resolution. |
| `pace` (level 2) | `hit` at about 95% or more, and `jitter` under about 1 ms: smooth. | `hit` under about 90%: the present GPU is too slow (90 to 95% is borderline); see [Lots of little stutters](#lots-of-little-stutters). In nested mode this can be normal; test in [native mode](#native-mode-from-a-text-console). |

(The numbers are rules of thumb, not guarantees.)

#### Every row and term

| Term | Meaning |
|---|---|
| `motion x2 . learned . 120Hz fixed` | active mode, frame multiplier, pipeline, refresh and display mode |
| `120Hz fixed` / `120Hz VRR` | display refresh, and whether VRR is actually driving it |
| `present:` | the GPU that generates and drives the display |
| `render` | which GPU the game renders on, inferred from its frame buffers |
| `buffers local` | game frames arrive on the same GPU: zero-copy |
| `buffers xGPU` | dual-GPU path active: frames are copied once into present-GPU memory ("cross-GPU") |
| `bidir` | bidirectional interpolation |
| `base` | base-layer mode: compositor overlays and the cursor are composited after generation; in-game UI is not separated |
| `net: off / blob / online` | learned refiner. `off`: not used. `blob`: weights loaded from the file named in `GAMESCOPE_FRAMEGEN_NET`. `online`: training while you play |
| `adapt` | self-supervised adaptation grading each prediction against reality |
| `requested(OFF)` | you asked for it (for example `NET_ONLINE=1` with the `checked` pipeline) but it is not running; read the terminal log for why (see [Reading the log](#reading-the-log)) |
| `source 40fps[200] gen 78fps[390] repeat 2fps fill 98%` | game (real) and generated frame rates with cumulative counts, repeated slots, and the share of display slots that carried fresh content |
| `ladder learned/motion rung 0/4 full` | effective pipeline and mode, current degradation rung (one cheaper step on the ladder), and recovery state (`full`: no step-down; `hold`: stepped down and holding; `probation`: just stepped back up) |
| `pace: hit 99% ... jitter +/-0.7ms` | share of presents that met their planned time, and the flip-time spread (jitter: how unevenly frames reach the screen), each with a recent-history sparkline |
| `net: online . 3.2k steps . profile loaded` | training steps this session, and whether a per-game profile was loaded |

Which rows appear at each level, and exact wording, are in the
[reference](framegen-reference.md#hud-rows). The `present ... render ... buffers`
row is the routing check; the `rates` row is the output-rate check.

### The metrics log

`GAMESCOPE_FRAMEGEN_METRICS=1` writes one `framegen-metrics:` line to the
terminal every five seconds, and a `TOTAL` line at exit. It also works with
frame generation off, as a baseline. A line looks like this (abbreviated):

```text
framegen-metrics: w=12 real=200 gen=390 dreal=0 rep=10 flip_ms_avg=8.333 ... fill=0.983 ...
```

Read it as: in this five-second window, 200 real and 390 generated frames
reached the screen, 10 slots repeated, and 98.3% of slots carried fresh content
((200+390)/(200+390+10)). That is healthy. The fields to watch:

| Field | What it tells you | Healthy / worrying, and what to do |
|---|---|---|
| `real` / `gen` / `rep` | real frames, generated frames, and refresh slots that repeated the previous image | `rep` small compared with `real + gen`. Large and growing: see [Lots of little stutters](#lots-of-little-stutters). |
| `fill` | share of display slots that carried fresh content; higher is smoother (see [framegen-testing.md](framegen-testing.md) for measured values) | Close to 1.0. Well below 1.0: see [Lots of little stutters](#lots-of-little-stutters). |
| `dl_hit` | share of generated frames that met their planned deadline. It says nothing about how many slots were filled, so read it together with `rep` and `fill` | Close to 1.000. Low and persistent: use a cheaper pipeline. |
| `flip_ms_sd` | frame-to-frame timing jitter in milliseconds | A fraction of a millisecond. Several ms (visible judder): a pacing problem; test in native mode. |
| `src_flip_ms_p50` / `_p95` | how long a live real frame waits from ready to on-screen | Lower is better |
| `bidir_delay_ms` | the added delay in bidirectional mode (0 otherwise) | Check that the number is one you can live with |
| `dreal_flip_ms_*` | the same wait for bidirectional mode's delayed real frames | |
| `slow`, `resets_ring` | frames dropped because generation was too slow; ring resets | `slow` small; `resets_ring` 0. A nonzero `resets_ring` means overload: lower the resolution or use a cheaper pipeline. |

Other fields (resets, commit leads, staging copies, the `q_*` quality meter,
`ring_ovf`, `dreal_drop`) are for diagnosing problems and comparing builds. All
are defined in the
[reference](framegen-reference.md#metrics-log-fields), which also has a
healthy-versus-worrying table. Latency analysis and the gating tools are covered
in [framegen-testing.md](framegen-testing.md).

**Saving the log.** Gamescope writes to the terminal (standard error). To keep
it:

```bash
gamescope ... 2>&1 | tee gamescope.log
grep framegen-metrics gamescope.log
```

If you launch from Steam, start Steam itself from a terminal so its output goes
there.

---

## 7. Troubleshooting

First, run `./scripts/framegen-doctor.sh` (see
[Reading framegen-doctor.sh](#reading-framegen-doctorsh)).

### Is it working?

Short version: `rates` shows `gen` above 0 and `fill` near 100%, `present` shows
`buffers xGPU` on two GPUs, and the HUD shows no `requested(OFF)`. Full detail:
[How to tell it works](#how-to-tell-it-works). If any of that fails, use the
tables below.

### Reading the log

With `--framegen-debug` (and without) the terminal prints lines. Debug lines are
sampled, so only a few appear per second.

| Log line | What it means and what to change |
|---|---|
| `framegen: generated N frame(s) ...` | It is generating. |
| `framegen: GAMESCOPE_FRAMEGEN_BIDIR ignored (requires motion mode and is incompatible with VRR-hybrid pacing)` | Add `--framegen-mode motion`, or drop `GAMESCOPE_FRAMEGEN_VRR_HYBRID`. |
| `framegen: GAMESCOPE_FRAMEGEN_BIDIR ignored on a Vulkan-swapchain backend` | You are on the SDL backend. Use `--backend wayland` (nested) or `drm`. |
| `framegen: --experimental-framegen ignored: this backend does not present generated frames` | You are on a headless or OpenVR backend. Use the `wayland`, `sdl` or `drm` backend. |
| `framegen: ignoring invalid GAMESCOPE_FRAMEGEN_X="..." (using ...)` | The value is not accepted and its default is used. Fix the value (on/off or a number in range). |
| `[Gamescope WSI] Forcing on VK_EXT_swapchain_maintenance1` followed by a black screen | See [Black screen](#black-screen). |

The meaning of every other log line is in the
[reference](framegen-reference.md). To capture a log, see
[Saving the log](#the-metrics-log).

### Common problems

Start with the first rows: they follow the first-run path.

| Symptom | Check | Fix |
|---------|-------|-----|
| `unrecognized option '--experimental-framegen'` | You ran the distribution's gamescope. | Use the built binary (absolute path, or `source ./scripts/env-gamescope-local.sh`), or install your build. |
| Black screen / will not start | See [Black screen](#black-screen). | |
| HUD missing | The HUD variable must come before `gamescope`: `GAMESCOPE_FRAMEGEN_HUD=1 gamescope ...`. | Move it. In Steam launch options, put it in front of `gamescope` too. |
| No `generated` lines, HUD shows `gen 0` | Most often your game already runs at (or near) the display refresh rate, so there are no empty slots to fill. | Cap the game below the refresh rate (the game's limiter, for example `DXVK_FRAME_RATE=60` on 120 Hz; or gamescope `--framerate-limit 30` for tests). Do not change `-r` to dodge this: it must match your monitor. If the game is well below the refresh rate and `gen` is still 0, check that the startup line `framegen: enabled ...` ends in `(dedicated queue)` (without it at most one frame is made per gap; see [Multipliers](framegen-reference.md#multipliers)), then run the doctor and look for `ignored` lines in the log. See [First, give it gaps to fill](../README.md#first-give-it-gaps-to-fill). |
| HUD shows `buffers local` with two GPUs | The game is rendering on the present GPU. | See [wrong GPU](#wrong-gpu-in-the-hud). |
| Lots of little stutters | See [Lots of little stutters](#lots-of-little-stutters). | |
| Controls feel heavy / laggy | See [Controls feel heavy](#controls-feel-heavy). | |
| Ghosted compositor overlay or cursor | Base-layer mode (d) helps. | Add `GAMESCOPE_FRAMEGEN_BASE=1`. |
| The game's own HUD, crosshair or menus smear or shimmer while the camera moves | See [In-game HUD ghosting](#in-game-hud-ghosting). | |
| HDR looks wrong | See [HDR](#hdr). | |
| Steam game does not start | See [Steam game does not start](#steam-game-does-not-start). | |
| A setting has no effect | `--framegen-pipeline` is ignored unless `--framegen-mode motion` is also given. `GAMESCOPE_FRAMEGEN_NET_*` needs the `learned` pipeline or above. Read the log for `ignored` or `ignoring invalid`. `GAMESCOPE_FRAMEGEN_*` variables must be in front of `gamescope`. | Fix the combination. |
| HUD shows `requested(OFF)` | You asked for a feature that is not running. | Read the log ([Reading the log](#reading-the-log)). |
| `ladder` keeps stepping down | The present GPU misses its deadline. | Cheaper pipeline, lower resolution, or `GAMESCOPE_FRAMEGEN_NET_EVERY=2` on an integrated present GPU. |
| Crash with "out of memory" | The **game** is too heavy for the render card. | Lower its settings. |
| Crash or hang | See [Crash or hang](#crash-or-hang). | |

#### Black screen

Work through these in order.

1. Run `./scripts/framegen-doctor.sh` and read the `display:` lines: the card
   that shows your monitor's connector is the present card. Without the
   doctor, `grep -l '^connected' /sys/class/drm/card*-*/status` lists the cards
   with a monitor.
2. Does `PRESENT_DEV` match that card? Does `RENDER_DEV` end in `!`? Are
   `-W`, `-H` and `-r` right for your monitor?
3. Try the same command without `--experimental-framegen`. If that is also
   black, it is a gamescope or display problem, not frame generation.
4. NVIDIA render GPU: check driver 515.43.04 or newer and
   `nvidia-drm.modeset=1` (see [Requirements](#requirements)).
5. Two GPUs, the game's GPU sits at 0%, no real frame arrives, and the log shows
   `[Gamescope WSI] Forcing on VK_EXT_swapchain_maintenance1`: this is an open
   issue. The Gamescope WSI layer hangs on the game's first present. Set
   `ENABLE_GAMESCOPE_WSI=0` in front of `gamescope` so the game presents through
   Xwayland. The root cause is not yet located.

#### Lots of little stutters

This is the most common complaint. Open the HUD at level 2 and look:

- **`ladder` rung above 0, or `pace` hit low (under about 90%; 90 to 95% is borderline):** the present
  GPU is too slow. Use a cheaper pipeline, a lower resolution, or
  `GAMESCOPE_FRAMEGEN_NET_EVERY=2` on an integrated present GPU.
- **`fill` well below 100% with rung 0:** there are empty slots. In
  bidirectional mode, raise the multiplier to at least refresh rate divided by
  the game's fps (this only helps here; if `ladder` steps down, the GPU is
  the limit, so use a cheaper pipeline instead). Otherwise, cap the game below
  the refresh rate.
- **The game runs below half the refresh rate:** see
  [Low frame rates](#low-frame-rates).
- **All numbers fine but motion is still uneven:** the game's own pacing. Set a
  frame cap that the game can hold even in busy scenes; a cap it keeps missing
  makes motion less even.
- **Running nested in a desktop:** test in
  [native mode](#native-mode-from-a-text-console). Uneven pacing is expected
  there.

#### Controls feel heavy

Frame generation never lowers input lag; the default modes add about 2 ms.

1. **Bidirectional mode adds about 20 to 45 ms more than the default modes**
   (more at low frame rates). Turn `GAMESCOPE_FRAMEGEN_BIDIR`
   off.
2. **At low fps the game's own frame queue adds delay.** For DXVK and Proton
   games, put `DXVK_CONFIG="dxgi.maxFrameLatency = 1; dxvk.latencySleep = True"`
   in front of `gamescope`.
3. Check that `GAMESCOPE_FRAMEGEN_BATCH_REPAINT` is not set.
4. Running nested: input timing is capped by your desktop. Test native mode.

#### Wrong GPU in the HUD

- **`present:` shows the wrong GPU:** `PRESENT_DEV` must be the card your monitor
  is plugged into (`--prefer-vk-device`).
- **`render:` shows the wrong GPU:** check `MESA_VK_DEVICE_SELECT=vendor:device!`
  (the `!` makes it exclusive). Remove any `DXVK_FILTER_DEVICE_NAME` or
  `VK_ICD_FILENAMES` you set; they can override it. On NVIDIA Optimus, add
  `__NV_PRIME_RENDER_OFFLOAD=1` and `__VK_LAYER_NV_optimus=NVIDIA_only`.
- On a hybrid laptop, run `scripts/gamescope-hybrid.sh --report -- vkcube`: it
  prints the detection and the decision.

#### In-game HUD ghosting

The game draws its own HUD, crosshair and menus into the picture gamescope
receives, so frame generation treats them as scene content and can smear them
while the camera moves. Base-layer mode (`GAMESCOPE_FRAMEGEN_BASE=1`) only helps
gamescope's own overlays and the cursor, not this. Things that may help: raise
the game's frame cap so fewer frames are invented (only if the game still holds
the higher cap in busy scenes, and stay below the refresh rate), and turn off animated or semi-transparent HUD elements in the game's
settings.

#### HDR

Generated frames are meant to keep HDR; see [section 8](#8-limits-and-what-does-not-work-yet)
for what is verified. If HDR looks washed out, too dark or flickers:

1. Run the same command without `--experimental-framegen`. If it still looks
   wrong, it is gamescope's HDR setup, not frame generation. HDR needs
   `--hdr-enabled`, the Gamescope WSI layer, and an HDR-capable display.
2. If only frame generation shows it, report it with the log (see
   [Crash or hang](#crash-or-hang) for how to collect one).

#### Steam game does not start

1. Start Steam from a terminal so you can see gamescope's errors.
2. `unrecognized option '--experimental-framegen'` means you are running a
   distribution gamescope. Use the absolute path of your build, or install it
   (`meson install`).
3. In launch options, use absolute paths, and keep `%command%` after the final
   `--`.
4. With `gamescope-hybrid.sh --child`, a gamescope host must already be running
   (see [section 4](#hybrid-laptops-launcher)).
5. Run `scripts/gamescope-hybrid.sh --report -- vkcube` to see what would be run.

#### Crash or hang

1. Retry with `--framegen-pipeline checked` and without
   `GAMESCOPE_FRAMEGEN_BIDIR`, `_BASE`, `_VRR_HYBRID` and `_NET_ONLINE`.
2. If it still crashes, run with `--framegen-debug` and
   `GAMESCOPE_FRAMEGEN_METRICS=1`, and keep the log (see
   [Saving the log](#the-metrics-log)).
3. Before you post the log, remove environment dumps, tokens and private paths.
   Some launchers print the full environment, including credentials. Report it
   in the project's issue tracker.

---

## 8. Limits and what does not work yet

### Limits you will notice

- **Experimental.** You may see shimmering edges, ghost trails behind fast
  objects, or the occasional crash. Disocclusions, transparency, particles, fast
  edges and UI are the hard cases (the words are explained in
  [section 1](#what-artifacts-look-like)).
- **Latency.** Frame generation is for smoothness, not lower input lag. Numbers
  and what they mean are in [What it costs](#what-it-costs). Bidirectional mode
  suits some games and players and not others (see
  [section 5](#c-bidirectional-the-smoothest-motion)).
- **Dense particle swarms are the hardest case.** Our test is GravityMark with a
  million asteroids. The finer motion detail (`GAMESCOPE_FRAMEGEN_FIELD_FINE=1`)
  looked blocky there, so it stays off by default. The sharp fallback, which is
  on by default, can fray the edges of the swarm, although it was still clearly
  smoother than frame generation off in our blind test.
- **The monitor must be on the present card** ([section 2](#2-hardware-setup))
  or it will not run.
- **Nested vs. native.** Launched from inside your normal desktop you can see
  frame generation working, but the true smoothness is capped by your desktop's
  own timing. The genuinely smooth experience needs
  [native mode](#native-mode-from-a-text-console). Native DRM/KMS, unusual
  dma-buf modifier combinations and mixed-vendor systems need broader testing.
- **HDR.** Gamescope's own HDR output needs `--hdr-enabled`, the Gamescope WSI
  layer and an HDR-capable display. Generated frames are designed to keep HDR
  (the sharp fallback, for example, shows only real pixels). The visual check on a real HDR
  display is still pending, so do not treat HDR output as validated.
- Cursor motion advances on generated frames: the cursor is kept out of the
  generated history and drawn live. `GAMESCOPE_FRAMEGEN_CURSOR=0` restores the
  old behavior, where it moves only with real frames.
- An overlay-only repaint can be held back up to four vblanks so a ready
  generated frame can fill the slot. A late generated frame is dropped; overload
  shows up as repeated display slots.

**Per mode.** See the notes in each subsection of
[section 5](#5-modes-and-presets). In short: bidirectional adds about 20 to 45
ms more delay than the default modes (on top of their ~2 ms), VRR mode is unvalidated and needs a real VRR monitor, and the AI
refiner needs `--framegen-mode motion` with `--framegen-pipeline learned` or
above.

**Older or weaker present cards** (the *present* card does the generation)

- If the present card cannot finish a generated frame in time, gamescope safely
  skips it and repeats the last frame, so you feel small hitches instead of
  smoothness. Fixes: a cheaper pipeline (`warp` or `checked` instead of
  `learned` or `guided`), a simpler mode (Simple instead of Bidirectional) or a
  lower resolution. Gamescope also steps down by
  itself (see [Automatic degradation](#automatic-degradation-the-ladder)).
- **AMD 5700 XT and similar (RDNA1) and older cards:** we expect the AI refiner
  (f) to be too heavy there, but this is an expectation, not measured here. Try
  `checked` or `warp` first. If the HUD `ladder` steps down, switch to `checked`
  or `warp`, or drop `GAMESCOPE_FRAMEGEN_NET_ONLINE`.
- Very demanding games (for example very high asteroid counts in GravityMark)
  can run the **render** card out of memory. If the game crashes on start with an
  out-of-memory error, lower its settings or resolution.

Open bugs, unexplained results and other tester-level notes are in the
[sharp-edges list](framegen-architecture.md#10-sharp-edges--wip--open-bugs) of
the architecture doc.

---

*Next: the [Configuration Reference](framegen-reference.md) for every flag,
variable and log field; [`framegen-architecture.md`](framegen-architecture.md)
for how it works under the hood; [`framegen-maintenance.md`](framegen-maintenance.md)
before changing the shader, queue, cache, learning or presentation contracts;
[`framegen-testing.md`](framegen-testing.md) to smoke-test a build in a nested SDL
window, including present-wait and SDL fault injection.*
