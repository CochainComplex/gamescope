# Frame Generation — How-To Guide

A plain-language guide to using gamescope's frame generation: what it does, how
to set up the two graphics cards, ready-to-paste commands for each mode, and an
honest list of what doesn't work yet.

> **Heads-up:** this is **experimental** and specific to this fork (it is not in
> upstream gamescope). Expect visual glitches and occasional crashes; retry with
> a lower quality tier or simpler mode when a workload is unstable.

## Preferred visual baseline: GravityMark x4

This is the accepted, most visually appealing GravityMark configuration from
live testing: Motion Extreme, three generated frames per real frame,
bidirectional presentation, strength `0.5`, and the frozen learned profile. It
prioritizes visual smoothness and quality; bidirectional mode adds one real-frame
interval of latency, so this is not the competitive-latency preset.

Set the two PCI IDs and paths for the machine, then run:

```bash
export PRESENT_DEV=1002:5678          # GPU physically connected to the display
export RENDER_DEV=10de:1234!          # GPU rendering GravityMark
export GRAVITYMARK_DIR=/path/to/GravityMark/bin
export FRAMEGEN_BEST_PROFILE="$HOME/.cache/gamescope-fg-gravitymark-best-dc58b2d5.bin"

test -r "$FRAMEGEN_BEST_PROFILE" # this must succeed before continuing
unset GAMESCOPE_FRAMEGEN_NET_ONLINE GAMESCOPE_FRAMEGEN_NET_PROFILE \
  GAMESCOPE_FRAMEGEN_RECORD_COLOR GAMESCOPE_FRAMEGEN_BIDIR_TRACE

GAMESCOPE_BUILD_DIR=build-perf \
GAMESCOPE_FRAMEGEN_BIDIR=1 \
GAMESCOPE_FRAMEGEN_NET="$FRAMEGEN_BEST_PROFILE" \
GAMESCOPE_FRAMEGEN_BIDIR_PHASE_BIAS=0 \
GAMESCOPE_FRAMEGEN_BIDIR_OCCLUSION=0 \
GAMESCOPE_FRAMEGEN_RESERVOIR=1 \
GAMESCOPE_FRAMEGEN_SHADING=1 \
GAMESCOPE_FRAMEGEN_DEBUG_EVERY=60 \
./env-gamescope-local.sh \
gamescope --expose-wayland --backend wayland \
  --prefer-vk-device "$PRESENT_DEV" \
  -W 2560 -H 1440 -r 120 \
  --experimental-framegen \
  --framegen-mode motion \
  --framegen-multiplier 4 \
  --framegen-quality extreme \
  --framegen-strength 0.5 \
  --framegen-debug \
  -- env -C "$GRAVITYMARK_DIR" MESA_VK_DEVICE_SELECT="$RENDER_DEV" \
    ./GravityMark.x64 -vk -temporal 0 -benchmark 0 -fps 1 \
    -asteroids 1000000 -width 2560 -height 1440 -vsync 0
```

The frozen command is repeatable and cannot modify the accepted profile.
Reservoir and shading remain enabled for the complete Extreme configuration,
although their causal passes are not scheduled while bidirectional mode is
active.

### Optional cadence-locked variant

At 120 Hz, x4 can fully populate the display only while the real-frame cadence
stays at or above 30 fps. A workload that changes from a short interval to five
or more vblanks when camera motion begins cannot be covered retroactively by
bidirectional generation: it has already drained the slots planned from the
previous interval, and x4 can insert at most three new frames. The result is an
honest repeat on the newest real frame, often perceived as a brief onset
stutter.

For a smoothness-first 120 Hz test, add this option before
`--experimental-framegen`:

```bash
  --framerate-limit 30 \
```

This asks Gamescope to release client frame callbacks on every fourth display
vblank. When the client and compositor meet those callbacks, each real frame
plus its three generated frames aligns to four display vblanks. It is
vendor-agnostic and does not change reconstruction quality, but it deliberately
caps the game and therefore adds input latency when the game could otherwise
render above 30 fps. Keep it optional rather than treating it as a general
frame-generation default.

The limiter cannot recover a missed client render or a late compositor wakeup.
In particular, a nested test running without `CAP_SYS_NICE` may still measure
three/five-vblank pairs or longer gaps around the intended four-vblank cadence.
At x4, a five-vblank gap necessarily leaves one repeat because only three
generated frames are available. Treat the limiter as a controlled pacing test,
not as the frozen visual baseline or a guaranteed stutter fix.

The original discovery run used in-situ learning. To continue training without
touching the frozen profile, make a writable copy:

```bash
export FRAMEGEN_WORKING_PROFILE="$HOME/.cache/gamescope-fg-gravitymark-working.bin"
cp -a "$FRAMEGEN_BEST_PROFILE" "$FRAMEGEN_WORKING_PROFILE"
chmod u+w "$FRAMEGEN_WORKING_PROFILE"
```

Then remove the frozen command's `GAMESCOPE_FRAMEGEN_NET=...` line and export:

```bash
export GAMESCOPE_FRAMEGEN_NET_ONLINE=1
export GAMESCOPE_FRAMEGEN_NET_PROFILE="$FRAMEGEN_WORKING_PROFILE"
```

---

## 1. What frame generation actually does

Games draw a certain number of frames per second ("fps"). Frame generation
**invents extra frames in between the real ones** so the motion on screen looks
smoother. A game running at 40 fps can be shown at 80 or 120.

It does **not** make the game respond faster — the extra frames are for
smoothness, not for lower input lag. (One mode, "bidirectional", even adds a
tiny bit of lag; more on that below.)

**How it differs from the game's own FSR/DLSS frame generation:** those get the
scene's real motion handed to them by the game (motion vectors, depth, and on
NVIDIA a special optical-flow chip). gameslop gets none of that — it sits
*outside* the game and only ever sees the **finished pictures**, so it has to
work out the motion itself just by looking at the frames. That's a harder job,
so expect it to be a little less clean than in-game DLSS/FSR — but it works on
**any** game and **any** GPU, which they can't.

**And this is on purpose.** Grabbing that data out of the game would mean
reaching into the card that's running it and shipping heavy data around every
frame — which would *slow the game down and add lag*. The whole point is the
opposite: **never touch the game or its responsiveness.** The second card just
takes the finished picture and polishes it afterwards, like a filter on the
output — the game runs exactly as if gameslop weren't there.

---

## 2. The two-graphics-card setup (read this first)

**This is fundamentally a *two-graphics-card* technique.** Making an extra frame
costs real GPU work. If you do it on the *same* card that's running the game,
you steal power *from the game* to fake frames on top of it — usually you'd have
been better off just letting the game render more real frames. So one card is
**generally not worth it**. The whole idea here is to give each card a job:

| Role | Which card | What it does |
|------|-----------|--------------|
| **Render card** | the **stronger** one (e.g. NVIDIA RTX PRO 500, or AMD 7900 XT) | Runs the actual game. |
| **Present card** | the card your **monitor is plugged into** (e.g. AMD 890M, or AMD 5700 XT) | Generates the extra frames **and** sends the picture to your screen. |

Splitting the work this way keeps the frame-generation cost **off** the card
that's busy running the game — the second card can be a laptop's built-in GPU or
an old card you never threw out, doing work "for free" while it would otherwise
sit idle.

**Can I use just one card?** Only in one situation: an **older game that has no
FSR/DLSS frame generation of its own** *and* doesn't fully use your (good) card —
then you have spare power to spend and it's a real win. For a modern game that
already maxes out your GPU, don't bother on a single card: use the game's own
frame-generation setting instead.

### ⚠️ The single most important rule

**Your monitor must be connected to the PRESENT card**, not the render card.
gamescope always draws to whichever card the display cable is physically
plugged into.

- **On a laptop** the built-in screen is normally wired to one GPU. Use that as
  the present device; the firmware wiring, not the vendor name, decides it.
- **On a desktop with two cards**, plug the monitor cable into the **weaker /
  present** card (e.g. the 5700 XT), **not** the powerful one (the 7900 XT).
  If your screen is plugged into the wrong card, frame generation can't run.

### Find your card IDs

Each card has an ID like `vendor:device`, written as four hexadecimal digits on
each side of the colon. List yours:

```bash
lspci -nn | grep -iE "vga|3d|display"
```

Example output from an NVIDIA-render / AMD-present system looks like:

```
01:00.0 3D controller [10de:1234]      ← NVIDIA  (render card — runs the game)
05:00.0 Display controller [1002:5678] ← AMD     (present card — screen is here)
```

`10de:…` is NVIDIA, `1002:…` is AMD, `8086:…` is Intel.

---

## 3. First-time setup (once per terminal)

Open a terminal in this checkout and load the build environment, then tell it
which card is which. Replace the example paths and IDs with your own:

```bash
cd /path/to/gamescope-gameslop
export GAMESCOPE_BUILD_DIR=build-perf
source ./env-gamescope-local.sh

export RENDER_DEV=10de:1234!   # strong card that runs the game
export PRESENT_DEV=1002:5678   # card your monitor is plugged into
```

That's it — the commands below and the supplied launch scripts reuse
`$RENDER_DEV` and `$PRESENT_DEV`.

---

## 4. The basic command

Every command has the same shape:

```bash
gamescope --prefer-vk-device "$PRESENT_DEV" -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode motion --framegen-multiplier 2 \
  -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME-HERE
```

Reading it left to right:

- `--prefer-vk-device "$PRESENT_DEV"` — do the compositing + frame generation on the present card.
- `-W 2560 -H 1440 -r 120` — screen width, height, and refresh rate (match your monitor).
- `--experimental-framegen` — **turns frame generation on** (required).
- `--framegen-mode motion` — the algorithm (see modes below).
- `--framegen-multiplier 2` — show 2× the frames (`3` or `4` for more).
- everything after `--` is **your game**; `MESA_VK_DEVICE_SELECT="$RENDER_DEV"` pins the game to the strong card.

Replace `YOUR-GAME-HERE` with how you'd normally launch the game. For Steam
games, put `gamescope … --` in front of the launch options, or use
`gamescope … -- steam steam://rungameid/<id>`.

### A ready-made test (no game needed)

If you have the GravityMark benchmark, this is a known-good test:

```bash
export GRAVITYMARK_DIR=/path/to/GravityMark/bin
(
  cd "$GRAVITYMARK_DIR"
  gamescope --prefer-vk-device "$PRESENT_DEV" -W 2560 -H 1440 -r 120 \
    --experimental-framegen --framegen-mode motion --framegen-multiplier 2 --framegen-debug \
    -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" \
    ./GravityMark.x64 -vk -temporal 0 -benchmark 0 -fps 1 \
      -asteroids 200000 -width 2560 -height 1440 -vsync 0
)
```

`--framegen-debug` makes gamescope print what it's doing — watch the terminal
for lines like `framegen: generated 1 frame(s) …`.

---

## 5. The modes — pick one, copy the command

Add the marked bits to the basic command. From simplest/safest to fanciest.

### a) Simple (default) — lowest lag, most compatible
```
--experimental-framegen --framegen-multiplier 2
```
Predicts the next frame. Lowest latency, works everywhere, but fast motion can
look a little juddery.

### b) Motion — the recommended everyday mode
```
--experimental-framegen --framegen-mode motion --framegen-quality high --framegen-multiplier 2
```
Tracks how things move for sharper, cleaner extra frames. Extra quality helpers
(consistency checks + self-tuning) are **on automatically**. Good default.

Choose the cost explicitly for the display/framegen GPU:

| Quality | Work enabled | Best fit |
|---------|--------------|----------|
| `low` | forward motion match only | older or bandwidth-limited GPUs |
| `medium` | + reverse consistency and edge agreement | balanced quality/cost |
| `high` | + self-tuning; optional AI | recommended default |
| `ultra` | + bounded acceleration prediction from the preceding checked field | fast present GPUs |
| `extreme` | + color-guided motion-layer reconstruction and a three-real-frame disocclusion resolver | idle second GPU, maximum forward quality |

Gamescope automatically walks downward through these levels if measured GPU
time misses the display deadline. It never oscillates upward within a scene.
The Extreme disocclusion search is on by default and never affects lower tiers;
set `GAMESCOPE_FRAMEGEN_RESERVOIR=0` only for live A/B attribution.

### c) Bidirectional — the smoothest motion
```
--experimental-framegen --framegen-mode motion --framegen-multiplier 2
```
…plus put **`GAMESCOPE_FRAMEGEN_BIDIR=1`** in front of `gamescope`:
```bash
GAMESCOPE_FRAMEGEN_BIDIR=1 gamescope --prefer-vk-device "$PRESENT_DEV" …
```
Blends *between* two real frames instead of guessing ahead — the smoothest
result and the best with see-through effects (smoke, glass). **Trade-off: adds
about one frame of lag**, so avoid it for fast/competitive shooters. Great for
single-player, racing, scenery. Extreme's causal acceleration, guided
reconstruction, disocclusion reservoir and shading-persistence correction do
not run in this mode; bidirectional interpolation uses two checked endpoint
fields instead. `GAMESCOPE_FRAMEGEN_BIDIR_PHASE_BIAS=0.25` is an experimental
compromise for low-source-rate A/B: it slightly evens generated phase spacing
without adding flip latency. The default `0` is the established sharp/snappy path.
`GAMESCOPE_FRAMEGEN_BIDIR_OCCLUSION=0.5` is a separate experimental edge A/B:
it lets a clearly surviving checked side retain slightly more authority instead
of dissolving into the unwarped crossfade. It changes neither motion fields nor
queue/flip timing, and defaults to `0` because the measured spatial gain is small
and still needs broad live validation.

`GAMESCOPE_FRAMEGEN_BIDIR_TRACE=0.5` is an Extreme-only quality candidate for
fast camera motion. It takes one additional, symmetric fixed-point sample of
each endpoint field and accepts it only when the two paths close, both endpoint
confidences support it, and neither path leaves the image. Across three
12-frame held-out GravityMark sets with deliberate camera movement, `0.5` won
26/36 exact pairs and improved pooled MAE/SSIM/edge error. A same-binary Radeon
890M timestamp test measured `0.571 ms` for the 1440p XB30 baseline pipeline and
`0.583 ms` for the separately specialized trace pipeline. Trace `0` compiles
the optional work out. It does not alter estimation, learning, queues, phases,
or flips. Keep it separate from the frozen baseline until it has broader live
validation.

### d) Base-layer — fixes blurry menus / HUD
```bash
GAMESCOPE_FRAMEGEN_BASE=1 gamescope --prefer-vk-device "$PRESENT_DEV" …
```
Generates the game picture *before* the on-screen menus/health bars/cursor are
added, so text and HUD stay crisp instead of smearing. Use this if your HUD
looks ghosted. (Can't be combined with bidirectional.)

### e) VRR / FreeSync / G-Sync monitors
```bash
GAMESCOPE_FRAMEGEN_VRR_HYBRID=1 gamescope --prefer-vk-device "$PRESENT_DEV" --adaptive-sync …
```
Keeps your monitor's adaptive-sync (FreeSync/G-Sync) working *with* frame
generation. **Only does something on a real VRR monitor that's actually in VRR
mode** — on an ordinary monitor (and most laptop screens) it quietly does
nothing and falls back to normal. This uses the causal timeline and cannot be
combined with bidirectional interpolation; requesting both keeps VRR hybrid and
ignores bidirectional mode.

### f) AI refiner (most experimental)
```bash
GAMESCOPE_FRAMEGEN_NET_ONLINE=1 gamescope --prefer-vk-device "$PRESENT_DEV" \
  --experimental-framegen --framegen-mode motion --framegen-quality extreme …
```
A tiny neural net cleans up the motion and **learns your game as you play**.
It works with zero-latency forward prediction; at Extreme it also learns a
three-frame-validated focus mask for tightly bounded shadow/reflection/specular
color trends. `GAMESCOPE_FRAMEGEN_SHADING=0` disables only that color-trend
head for A/B. In bidirectional mode, ML is intentionally conservative: it keeps
both checked motion fields unchanged, can only lower confidence toward the
phase-correct crossfade, and in-situ training updates only that confidence
output row. This avoids endpoint-trained flow corrections that look valid on
the two real frames but trace an artifacting path between them.
Newest and least-tested; turn it off if anything looks worse. To remember what it learned between sessions, add
`GAMESCOPE_FRAMEGEN_NET_PROFILE=$HOME/fg-<gamename>.bin`.

Power users can also train the net offline on a captured scene and measure it
before use (`scripts/framegen-net-train.py` then `scripts/framegen-net-eval.py`)
— see [The learned refiner (Stage C)](framegen-proposals/README.md#the-learned-refiner-stage-c-in-four-steps).

---

## 6. Which one should I use?

| You want… | Use |
|-----------|-----|
| Just try it, keep it simple | **Motion high, x2** (b) |
| Weak display/framegen GPU | **Motion low or medium, x2** |
| The smoothest single-player experience | **Bidirectional** (c) |
| Competitive / fast shooter (lowest lag) | **Simple** (a), *not* bidirectional |
| Crisp menus and HUD | add **Base-layer** (d) |
| You own a FreeSync/G-Sync monitor | add **VRR** (e) |
| Maximum quality without added latency | **Motion extreme + AI** (f) |
| Smoothest output, one-frame latency acceptable | **Bidirectional + AI** |

Start with `--framegen-multiplier 2`. Try `3` or `4` only if the present card
can keep up (see limits).

---

## 7. Limits & what doesn't work yet

**General**
- **Experimental.** You may see shimmering edges, ghost trails behind fast
  objects, or the occasional crash. Frame generation is for *smoothness*, not
  lower input lag.
- **Monitor must be on the present card** (section 2) or it won't run.
- **Nested vs. native.** Launched from inside your normal desktop, you can *see*
  frame generation working, but the true smoothness is capped by your desktop's
  own timing. The genuinely smooth experience needs "native" mode — starting
  gamescope from a text console (Ctrl+Alt+F3), which takes over the screen like
  a game console. That's an advanced step.

**Per mode**
- **Bidirectional** adds ~1 frame of lag — don't use it for reaction-heavy games.
  At 20 fps that interval alone is ~50 ms. Anti-lag/Reflex must pace the game on
  the **render GPU** where input is sampled; a Mesa/RADV setting on a downstream
  AMD presenter cannot reduce an NVIDIA-rendered game's input queue. Generated
  frames improve motion cadence, not the 20 Hz input sampling rate.
- **VRR mode** needs a real VRR (FreeSync/G-Sync) monitor actively in VRR; it
  does nothing on ordinary screens or typical laptop panels.
- **Base-layer** and **bidirectional** can't be used at the same time. So can't
  bidirectional and VRR, or bidirectional and JIT — pick one "smart" mode.
- **AI refiner** needs `--framegen-mode motion` and `--framegen-quality high`
  or above; bidirectional is optional (it works in the zero-latency forward
  path too).

**Older / weaker present cards** (this matters because the *present* card does
the generation)
- If the present card can't finish a generated frame in time, gamescope safely
  **skips it and repeats the last frame** — you'll feel small hitches instead of
  smoothness. Fixes: use a lower multiplier (`2` instead of `4`), a simpler mode
  (a/b instead of c/f), or a lower resolution.
- **AMD 5700 XT and similar (RDNA1) and older cards:** they lack the newer
  matrix/AI acceleration, so the **AI refiner (f) is slow** on them — leave it
  off, or keep the multiplier low. Motion and bidirectional still work.
- Very high asteroid counts / very demanding games can run the **render** card
  out of memory — if the game crashes on start with an out-of-memory error,
  lower its settings/resolution.

**Good to know**
- The performance tuning is automatic: on newer hardware it uses packed-fp16
  shaders, on others it falls back — you don't set this.
- If quality drops because the card is overloaded, gamescope automatically
  steps *down* (motion → simpler → fewer extra frames) to avoid stutter. The
  step-down holds for the rest of the scene (it never oscillates back up
  mid-scene); full quality is re-probed only at the next scene change.

---

## 8. Is it working? / troubleshooting

Add `--framegen-debug` and watch the terminal:

- `framegen: generated N frame(s) …` → it's generating. ✅
- `bidirectional interpolation requested` / `learned bidirectional confidence
  veto active` / `self-supervised adaptation active` → the conservative bidir
  ML path is on. Causal mode instead logs `learned forward-field refinement`.
- `content scene cut detected` → motion mode deliberately presented a real
  endpoint instead of predicting across unrelated content.
- `… ignored (requires …)` → you enabled a mode without its requirement (e.g.
  `GAMESCOPE_FRAMEGEN_BIDIR=1` without `--framegen-mode motion`, or combined
  with a mode that owns its own timeline like base-layer/VRR/JIT).

| Problem | Likely cause / fix |
|---------|--------------------|
| Black screen / won't start | Monitor not on the present card, or wrong `--prefer-vk-device`. |
| No `generated` lines | Game is already hitting your refresh rate (no gaps to fill) — it only generates when the game runs *below* the screen's rate. |
| Lots of little stutters | Present card can't keep up — lower the multiplier / mode / resolution. |
| Ghosted menus / HUD | Add **base-layer** mode (d). |
| Feels laggy | You're on **bidirectional** — switch to simple/motion for lower lag. |
| Crash with "out of memory" | The **game** is too heavy for the render card — lower its settings. |

---

*For the full list of flags and environment toggles (including developer
options), see [`framegen-proposals/README.md`](framegen-proposals/README.md).
For how it works under the hood, see
[`framegen-architecture.md`](framegen-architecture.md). Maintainers should also
read [`framegen-maintenance.md`](framegen-maintenance.md) before changing the
shader, queue, cache, learning, or presentation contracts.*
