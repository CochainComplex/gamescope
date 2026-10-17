# gamescope-gameslop Configuration Reference

Every command-line flag, environment variable, HUD row and metrics field of
the frame-generation feature. This is a lookup page: do not read it from top
to bottom. For a guided introduction, start with the
[User Guide](framegen-howto.md). For how the pieces work internally, see
[`framegen-architecture.md`](framegen-architecture.md).

What frame generation does, in one paragraph: the game draws real frames at
its own pace, and gamescope fills the free display refreshes in between with
generated frames, so motion looks smoother at the same game frame rate. The
default modes hold no real frame back and add about 2 ms (the forced
composite). Bidirectional mode adds about 20 to 45 ms more and is opt-in.

Where to look:

| I want to... | Go to |
|---|---|
| Understand a word | [Glossary](#glossary) |
| Start with a safe command | [Command-line flags](#command-line-flags) |
| Choose a mode or pipeline | [Modes](#modes), [Pipelines](#pipelines) |
| Change a setting | [Start here: the eight variables most people need](#start-here-the-eight-variables-most-people-need) |
| Fix a symptom (ghosting, stutter, lag) | [Symptom to setting](#symptom-to-setting) |
| Read the on-screen HUD | [HUD rows](#hud-rows) |
| Read the metrics log | [Metrics log fields](#metrics-log-fields) |

Contents:

- [Glossary](#glossary)
- [Command-line flags](#command-line-flags)
- [Modes](#modes)
- [Pipelines](#pipelines)
- [Multipliers](#multipliers)
- [Environment variables](#environment-variables)
- [HUD rows](#hud-rows)
- [Metrics log fields](#metrics-log-fields)

## Glossary

Read this first. The rest of the page uses these words without explaining them again.

**Frames and timing**

| Term | Meaning |
|---|---|
| Real frame | A frame the game drew. |
| Generated frame | A frame gamescope made itself to fill a gap between real frames. |
| Slot | One refresh of the display (at 120 Hz, 120 slots per second). A slot is "filled" when it shows new content, real or generated, and "repeated" when it shows the previous image again. |
| Causal, forward | Uses only frames that already exist: predicts ahead from the latest real frame. Real frames are shown live, so this adds no delay. `extrapolate` and `motion` modes work this way. |
| Extrapolate vs interpolate | Extrapolating guesses what comes after the newest real frame. Interpolating (bidirectional) builds a frame between two real frames, so it needs the newer one first and real frames are shown late. |
| Bidirectional | Interpolation between two real frames. Smoother, but real frames are shown late (see `bidir_delay_ms`). Opt-in. |
| Dormant | The scheduler generates nothing and passes real frames through, because the game leaves no free slot (it runs near the refresh rate). |
| Dedicated framegen queue | A separate GPU work lane for generation, so it does not wait behind rendering and compositing. Most modern GPUs have one. The startup line `framegen: enabled ...` ends in `(dedicated queue)` when it exists. Without it, the log says `no dedicated queue` and expect fewer generated frames (see [Multipliers](#multipliers)). |
| Output ring | The fixed set of images that generated and real frames are queued in before the display takes them. Its size depends on the multiplier. |
| Rung, ladder | The degradation ladder: ordered steps of cheaper generation that gamescope walks down when the deadline is missed and back up on headroom. The HUD `ladder` row shows the current rung. |
| Provenance | Where a frame timestamp comes from (for example display feedback versus software clock). A change forces a pacing reset (`resets_prov`). |

**Image processing**

| Term | Meaning |
|---|---|
| History | The last few real frames that generation looks at. It is invalidated, and generation restarts, after scene cuts and mode changes. |
| Motion field | A coarse grid of motion vectors saying where each block of the picture moved between two real frames. Normally 1/8 of the output size. |
| Confidence, killed | Each vector carries a confidence. A vector whose confidence falls below the survive threshold is "killed": that area falls back to a safer guess instead of being moved. |
| FB check | Forward-backward consistency check: a vector is trusted only if following it forward and then back lands near the start. |
| Disocclusion | Newly uncovered background that no earlier frame showed, so motion matching cannot predict it. |
| Reservoir | A store of three past real frames, used to fill newly uncovered background (the `guided` pipeline). |
| Shading persistence | Tracks slow colour changes such as shadows, reflections and specular highlights, so they keep changing smoothly (the `guided` pipeline with the net). |
| Self-calibration, adaptation | Gamescope grading its own motion estimate against the next real frame and adjusting its trust. |
| Net, blob | The small learned refiner and its weights file. `NET_ONLINE` trains it while you play; a blob is trained offline. |
| Base layer | The game picture before compositor overlays and the gamescope cursor are drawn. Base-layer generation works on that and composites overlays afterwards. |
| VRR hybrid | A mode that keeps adaptive sync (VRR) working with generation. |

**GPUs**

| Term | Meaning |
|---|---|
| Render GPU | The GPU the game draws on. |
| Present GPU | The GPU that generates frames and drives the display. On a hybrid laptop this is normally the integrated GPU. |
| Local vs xGPU buffers | `buffers local`: game frames are already in present-GPU memory. `buffers xGPU`: each frame is copied once from the render GPU. |


## Command-line flags

Quick reference. A minimal working command, a safe first test (first-run
walk-through: [User Guide, section 3](framegen-howto.md#3-first-run); the
settings that felt best in real games are in
[Choosing settings](../README.md#choosing-settings)):

```bash
gamescope --prefer-vk-device "$PRESENT_DEV" -W 2560 -H 1440 -r 120 \
  --experimental-framegen --framegen-mode motion --framegen-pipeline checked \
  --framegen-multiplier 2 -- env MESA_VK_DEVICE_SELECT="$RENDER_DEV" YOUR-GAME
```

Verified against `src/main.cpp`. Invalid values make gamescope print an error
and exit.

| Flag | Values (default) | Effect |
|---|---|---|
| `--experimental-framegen` | none (off) | Turns frame generation on. Forces the composite path and disables adaptive sync (VRR) and tearing flips while active, so on a VRR monitor you get a fixed refresh unless you use `GAMESCOPE_FRAMEGEN_VRR_HYBRID=1` with `--adaptive-sync`. Ignored, with a log line, on backends that do not present generated frames (headless, OpenVR). |
| `--framegen-multiplier N` | `2`, `3`, `4` (`2`) | Frame multiplier. Default forward generation fills every free display slot whatever N; bidirectional mode makes at most N − 1 frames per real pair. Use `2` at half the refresh rate or more, otherwise `4`; see [Multipliers](#multipliers). |
| `--framegen-mode MODE` | `extrapolate`, `motion`, `blend` (`extrapolate`) | Generation algorithm. See [Modes](#modes). |
| `--framegen-pipeline NAME` | `warp`, `checked`, `learned`, `predict`, `guided` (`warp`) | Amount of work done per generated frame. Has an effect only with `--framegen-mode motion`. Gamescope does not reject the combination: the startup line `framegen: enabled mode=extrapolate pipeline=checked ...` still prints the pipeline, so check the `mode=` part. See [Pipelines](#pipelines). |
| `--framegen-strength S` | 0.0 to 1.0 (`0.5`) | How far a forward-predicted frame is pushed ahead of the real one. `0.5` is the neutral value. Lower means less ghosting (trails behind moving objects) but less prediction, so motion can look less smooth. Applies to forward frames in `extrapolate` and `motion` modes, on the default deadline path as well as classic pacing; not used by `blend`. Bidirectional interpolation takes its phases from the real frames' timestamps instead (see [Pipelines](#pipelines)). |
| `--framegen-debug` | none (off) | Logs framegen history, dispatch and present cadence. `GAMESCOPE_FRAMEGEN_DEBUG_EVERY` throttles repeated lines. Also enables the quality meter behind the `q_*` metrics fields. |
| `--framerate-limit N` | integer (`0`, off) | Upstream gamescope flag, useful for cadence-locked tests: a divisor of the refresh rate, rounded down (60 / 59 gives 60 fps, 60 / 25 gives 30 fps). Releases client frame callbacks on every Nth vblank. Also adjustable through the `GAMESCOPE_FPS_LIMIT` X11 property. |

Related upstream flags used in this documentation: `--prefer-vk-device
vendor:device` (the present GPU; find the IDs in
[Find your card IDs](framegen-howto.md#find-your-card-ids)), `-W`/`-H`/`-r`
(output size and refresh), `-f` (fullscreen), `--backend` (`wayland` for nested
use, `sdl` for windowed use, `drm` for native), `--expose-wayland`,
`--adaptive-sync`.

Backends that own their output images, so every framegen feature is available:
DRM and nested Wayland. Backends that present through a Vulkan swapchain (SDL)
do not: bidirectional mode is ignored there and base-layer generation is forced
on. Headless and OpenVR do not present generated frames at all.

## Modes

| Mode | Selection | Behavior |
|---|---|---|
| Extrapolate | `--framegen-mode extrapolate` (default) | Forward prediction from the latest real frame. Lowest generation cost and latency; fast motion can look juddery. |
| Motion | `--framegen-mode motion` | Motion-compensated forward prediction. Recommended. The [pipelines](#pipelines) apply to this mode. |
| Blend | `--framegen-mode blend` | Plain frame average. Kept as a debug aid. |

Presentation variants, selected by environment variable on top of motion mode:

| Variant | Setting | Behavior |
|---|---|---|
| Bidirectional interpolation | `GAMESCOPE_FRAMEGEN_BIDIR=1` | Blends between two real frames. Shows every real frame later than live (see `bidir_delay_ms`). Requirements: see the [`GAMESCOPE_FRAMEGEN_BIDIR`](#onoff-switches) row. |
| Base layer | `GAMESCOPE_FRAMEGEN_BASE=1` | Generates before compositor overlays and the gamescope cursor are composited. In-game UI is part of the game frame and is not separated. See the [`GAMESCOPE_FRAMEGEN_BASE`](#onoff-switches) row. |
| VRR hybrid | `GAMESCOPE_FRAMEGEN_VRR_HYBRID=1` with `--adaptive-sync` | Causal path that keeps adaptive sync working. Needs a dedicated framegen queue and a display actually in VRR. Not validated on a VRR panel. |

With a dedicated framegen queue, fixed-refresh generation uses deadline
scheduling by default (see the architecture document). The startup log line
`causal fixed-refresh deadline scheduling active by default` confirms the causal
path is active.

## Pipelines

The pipelines are cumulative: each one includes every layer of the pipelines
before it and adds work on top, in the order `warp`, `checked`, `learned`,
`predict`, `guided`. Gamescope steps down automatically when measured GPU time
misses the display deadline (one rung of the degradation ladder at a time,
toward cheaper pipelines and finally the extrapolate mode, or the blend crossfade in bidirectional mode), and steps back up
one rung after sustained headroom (`GAMESCOPE_FRAMEGEN_RECOVER`; no step-up with bidirectional mode or VRR hybrid). The ladder never stops generating. In forward mode it never lowers the number of frames made; multiplier notches apply only to the classic, shared-queue and bidirectional paths. The
architecture document describes the ladder.

| Pipeline | Work added | Status |
|---|---|---|
| `warp` | Forward motion match only | Cheapest; default. Snappy, but shows artifacts. |
| `checked` | Reverse (forward-backward) consistency and full-resolution edge agreement | Its checks reject motion vectors, and it felt choppy in a real game. |
| `learned` | Self-supervised adaptation and the optional learned refiner | Best real-game start so far, with in-game learning on (see the notes). Not the default yet. |
| `predict` | Bounded acceleration prediction from the preceding checked field | Experimental. Its acceleration alters motion; it felt choppy in a real game and looks net negative so far. |
| `guided` | Color-guided motion-layer reconstruction, a three-real-frame disocclusion resolver and the shading-persistence head | Close second to `learned` in the real-game tests; most passes and the highest GPU cost. Experimental. |

The Status column is a short verdict from a few real-game tests. The User Guide
keeps the practical advice in one place: [Which one should I
use?](framegen-howto.md#which-one-should-i-use).

Notes:

- The guided disocclusion search is on by default and never affects cheaper
  pipelines (`GAMESCOPE_FRAMEGEN_RESERVOIR=0` turns it off for A/B testing).
- Bidirectional mode forces reservoir and shading off and uses two checked
  endpoint fields instead of guided's causal reconstruction. The learned refiner
  is conservative there (it can only lower confidence). Details:
  [Bidirectional interpolation](framegen-architecture.md#33-bidirectional-interpolation-b3-gamescope_framegen_bidir1-opt-in).
- The learned refiner (the net) runs only with `GAMESCOPE_FRAMEGEN_NET_ONLINE=1`
  or a `GAMESCOPE_FRAMEGEN_NET` weights blob. Without either, `learned` runs
  without it: checked plus self-calibration (the adaptation layer) only. The
  net may raise a vector's confidence only where the refined vector predicts
  the pair of real frames better than holding (zero motion), so it rescues
  vectors that the checks rejected but that demonstrably beat holding; that
  helped both smoothness and image quality in the real-game tests. For in-game
  learning, give each game its own `GAMESCOPE_FRAMEGEN_NET_PROFILE` file, so
  the net keeps what it learned.
- GPU cost per generated frame on a Radeon 890M at 2560x1600, depending on the
  game: `warp` about 1.1 to 1.3 ms, `learned` 1.2 to 3.1 ms, `guided` 1.2 to
  3.8 ms (three Far Cry games; a frame at 120 Hz is 8.3 ms). Weaker GPUs or
  higher resolutions may need `warp` or `checked`.
- The Status column summarizes real-game tests with one viewer on one hybrid
  laptop (Far Cry Primal, Far Cry New Dawn and Far Cry 3 Blood Dragon). The blind A/B tests of
  `GAMESCOPE_FRAMEGEN_SHARP_FALLBACK` and `GAMESCOPE_FRAMEGEN_FIELD_FINE` used
  `checked`; `learned` has not been blind-tested yet on a sharp-edged shooter or
  a particle swarm, and the sharp fallback is not yet verified on it. What suits
  a game depends on the game and on the hardware; results and a starting
  recipe: [Choosing settings](../README.md#choosing-settings).

## Multipliers

`2`, `3` or `4` (default `2`). Rule: use `2` when the game runs at half the
refresh rate or more, otherwise `4`. The multiplier adds no latency. What N
does depends on the scheduling path; the startup log line
`framegen: enabled ...` ends in `(dedicated queue)` when a dedicated framegen
queue exists.

| Path | What N does |
|---|---|
| Forward (`extrapolate`, `motion`) on a dedicated framegen queue: causal deadline scheduling, the default | Generation does not depend on N. Every free display slot until the next real frame gets one generated frame, up to 1.5 real-frame intervals ahead. N only reserves output buffers: 2 × N generated images and 4 + 2 × N output ring images. |
| Bidirectional | At most N − 1 interpolated frames per pair of real frames. With N below refresh ÷ source FPS, display slots stay empty and frames repeat. |
| Classic pacing (`GAMESCOPE_FRAMEGEN_CLASSIC=1`) on a dedicated framegen queue | N − 1 frames are predicted after each real frame and the ones the next real frame makes unnecessary are discarded, so a higher N costs GPU time when the source is fast. |
| VRR hybrid | One generated frame per real frame, whatever N. |
| No dedicated framegen queue | At most one generated frame per gap between real frames, whatever N. |

Measured at about 41 FPS on 120 Hz (Far Cry New Dawn with HD textures; see the
[testing guide](framegen-testing.md)): forward x2, x3 and x4 generated the same
~75 frames/s and filled about 95–99% of the display slots (the 95% run was a game-rate dip to 37 FPS); bidirectional filled 71%
at x2, 98% at x3 and 100% at x4, with the same latency. So the rule matters
for bidirectional mode: it needs N of at least refresh rate ÷ game FPS (40 FPS
on 120 Hz: 3 or 4). In forward mode the number changes nothing you can see.

## Environment variables

On/off switches accept `1`/`0`, `true`/`false`, `yes`/`no` and `on`/`off`. Any
other value, including an empty one, keeps the default and logs
`framegen: ignoring invalid ...` once. Numeric variables that have a range
clamp out-of-range values to it silently (for example `FB_TOL=20` becomes `8`
and `BIDIR_OCCLUSION=5` becomes `1`); non-numeric values keep the default.

Variables are read once at startup. The eight most useful come first, then the
full tables; the [Developer and validation](#developer-and-validation) section
holds variables for testing that are not part of normal use.

### Start here: the eight variables most people need

Everything else has a sensible default. Set these as needed, in front of the
gamescope command (for example `GAMESCOPE_FRAMEGEN_HUD=2 gamescope ...`).

| Variable | Default | Use when |
|---|---|---|
| `GAMESCOPE_FRAMEGEN_HUD` | `0` | You want to see on screen what generation is doing. `1` shows the basics, `2` adds timing. See [HUD rows](#hud-rows). |
| `GAMESCOPE_FRAMEGEN_METRICS` | off | You want numbers to compare two settings or to report a problem. Logs a line every 5 s. See [Metrics log fields](#metrics-log-fields). |
| `GAMESCOPE_FRAMEGEN_BIDIR` | off | You prefer smoother motion over the lowest delay. Real frames are then shown late. |
| `GAMESCOPE_FRAMEGEN_BASE` | off | Compositor overlays or the gamescope cursor smear or ghost in generated frames. In-game UI is part of the game frame and is not separated. |
| `GAMESCOPE_FRAMEGEN_NET_ONLINE` | off | You use `--framegen-mode motion` with `--framegen-pipeline learned` (or `guided`) and want the refiner to learn while you play. |
| `GAMESCOPE_FRAMEGEN_NET_PROFILE` | unset | You use `NET_ONLINE`: one file path per game, so what was learned is kept between sessions. |
| `GAMESCOPE_FRAMEGEN_NET_EVERY` | `1` | You use `NET_ONLINE` on an integrated GPU: set `2` to halve the training cost. |
| `GAMESCOPE_FRAMEGEN_RECOVER` | on | Set `0` if you want a stepped-down rung to stay down until the history is invalidated, instead of stepping back up after headroom. No effect with `GAMESCOPE_FRAMEGEN_BIDIR` or VRR hybrid. |

### Symptom to setting

| What you see | Try | Where |
|---|---|---|
| Ghost trails behind moving objects | `--framegen-strength 0.3` (lower means less ghosting, less prediction) | [flags](#command-line-flags) |
| Judder or uneven pacing | `GAMESCOPE_FRAMEGEN_HUD=2`, then check `pace` and `ladder`; a cheaper pipeline (`warp`) if the ladder keeps stepping down; `GAMESCOPE_FRAMEGEN_NET_EVERY=2` if the net is training on an iGPU | [HUD rows](#hud-rows), [Pipelines](#pipelines) |
| Input feels delayed | Make sure `GAMESCOPE_FRAMEGEN_BIDIR` is off; compare `src_flip_ms_p95` with framegen off | [Metrics](#metrics-log-fields) |
| Overlay or cursor smears | `GAMESCOPE_FRAMEGEN_BASE=1` | [Modes](#modes) |
| Few generated frames, `rep` high | Look for `(dedicated queue)` in the startup log; or the game already runs near the refresh rate (nothing to fill): cap it below, see [User Guide, Frame-rate limit](framegen-howto.md#frame-rate-limit-optional); raise the multiplier only for bidirectional mode | [Multipliers](#multipliers) |
| Black screen with game on one GPU, gamescope on the other | `ENABLE_GAMESCOPE_WSI=0` for the game | [note on `ENABLE_GAMESCOPE_WSI`](#developer-and-validation) |

### All user-facing variables

#### On/off switches

| Variable | Values (default) | Effect |
|---|---|---|
| `GAMESCOPE_FRAMEGEN_METRICS` | on/off (off) | Logs one `framegen-metrics:` summary line per 5 s window plus a `TOTAL` line at exit. Also works with framegen off, as a baseline. See [Metrics log fields](#metrics-log-fields). |
| `GAMESCOPE_FRAMEGEN_BIDIR` | on/off (off) | Bidirectional interpolation. Needs motion mode on a backend that owns its output images (DRM, nested Wayland); excluded by VRR hybrid; with `_BASE`, decided per scene. |
| `GAMESCOPE_FRAMEGEN_BASE` | on/off (off; forced on, whatever you set, on Vulkan-swapchain backends such as SDL) | Base-layer generation with late overlay composite. With `_BIDIR`, decided per scene: base-layer scenes run causal, the others bidirectional. |
| `GAMESCOPE_FRAMEGEN_VRR_HYBRID` | on/off (off) | Keeps adaptive sync with frame generation. Needs a dedicated framegen queue; on a shared queue it is ignored and the log says so. |
| `GAMESCOPE_FRAMEGEN_CURSOR` | on/off (on) | Keeps the cursor out of generated history and draws the current cursor on every generated frame. Off bakes the cursor into history, so it moves only with real composites. |
| `GAMESCOPE_FRAMEGEN_RECOVER` | on/off (on) | Lets the degradation ladder step back up after sustained headroom. Off holds a stepped-down rung until the history is invalidated. No effect with `GAMESCOPE_FRAMEGEN_BIDIR` or VRR hybrid. |
| `GAMESCOPE_FRAMEGEN_SHARP_FALLBACK` | on/off (on) | Where the motion evidence is weak, moves the real frame only as far as the confidence allows instead of crossfading the motion warp with the color extrapolation; `0` restores the crossfade (double edges and blur at confidence changes). Applies to the motion-mode pipelines `warp`, `checked` and `learned` (and to `predict` only until it has history). `guided` and bidirectional mode use their own shaders and are not affected. Shows only real pixels, so HDR is unaffected; about 10% cheaper per generated frame at 1440p and 2160p, equal at 1080p (Radeon 890M); frays edges on dense particle swarms. |
| `GAMESCOPE_FRAMEGEN_FIELD_FINE` | on/off (off; only has an effect with `warp` and `checked`) | Uses a finer motion field, 1/4 of the output size instead of 1/8: one more pyramid level, 4x the field texels and one more pyramid dispatch and refinement per direction. Ignored for `learned`, `predict` and `guided`, and `GAMESCOPE_FRAMEGEN_RECORD` is ignored while it is on. Looked crisper on a sharp-edged shooter but blocky on dense particle swarms, so it stays opt-in. |
| `GAMESCOPE_FRAMEGEN_NET_ONLINE` | on/off (off) | Trains the learned refiner while you play. `learned` and above use the net only with this or a `GAMESCOPE_FRAMEGEN_NET` blob; without either they run without it (checked plus self-calibration only). Pair it with `GAMESCOPE_FRAMEGEN_NET_PROFILE`, one file per game, to keep what it learned. |

#### Values

Tuning and experimental variables (`FB_TOL`, `DEBUG_EVERY`, `NET_LR`, `BIDIR_OCCLUSION`, `BIDIR_TRACE`) are in [Advanced and A/B switches](#advanced-and-ab-switches).

| Variable | Values (default) | Effect |
|---|---|---|
| `GAMESCOPE_FRAMEGEN_HUD` | 0 to 2 (0) | On-screen HUD detail. See [HUD rows](#hud-rows). |
| `GAMESCOPE_FRAMEGEN_HUD_SCALE` | 0 to 6 (0) | HUD scale. `0` means automatic: 2 to 4 based on output height (height / 720, clamped). |
| `GAMESCOPE_FRAMEGEN_HUD_CORNER` | `tl`/`tr`/`bl`/`br`/`bc` (`bc`) | HUD position. |
| `GAMESCOPE_FRAMEGEN_NET` | path | Offline weights for the learned refiner: a bare GSFR blob from `scripts/framegen-net-train.py`. It is the starting point for every input encoding the profile has no set for. A per-game profile is rejected here; pass it as `NET_PROFILE`. |
| `GAMESCOPE_FRAMEGEN_NET_EVERY` | >= 1 (1) | Train on every Nth real frame. Use `2` on an iGPU-class present GPU (pacing stays equal; it played best in the real-game test) and `1` on a desktop card. |
| `GAMESCOPE_FRAMEGEN_NET_PROFILE` | path | Per-game profile the in-situ net loads and, with `NET_ONLINE`, saves. Without `NET_ONLINE` it is served read-only. Give each game its own file. It holds one weight set per input encoding: the composited output in SDR, in HDR10 (PQ), or with colour management off, or the game's base layer by colourspace. A set is only served and trained under its own encoding, and all pipelines share it. Format GSFP v1; any other file, including profiles from builds before 2026-10-17, is ignored and replaced by a fresh profile on the first save. |

#### Advanced and A/B switches

For comparing one layer against another. Normal use never needs them; each
layer is on already wherever its pipeline uses it. Setting a layer's switch to
`0` turns it off so you can see what it contributes. The last five rows are
tuning and experimental values.

| Variable | Values (default) | Effect |
|---|---|---|
| `GAMESCOPE_FRAMEGEN_FB` | on/off (on; only has an effect from `checked` up) | Forward-backward consistency check of the motion field. |
| `GAMESCOPE_FRAMEGEN_AGREE` | on/off (on; only has an effect from `checked` up) | Full-resolution agreement check of warped content. |
| `GAMESCOPE_FRAMEGEN_ADAPT` | on/off (on; only has an effect from `learned` up) | Self-supervised adaptation: each real frame grades the field that predicted it. |
| `GAMESCOPE_FRAMEGEN_RESERVOIR` | on/off (on; only has an effect with `guided`) | Three-real-frame disocclusion reservoir. |
| `GAMESCOPE_FRAMEGEN_SHADING` | on/off (on; only has an effect with `guided` and the net) | Shading-persistence color-trend head (shadow, reflection and specular color trends). |
| `GAMESCOPE_FRAMEGEN_SINGLE_QUEUE` | on/off (off) | Does not request a dedicated framegen queue; generation shares the composite queue. |
| `GAMESCOPE_FRAMEGEN_FB_TOL` | 0.05 to 8 (0.75) | Forward-backward round-trip tolerance in motion-field texels. Setting it stops automatic calibration. |
| `GAMESCOPE_FRAMEGEN_DEBUG_EVERY` | >= 1 (60) | With `--framegen-debug`, log every Nth repeated debug line. Set `1` briefly for a scheduling trace, and restore `60` for visual judgment so logging does not add CPU jitter. An invalid value prints an error and uses 60. |
| `GAMESCOPE_FRAMEGEN_NET_LR` | (0, 0.1] (3e-4) | In-situ learning rate. |
| `GAMESCOPE_FRAMEGEN_BIDIR_OCCLUSION` | 0 to 1 (0) | Experimental. Lets a clearly surviving checked side keep more authority in bidirectional mode instead of dissolving into the unwarped crossfade (`0.5` is a moderate value). Changes neither motion fields nor queue or flip timing. Off by default pending broader live validation. |
| `GAMESCOPE_FRAMEGEN_BIDIR_TRACE` | 0 to 1 (0; `guided` only) | Experimental. Adds one extra sample of each endpoint field to help fast camera motion in bidirectional mode. `0` compiles the optional work out. Measurements: [`framegen-testing.md`](framegen-testing.md). |

### Developer and validation

These exist for testing. For the `RECORD_COLOR` family and the validation
workflow, see [`framegen-testing.md`](framegen-testing.md).

| Variable | Values (default) | Effect |
|---|---|---|
| `GAMESCOPE_FRAMEGEN_CLASSIC` | on/off (off; validation) | Selects the classic batch scheduling path instead of deadline scheduling on a dedicated queue. |
| `GAMESCOPE_FRAMEGEN_BENCHMARK` | on/off (off; developer) | Runs the generation microbenchmark and exits. |
| `GAMESCOPE_FRAMEGEN_COMMIT_LEAD_MS` | > 0 (unset: automatic) | Experiment, native DRM only: replaces the learned generated-frame commit lead after its normal warm-up. |
| `GAMESCOPE_FRAMEGEN_BATCH_REPAINT` | on/off (off; experiment) | Requests a composite after each generated batch instead of only waking the compositor. Causal mode only: ignored in bidirectional mode and with VRR hybrid. On a GPU-bound client (GravityMark 1M, 2048x1280 at 120 Hz) real frames rose from 34 to 44 FPS with steadier flips, but the source-to-flip p95 grew from about 11 to 25 ms, and a blind A/B rated it laggier, so it is off. |
| `GAMESCOPE_FRAMEGEN_RECORD` | directory (developer) | Writes field-resolution training samples. Ignored while `GAMESCOPE_FRAMEGEN_FIELD_FINE=1`, because the samples hold 1/8-scale fields. |
| `GAMESCOPE_FRAMEGEN_RECORD_MAX` | >= 1 (1000) | Maximum training samples. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR` | directory (validation) | Full-colour held-out validation capture (motion and bidirectional, dedicated queue, base-layer off). |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR_MAX` | >= 1 (8) | Maximum validation samples. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR_SKIP` | >= 0 (0) | Real frames skipped before capturing. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR_SPAN` | 2 to 16 (2) | Real frames between the A and C endpoints. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR_OFFSET` | 1 to SPAN-1 (1) | Position of the held-out B frame. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR_PHASE_TOLERANCE` | 0 to 1 (1) | Accepted deviation of the B frame's measured phase from OFFSET/SPAN. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR_SWEEP` | `occlusion`/`trace` (`occlusion`) | Which experimental strength the 0/0.5/1 sweep varies. |

Removed or no-op variables: `GAMESCOPE_FRAMEGEN_JIT` is accepted and ignored
(any value, even `0`, only logs a notice); use `GAMESCOPE_FRAMEGEN_CLASSIC=1`
for the classic path.

Every `GAMESCOPE_FRAMEGEN_*` name found in `src` and `scripts` appears in the
tables above or in the note on removed variables.

Other variables that matter for a setup, not read by gamescope itself:
`MESA_VK_DEVICE_SELECT=vendor:device` (set for the game, pins the render GPU; IDs: [Find your card IDs](framegen-howto.md#find-your-card-ids)) and, on NVIDIA
Optimus, `__NV_PRIME_RENDER_OFFLOAD=1` and `__VK_LAYER_NV_optimus=NVIDIA_only`.
`ENABLE_GAMESCOPE_WSI=0` (read by the Gamescope WSI layer; gamescope sets it
to 1 only when unset) makes the game present through Xwayland: the current
workaround for a black screen with a game rendering on one GPU and gamescope
on the other.
The launcher scripts (`gamescope-hybrid.sh`, `framegen-doctor.sh`,
`run-framegen-native.sh`) additionally read `GAMESCOPE_BUILD_DIR`, `PRESENT_DEV`
and `RENDER_DEV`; see [`scripts/README.md`](../scripts/README.md) and
[Hybrid laptops](../scripts/README.md#hybrid-laptops-gamescope-hybridsh).

`gamescope-hybrid.sh` takes its own flags before the gamescope options: `--report`
(print the detected split and the command, run nothing), `--steer-child` (also
steer gamescope's direct child to the game GPU; for a single game, never the Steam
client) and `--child` (start no gamescope, only steer the game; for a launcher
prefix such as `--child -- %command%`). See `scripts/gamescope-hybrid.sh --help`.

## HUD rows

`GAMESCOPE_FRAMEGEN_HUD=1` draws the first five rows; `2` adds the last two.
`GAMESCOPE_FRAMEGEN_HUD_SCALE` sizes it and `GAMESCOPE_FRAMEGEN_HUD_CORNER`
places it.

| Level | Row | Content |
|---|---|---|
| 1 | `gameslop` | Version, active mode, multiplier, pipeline, refresh, and `fixed` or `VRR` (whether VRR is actually driving the display). |
| 1 | `present` | The GPU that generates and drives the display; `render` (vendor of the client dma-buf from its DRM modifier; linear staged buffers show the single other Vulkan device name when known, otherwise other-GPU; linear local buffers show present-GPU); `buffers local` (client frames arrive on the same GPU, zero-copy) or `buffers xGPU` (dual-GPU path: frames are copied once into present-GPU memory). |
| 1 | `modes` | `bidir`, `base`, `net` and `adapt` state. `bidir` is bidirectional interpolation. `base` is base-layer mode. `net` is `off`, `blob` (offline weights) or `online` (training while you play). `adapt` is self-supervised adaptation. `requested(OFF)` means you asked for the feature but it is not running; check the terminal log for the reason (fallback or incompatible combination). `vrr:requested(off)` appears when VRR hybrid was requested but is not active. |
| 1 | `rates` | `source 40fps[200] gen 78fps[390] repeat 2fps fill 98%`-style line: source (real) and generated frame rates with cumulative counts in brackets, repeated slots per second, and `fill`, the share of presented slots that carried fresh content (at most 100%; `fill 98%` is healthy, a value well below that means many slots repeat the previous frame). |
| 1 | `ladder` | Effective pipeline and mode, current degradation rung out of the maximum, and recovery state: `full` (no step-down), `hold` (stepped down and holding), `probation Ns` (just stepped back up; a miss now doubles the wait), `recover in Ns` (back-off before the next step-up), `watching` (measuring headroom for a step-up). |
| 2 | `pace` | `hit N%` (share of presents that met their planned deadline, with a sparkline of recent windows) and `jitter +/-N ms` (flip-time spread, with a sparkline on a fixed 0 to 4 ms scale). |
| 2 | `net` | Shown only while the refiner is active: `online` or `offline`, training steps this session, and whether a per-game profile was loaded. |

## Metrics log fields

`GAMESCOPE_FRAMEGEN_METRICS=1` logs one line per five-second window:

    framegen-metrics: w=12 real=200 gen=390 dreal=0 rep=10 flip_ms_avg=8.333 ... fill=0.983 ...

(abbreviated; the real line carries every field below as `name=value`). The
label after the colon is `w=N` (window number) and the final line over the whole
run is labeled `TOTAL w=N`. Extract the lines with
`grep framegen-metrics gamescope.log`. New fields are appended at
the end of the line, so parsers that read fields by name keep working. Fields
that were not measured print `0` (or `0.000`).

### Key fields to watch

Read these first when comparing runs. Thresholds are rules of thumb, not
guarantees.

| Field | Healthy | Worrying |
|---|---|---|
| `fill` | Close to 1.0 | Well below 1.0: many slots repeat the previous frame |
| `rep` | Small relative to `real + gen` | Large and growing |
| `flip_ms_sd` | A fraction of a millisecond | Several ms (visible judder) |
| `dl_hit` | Close to 1.0 | Low: planned deadlines are being missed |
| `slow` | Small | Persistent: generation is too slow; expect the ladder to step down |
| `resets_ring` | 0 | Nonzero: overload |

#### Symptom to field

| What you see | Fields to read |
|---|---|
| Stutter or judder | `fill`, `rep`, `flip_ms_sd`, `dl_hit`, `slow`, `resets_ring` |
| Input lag | `src_flip_ms_p95` (compare with a framegen-off run), `bidir_delay_ms` (bidirectional mode) |
| Slow or wrong-GPU copies (dual-GPU) | `copy_ms_avg`, `copy_bytes` |
| Artifacts | `q_bad_mv`, `q_killed_mv` (compare runs on the same scene only) |

Fields marked "developer diagnostic" below (commit lead, bias, provenance,
chain resets and similar) matter only when a developer asks for them. Ignore
them otherwise.

### Frame counts

| Field | Meaning |
|---|---|
| `real` | Real (game) frames presented fresh. |
| `gen` | Generated frames presented. |
| `dreal` | Delayed real frames presented (bidirectional mode shows real frames late). 0 outside bidirectional mode. |
| `rep` | Refresh slots that repeated the previous scanout. |
| `fill` | `(real + dreal + gen) / (real + dreal + gen + rep)`: the share of presents that carried fresh content. Same definition as the HUD `fill`. |
| `slow` | Queued frames dropped because generation could not complete in time (capacity evidence for the degradation ladder). In a healthy run it stays small. |
| `disc` | Queued generated frames discarded without being presented (stale predictions after a cut, retargeting or shedding). |
| `adm` | Generation attempts skipped at admission because the next real frame was safely due, so no backup frame was needed. |
| `dreal_drop` | Queued delayed real frames dropped in bidirectional mode: the newest frame when a source faster than the display overflows the queue, or the excess of a persistent backlog outside a scene cut. Each is logged. Not a re-prime. |
| `reprime` | Declared pacing re-primes in the window: scene cuts, hitches, or the bidirectional delay shortening to its ring bound. |

### Timing and pacing

| Field | Meaning |
|---|---|
| `flip_ms_avg`, `_min`, `_max`, `_sd`, `_p95` | Statistics of the interval between consecutive flips, in ms. `flip_ms_sd` is the jitter figure. |
| `bias_ms` | Developer diagnostic. Display-chain timing bias learned by the scheduler and compensated automatically. |
| `dl_hit` | Fraction of generated and delayed-real presents that met their planned deadline. It says nothing about how many slots were filled; read `rep` and `fill` for that. |
| `dl_ms_avg`, `dl_ms_p95`, `dl_ms_worst` | Deadline error of presents against plan, in ms (the average is signed; the other two use the absolute error). |
| `lead_ms_min`, `lead_ms_avg`, `lead_ms_max` | Developer diagnostic. Commit lead of real frames on the native KMS path. |
| `lead_ema` | Developer diagnostic. Habitual commit-lead average the scheduler holds. |
| `lead_viable` | Developer diagnostic. Learned minimum viable native KMS commit lead, in ms. |
| `gen_late` | Generated frames that flipped more than half a refresh late. |
| `real_wait` | Times a real frame had to wait while a generated frame was being produced. |
| `src_flip_ms_p50`, `_p90`, `_p95`, `_fs` | Latency of live real frames from source-ready to the actual flip, in ms. `_fs` is the histogram full scale (256 ms); a percentile at full scale only says the value is beyond it. |
| `src_rec_ms_p50`, `src_rec_ms_p95`, `rec_flip_ms_p50`, `rec_flip_ms_p95` | The two parts of the `src_flip_ms_*` latency of live real frames, in ms. `src_rec_ms_*` runs from source-ready to the first paint (latch) of that frame; `rec_flip_ms_*` runs from the latch to the actual flip. Same full-scale rule as `src_flip_ms_*`. These are the last four fields on the line, and they are recorded with framegen off too (with `GAMESCOPE_FRAMEGEN_METRICS=1`). |
| `dreal_flip_ms_p50`, `_p90`, `_p95`, `_fs` | Latency of delayed real frames from their record to the actual flip, in ms (bidirectional mode only). Same full-scale rule. |
| `bidir_delay_ms` | Largest target delay D in force during the window, in bidirectional mode: how late real frames are shown relative to live. 0 outside bidirectional mode. How D is chosen: [Bidirectional delay controller](framegen-architecture.md#bidirectional-delay-controller). |
| `ring_ovf` | Bidirectional queue fronts that the output-ring ceiling forced out ahead of their planned slot in the window. Each is an overflow the ring bound learns from. 0 outside bidirectional mode. |

### Resets

| Field | Meaning |
|---|---|
| `resets` | Total pacing and history resets in the window. |
| `fbdisc` | Developer diagnostic. Display feedback records discarded because the frame was not presented. Expected to grow while presents are deliberately withheld. |
| `resets_cut` | Resets caused by scene cuts. |
| `resets_grid` | Developer diagnostic. Resets caused by a change of the display grid. |
| `resets_prov` | Developer diagnostic. Resets caused by a change of timestamp provenance. |
| `resets_hitch` | Resets caused by hitches (for example loading). |
| `resets_ring` | Emergency queue flushes. Nonzero means overload. |
| `resets_chain` | Developer diagnostic. Display-chain learner resets caused by a change of backend, connector, refresh interval, VRR state or timestamp provenance. |

### Staging copies (dual-GPU)

| Field | Meaning |
|---|---|
| `copy_bytes` | Bytes copied from the render GPU into present-GPU memory in the window. |
| `copy_ms_avg`, `copy_ms_max` | GPU time of those staging copies, in ms. Exactly 0 when nothing was measured (single-GPU route, or no timestamp support), never an estimate. |

### Quality meter (`q_*`)

A measurement-only meter. It never changes a generated frame. It runs for every
motion pipeline when `GAMESCOPE_FRAMEGEN_METRICS=1` or `--framegen-debug` is
set; on `learned` and above with adaptation on, the adaptation probe supplies
the same counters. With the meter on, each generation batch completes a little
later (about 20 microseconds per real frame on a Radeon 890M at 1440p), so
metrics runs carry that cost and production runs do not.

Each real frame grades the field that was estimated for it, by warping the
previous frame's block luma along the final field and comparing it with the real
frame. Only changed texels count: those whose content changed between the real
frames, where holding the previous frame would mispredict the real one, whatever
the field says. Older builds counted the texels the field calls moving instead,
so their `q_*` values are not comparable with current ones. A static background
predicts itself, and a near-zero vector is not motion evidence. Every pipeline
is measured before the learned pipelines' global field trust, which scales
confidence afterwards. Values are ratios of sums over the window, so every
changed texel weighs the same.

| Field | Meaning |
|---|---|
| `q_resid` | Mean normalized luma residual of the warp against the real frame. |
| `q_bad_mv` | Share of the changed texels that survived the checks whose vectors still mispredict the real frame (bad vectors that were kept). |
| `q_killed_mv` | Share of changed texels whose final confidence fell below the survive threshold. |
| `q_killed_match` | The part of `q_killed_mv` the matcher had already left unconfident. |
| `q_killed_fb` | The rest of `q_killed_mv`, cut by the forward-backward check. On `learned` and above it also includes confidence the refiner cut. |
| `q_samples` | Number of frames measured. With 0 (meter off, or no fresh field in the window) every `q_*` field prints 0. |

The `q_*` fields measure the content as much as the code: the same code can
show very different values on different games. Compare runs on the same
content. For gating and comparison tooling, see
[`framegen-testing.md`](framegen-testing.md).
