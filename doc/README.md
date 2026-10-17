# Documentation index

All documents describe the frame-generation work in this fork. Upstream gamescope usage is covered at the end of the [top-level README](../README.md).

Frame generation inserts extra frames between the ones a game draws, so motion looks smoother. Pick your path below. Not sure which one you are? Start with "I want to play" below.

## Start here

### I want to play

Follow these four steps in order. You can ignore everything outside this list until you want to tune.

1. **Build it.** [Top-level README](../README.md): read Requirements and Install. The build is the slow part; allow time for it.
2. **Prove it works with a test window.** Top-level README, [Quick start](../README.md#quick-start), steps 0 to 3. This opens a test window with the HUD on, so you can see it working.
3. **Run a real game.** Top-level README, [Quick start step 4](../README.md#step-4-use-it-with-a-real-game), then copy the command from [A good start for most games](../README.md#a-good-start-for-most-games).
4. **If something is wrong.** [User Guide](framegen-howto.md), [section 7, Troubleshooting](framegen-howto.md#7-troubleshooting).

The User Guide has a shorter [Quick start](framegen-howto.md#quick-start) that assumes you already built. The User Guide Quick start goes straight to the recommended `learned` command; the README one stops at a safe `checked` start first.

Extras, only when you need them:

- Hybrid laptop (iGPU plus dGPU): run [`scripts/framegen-doctor.sh`](../scripts/framegen-doctor.sh) first, then use the launcher described in [`scripts/README.md`](../scripts/README.md#hybrid-laptops-gamescope-hybridsh).
- Native mode (from a text console): run the doctor script as a preflight check, then launch with [`scripts/run-framegen-native.sh`](../scripts/run-framegen-native.sh). See the [native mode section](framegen-howto.md#native-mode-from-a-text-console).
- Steam, Lutris or Heroic: [User Guide, section 4](framegen-howto.md#4-playing-games).

### I want to tune

1. User Guide: [Modes and presets](framegen-howto.md#5-modes-and-presets), including the "Which one should I use?" table.
2. [Configuration Reference](framegen-reference.md): look things up here, do not read it end to end. Start at its [Glossary](framegen-reference.md#glossary) and the table of the eight variables most people need.
3. [`scripts/README.md`](../scripts/README.md): hybrid-laptop launcher and the showcase A/B tools.

### I want to understand or modify the code

Read in this order:

1. [Architecture](framegen-architecture.md), the "Reading order: the 30-second model" intro, then sections 0 to 2: scope, entry points and the dataflow of one real frame.
2. Architecture sections 3.1 and 4.1: what runs, and who decides.
3. One mechanism you care about: bidirectional interpolation (3.3), the degradation ladder (4.4), or the learned refiner (3.5).
4. [Maintenance](framegen-maintenance.md), "Source map": which file owns what, plus invariants and validation by change class.
5. [Testing](framegen-testing.md), "Overview and quick commands": the test harnesses and A/B tooling.

Build and run the contract tests. The README Install step configured `build/` with tests off, so turn them on in the same directory (`enable_tests` is a Meson option in [`meson_options.txt`](../meson_options.txt)):

```bash
meson configure build -Denable_tests=true && meson test -C build --print-errorlogs
```

Or use a separate directory: `meson setup build-tests -Denable_tests=true && meson test -C build-tests --print-errorlogs`. If Meson says `catch2-with-main` is missing, install your distribution's Catch2 package; it is needed only for the tests.

Where the code lives:

| What | Where |
|---|---|
| Scheduler (generate, dormant or pass through) | `framegen_record_real_frame` in `src/rendervulkan.cpp` |
| Present arbiter (real, generated or repeat) | `src/steamcompmgr.cpp` |
| Policy headers (pure logic) | `src/framegen/*.hpp` |
| Shaders | `src/shaders/cs_framegen_*.comp` |
| Tests | `tests/` |

### I want background

You can skip these. They explain how the design came to be and are not needed to use or change the code.

- [Research notes](research-framegen.md): frames-only frame-generation research. Not needed to play.
- [Design proposals](framegen-proposals/README.md): historical design records, notes 01-07.

## All documents

| Document | Audience | What it gives you |
|---|---|---|
| [framegen-howto.md](framegen-howto.md) | Players | User Guide: GPU setup, first run, choosing a mode, Steam and Lutris, HUD, troubleshooting, limits. |
| [framegen-reference.md](framegen-reference.md) | Players, developers | Configuration Reference: glossary, every CLI flag, environment variable, HUD row and metrics field. A lookup page; do not read it end to end. |
| [framegen-architecture.md](framegen-architecture.md) | Developers | How generation, scheduling, and presentation work. |
| [framegen-maintenance.md](framegen-maintenance.md) | Developers | Source map, invariants and what to validate for each kind of change. |
| [framegen-testing.md](framegen-testing.md) | Developers | Test harnesses, A/B tooling and recorded results. |
| [framegen-proposals/](framegen-proposals/README.md) | Developers | Design proposals 01-07: historical design records. |
| [../scripts/README.md](../scripts/README.md) | Players (first sections), developers | Players: read only "If something is wrong", "For players" and "Hybrid laptops". The rest is developer tooling. |
| [research-framegen.md](research-framegen.md) | Developers, curious readers | Background research. Not needed to play. |
