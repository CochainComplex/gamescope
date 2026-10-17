# Frame Generation Testing

How frame generation is tested at runtime: the nested smoke harness with
fault injection, the end-to-end showcase and baseline compare, how to judge a
setting by blind A/B (with the results so far), the latency gate, and the
offline evaluators and gates. This document states what each check proves and
what it cannot.

Audience: developers validating a framegen change and reviewers reading the evidence.
Related: [Documentation index](README.md) · [User guide](framegen-howto.md) · [Configuration reference](framegen-reference.md) · [Architecture](framegen-architecture.md) · [Maintenance](framegen-maintenance.md) · [Proposals](framegen-proposals/README.md)

## Contents

In file order. "Procedure" sections stay valid as the code changes. "Evidence" sections are dated results from specific games, builds and panels: read them as history, not as a contract.

- [Overview and quick commands](#overview-and-quick-commands) (procedure)
- [Smoke harness: setup and options](#smoke-harness-setup-and-options) (procedure)
- [Case catalogue](#case-catalogue) (procedure)
- [Fault-injection grammar](#fault-injection-grammar) (procedure)
- [Adding a fault-injection case](#adding-a-fault-injection-case) (procedure)
- [Scope limits](#scope-limits) (procedure)
- [Shutdown signal](#shutdown-signal) (procedure)
- [End-to-end showcase and baseline compare](#end-to-end-showcase-and-baseline-compare) (procedure)
- [How to judge a setting](#how-to-judge-a-setting) (procedure: blind A/B protocol)
- [Real-game A/B: Far Cry Primal](#real-game-ab-far-cry-primal) (evidence)
- [Blind A/B results (laptop panel, 2026-10-16/17)](#blind-ab-results-laptop-panel-2026-10-1617) (evidence)
- [Latency gate: framegen on vs off](#latency-gate-framegen-on-vs-off) (procedure)
- [Offline prediction-error split (S1/S2)](#offline-prediction-error-split-s1s2) (evidence)
- [Offline gates on captured content](#offline-gates-on-captured-content) (procedure)
- [Offline refiner and colour evaluators](#offline-refiner-and-colour-evaluators) (procedure)
- [Reference presets](#reference-presets) (evidence: benchmark configurations, not needed for normal play)

If you only want to run a test, skip the evidence sections. Some procedure sections
(the blind A/B protocol, the latency gate) end with the dated numbers that motivated
them; those numbers are examples, not thresholds.

## Overview and quick commands

If you just changed code, run these three commands from the repository root. They need a configured build tree `build` (the [README](../README.md) Install steps create it) and `GAMESCOPE_LOCAL_PREFIX` as described below.

```bash
meson test -C build --print-errorlogs             # contract tests
python3 scripts/framegen-nested-smoke.py baseline # nested smoke, one case
bash tests/gamescope_hybrid_test.sh               # only if you touched the hybrid launcher scripts
```

Which check proves what, and the checks for other kinds of change, are below.

| Tool | Question it answers | Entry point |
|---|---|---|
| Nested smoke harness | Do the present-wait worker, the SDL event worker, shutdown and generation survive injected faults? | `scripts/framegen-nested-smoke.py` |
| Showcase and compare | Does a candidate build regress any factor against a saved baseline on real content? | `scripts/framegen-showcase.sh` (summary in `scripts/framegen-showcase-report.py`) |
| Headless host | A private display for the showcase when the desktop would interfere | `scripts/framegen-headless-host.py` |
| Offline evaluators | Where does prediction error come from, and does a learned blob help? | `scripts/framegen-hindsight-eval.py`, `scripts/framegen-net-eval.py`, `scripts/framegen-color-eval.py` |
| Offline gates | Does a motion-field or output change help on captured frames, and does the numpy port still match the GPU? | `scripts/framegen-capture.sh`, `scripts/framegen-field-gate.py`, `scripts/framegen-temporal-gate.py` |

Which check for which change, in the order to run them:

| Change | Run | Needs |
|---|---|---|
| Any C++ or shader change | Contract tests: `meson test -C build --print-errorlogs` (see [Validation by change class](framegen-maintenance.md#validation-by-change-class)) | A configured build tree; `-Denable_tests=false` skips them |
| Hybrid launcher script | `bash tests/gamescope_hybrid_test.sh` (fixture test, not part of `meson test`; no GPU needed) | Nothing |
| Present-wait, SDL, shutdown, bidir timeline | Nested smoke harness | Built tree, `vkcube`, the dependency prefix (below); `bidir` also `gnome-shell` |
| Anything that can move pacing, latency or fill on real content | Showcase compare | Built tree and a saved baseline; GravityMark in `GM_DIR` (default `$HOME/GravityMark/bin`) or your own `CLIENT` |
| Motion-field or refiner change | Offline evaluators; add the offline gates for a field or output change | A capture and `numpy`; recording frames also needs `gst-launch-1.0` with `pipewiresrc` |
| A new default, or any field or output change | Blind A/B on all three content classes ([How to judge a setting](#how-to-judge-a-setting)) | The viewer, the same content for A and B, HUD off |

`GAMESCOPE_LOCAL_PREFIX` is the directory holding your locally built
dependencies (the prefix that `scripts/env-gamescope-local.sh` puts on the
library and pkg-config paths; its default is `$HOME/.local/gamescope-wayland`).
The smoke harness refuses to start without it: it exits with
`no local dependency prefix` unless `--prefix` or the variable is set, and the
directory must exist. The prefix only has to be an existing directory; the
harness uses `<prefix>/bin/vkcube` if present and otherwise `vkcube` from
`PATH`. If you have no local prefix because your system packages are new
enough, point it at any existing directory (for example `export
GAMESCOPE_LOCAL_PREFIX=$HOME`) and make sure `vkcube` is on `PATH`. If you
still need to build the dependencies or the tree, start with the Build
section of the [README](../README.md) and run `scripts/framegen-doctor.sh`.

Smoke harness (requirements and options are in the next section):

```bash
export GAMESCOPE_LOCAL_PREFIX=/path/to/prefix
python3 scripts/framegen-nested-smoke.py baseline      # one case
python3 scripts/framegen-nested-smoke.py all           # full matrix, PASS/FAIL evidence table
```

What a passing smoke run looks like:

- The harness prints `logs: <dir>` first, then the exact commands it runs. The
  log directory is `gamescope-framegen-smoke-*` under `$TMPDIR` (or `/tmp`)
  unless you pass `--logdir`.
- Run time is the hold time (18 s, or 27 s for `bidir`) plus startup and
  shutdown, so one case takes well under a minute.
- It ends with an evidence table and exits 0 only if every selected case
  passed (1 if any failed). The table has this shape (the evidence text is
  case-specific; `...` stands for it here):

```text
RESULT             STATUS  EVIDENCE
-----------------  ------  --------
baseline           PASS    ...
```

- On FAIL, each failed check is listed under its case with a `-` prefix, and the
  outer, inner and client logs are kept in the log directory. Look the failure
  string up in the [Case catalogue](#case-catalogue) (last column) to see what
  it usually means. The runtime directory `fgsmoke-*` under `/tmp` is only for
  sockets and is removed afterwards.

Showcase (details, settings and gates in
[End-to-end showcase and baseline compare](#end-to-end-showcase-and-baseline-compare)):

```bash
scripts/framegen-showcase.sh --list                    # the run table and settings
REPS=1 scripts/framegen-showcase.sh --baseline base.json 1 3   # fast first run: off and motion x2, one rep
scripts/framegen-showcase.sh --baseline base.json      # save a baseline: all 10 runs x 3 reps, about 30 minutes
scripts/framegen-showcase.sh --compare base.json       # judge a candidate: exit 0 pass, 1 regression, 2 not comparable
```

Use the same `W`, `H`, `HZ`, `RENDER_DEV`, `PRESENT_DEV` and `CLIENT` on
`--compare` as on `--baseline`, or the sessions are not comparable (exit 2).

## Smoke harness: setup and options

`scripts/framegen-nested-smoke.py` is a retained runtime smoke harness for the
Vulkan present-wait worker, the SDL event worker, and frame generation in a
nested compositor. It has six cases. It creates a private mode-0700
`XDG_RUNTIME_DIR` under `/tmp`, runs an outer headless gamescope, runs an inner
SDL gamescope at 640x480@60 with motion/checked x2 and a 30 fps cap, and runs
`vkcube` inside for 18 seconds by default. The `bidir` case replaces the
headless outer with a `gnome-shell --headless --wayland --virtual-monitor
640x480@60` host and the SDL inner with `--backend wayland`, and holds for 27
seconds by default. Logs go to a new directory under `$TMPDIR` (or `/tmp`)
unless `--logdir` is given. The private runtime directory is always a short
`fgsmoke-*` directory directly under `/tmp`, independent of `--logdir` and
`TMPDIR`, because the Wayland and X socket paths must fit the 108-byte
`sun_path` limit; its path is saved as `runtime-dir.txt` in the scenario log
directory, and the runtime directory is removed after the scenario.

Requirements:

- `--prefix` or `GAMESCOPE_LOCAL_PREFIX` points to your local dependency
  prefix, which the harness passes to `scripts/env-gamescope-local.sh`. The
  harness refuses to start without it.
- `vkcube` is in `<prefix>/bin` or on `PATH`.
- The gamescope binary and the WSI layer come from `<build-dir>` (default
  `build`, relative to the repository root).
- No special build option is needed: fault injection is a runtime environment
  knob that does nothing unless set.
- `bidir` only: `gnome-shell` (40 or later, with `--headless` and
  `--virtual-monitor`) and `dbus-run-session` on `PATH`. Without them `bidir`
  fails with a setup error, and the other cases still run.
- Optional: `gdb`, for stack capture when a process hangs; without it the
  harness records `/proc` task state instead.

Run one case (the later examples assume `GAMESCOPE_LOCAL_PREFIX` is exported):

```bash
GAMESCOPE_LOCAL_PREFIX=/path/to/prefix python3 scripts/framegen-nested-smoke.py baseline
python3 scripts/framegen-nested-smoke.py present-timeout --logdir /tmp/fg-smoke
python3 scripts/framegen-nested-smoke.py sdl-error --sdl-inject error:1
python3 scripts/framegen-nested-smoke.py sdl-error-pumped
python3 scripts/framegen-nested-smoke.py bidir
```

Run the full matrix and print a PASS/FAIL evidence table:

```bash
python3 scripts/framegen-nested-smoke.py all
python3 scripts/framegen-nested-smoke.py all --build-dir build-release
```

The harness pins the wrapper's `GAMESCOPE_REPO` and `GAMESCOPE_BUILD_DIR` to
this checkout and `--build-dir`, so the binary and the WSI layer always come
from the same build. An exported `GAMESCOPE_BUILD_DIR` is not used; pass
`--build-dir` instead.

Use `--hold-seconds N` to change the pre-SIGTERM run duration (default 27 for
`bidir`, 18 otherwise) and
`--sigterm-grace N` to change the clean-shutdown allowance (default 5). The
`--sdl-inject SPEC` and `--present-inject SPEC` flags replace a scenario's
default injection, which is useful for manually bisecting an injection count
or error. Supplying either override replaces the complete default, so only the
explicitly supplied injection kinds remain active; both override flags may be
given together. `--no-inject` clears the scenario default and cannot be
combined with an override. Every scenario header prints the effective present
and SDL specs, and the scenario is validated against them. A hold too short
to leave one complete metrics window after a case's reference marker fails
with `raise --hold-seconds`; it never passes unjudged.

The script prints the exact outer, inner, and client command lines before each
run. On failure it terminates the recorded process group and preserves the
outer, inner, and client logs. A missing prefix, binary, or `vkcube` is a
clear, non-zero preflight failure. `GAMESCOPE_ZENITY_DISABLE=1` prevents modal
error dialogs. `DISABLE_GAMESCOPE_WSI=1` applies only to the inner SDL
compositor; the client explicitly removes it so `vkcube` uses gamescope's WSI
integration. If shutdown fails, the summary's evidence block includes the
timestamped last inner-log line observed before SIGTERM and the final 15
inner-log lines.

Stack capture: when the inner or outer compositor is still running 3 seconds
into its shutdown wait (ahead of gamescope's own five-second shutdown
watchdog), or startup times out waiting for a pid file (the inner if it is
running, otherwise the outer), the harness saves every thread's stack to
`stacks-inner-<pid>.txt` or `stacks-outer-<pid>.txt` in the scenario log
directory and names the file in the error and in the shutdown evidence. It
uses `gdb -batch` when gdb is installed and can attach (20-second limit),
otherwise each thread's `comm`, `stat`, `wchan`, `syscall`, and `stack` from
`/proc`, with unreadable entries marked by their errno. Under Yama
`ptrace_scope` 1 the inner and outer processes declare the harness as their
ptracer, so only the harness and its descendants may attach. The capture's
time counts toward the shutdown time judged against `--sigterm-grace`.

## Case catalogue

How to read a result: find your case's row below, compare the evidence line from
the table with its PASS condition, and match any failure text against the last
column. The bullets after the table give the full checks for each case; you
only need them when a check fails or when you change the harness.

| Case | What it proves | PASS condition | Typical failure string |
|---|---|---|---|
| `baseline` | Generation runs at full rate in steady state | Common checks, plus every judged window has `gen > 0`, `slow/(real+dreal) <= 0.05`, `fill >= 0.95` | `framegen generation stayed at 0 after framegen enable`; a throughput failure naming the window and its `real`, `dreal`, `gen`, `slow`, `fill` |
| `present-timeout` | No present completion is fabricated while the driver wait times out | Windows wholly inside the stall have `real=0 gen=0 dreal=0`; recovery markers appear | A missing injection, stall or recovery marker; a non-zero count inside the stall |
| `present-error` | One out-of-date error costs one swapchain recreation and one history reset, then generation resumes | Exactly one recreation and one reset after the request; generation recovers | `framegen generation stayed at 0 after recovery`; a failure listing the reset reasons |
| `sdl-error` | The map gate keeps an unpumped SDL startup from wedging WSI | Injection and resume markers in order; presentation and generation recover | A missing `window mapped; presentation resumed` marker; `generation stayed at 0 after recovery` |
| `sdl-error-pumped` | The SDL error path is bounded and deduplicated while SDL keeps pumping | As `sdl-error` | As `sdl-error` |
| `bidir` | The delayed timeline stays within its logged delay D across a focus change | Every bidir check below; the harness-validity check | `harness invalid: ...`; `no settled window before\|after the focus change at Xs`; `N/N metrics windows lack bidir_delay_ms`; unseen, order, delay, spacing or gap failures |

Terms used in the table and bullets:

- `H`: `BIDIR_SETTLE_SECONDS`, 2 s, the settling allowance after the bidir
  marker or a transition.
- `D`: the target delay in force, read from `bidir_delay_ms`. `T`: one refresh,
  1/60 s.
- Window: one five-second `framegen-metrics:` line. Settled: judged for fill and
  slow. Unsettled: it overlaps a transition or H. Off cap: the client did not
  hold the 30 fps cap. Transition and drop excuse: see the bidir bullet.
- GMR-4: the internal tag of the persistent-present-wait-failure worst case
  under `present-error`.

- Every case rejects `abort`, `terminate`, and `corrupted` in the inner log,
  requires the inner compositor to exit with status zero within
  `--sigterm-grace` seconds of SIGTERM, and requires a clean-shutdown proof
  that shutdown ran rather than the process merely disappearing. The SDL cases
  require `present_wait: waiter joined`, which shows that shutdown joined the
  Vulkan present waiter. `bidir` requires `framegen-metrics: TOTAL`, logged by
  the frame-generation shutdown, because the Wayland backend has no Vulkan
  present waiter.
- Generation recovery. Periodic `framegen-metrics:` windows are five-second,
  grid-aligned counters. A window starts at its log time minus five seconds,
  and it is complete if it was logged at or before the last inner-log line
  read before SIGTERM.
  - From a case's reference marker, the window in progress at the marker is
    the settling window and is not judged. Every later complete window must
    have `gen > 0`, including the last one before SIGTERM, so generation must
    resume and must not stop again.
  - Otherwise the run fails with
    `framegen generation stayed at 0 after recovery` (`after framegen enable`
    for `baseline`).
  - Reference markers: `framegen: enabled mode=` for `baseline`;
    `present_wait: re-enabled after swapchain recreation` for
    `present-error`; the last `window mapped; presentation resumed` for the
    SDL cases. `baseline` starts at the enable marker rather than at process
    start because the window in progress at enable covers history priming,
    the first dispatches, and the client frame limiter settling, and every
    measured startup too-slow drop falls in that first second.
  - `real > 0` with `gen = 0` (typically `slow` equal to `real`) is the
    signature of the generation ladder stuck at its cheapest rung, discarding
    every generated frame as `generation_too_slow`. Presenting real frames
    alone does not prove recovery.
  - `gen > 0` is a liveness floor, not a rate: a window with only a few
    generated frames passes. The check proves that generation resumed and kept
    running, not that it ran at full rate, except in `baseline` and `bidir`,
    which also apply the throughput gate.
  - The 18-second default leaves two judged windows in `baseline` and after
    the SDL resume.
- `baseline` requires generation recovery from `framegen: enabled mode=`.
  - Throughput gate: every judged window must also have
    `slow/(real+dreal) <= MAX_SLOW_DROP_RATIO` (0.05) and `fill >= MIN_FILL`
    (0.95); a failure names the window and its `real`, `dreal`, `gen`, `slow`,
    and `fill`. The SDL inner presents no delayed reals, so `dreal` is 0 there.
  - `slow=` is the total `generation_too_slow` drop count. `fill` is computed
    from the parsed counters as `(real+dreal+gen)/(real+dreal+gen+rep)`, the
    definition of the binary's `fill=` and the HUD fill
    (`framegen_metrics_fill_rate()` in `src/rendervulkan.cpp`), but unrounded.
  - An ideal x2 run at 30 to 60 fps gives `real=150 gen=150 rep=0` per window,
    so the gate allows at most 7 too-slow drops per 150 real frames and about
    15 repeats per about 300 presents.
  - Why the gate exists: the `gen > 0` floor passed windows with 10-30% of the
    generated slots dropped (`slow=16..65` of about 150 real), which is
    visible judder.
  - Why both thresholds: `fill` catches every repeat source; for example,
    provisional-slot discards after an anchor correction gave `slow=6 rep=59`
    and fill 0.801. `slow/real` isolates the drops caused by late GPU work.
  - Other drop counters such as `slow_ev=` and `early=` are ignored and may be
    absent. A metrics line without `slow=` fails.
  - The fault cases keep only the floor, because their judged windows
    legitimately contain recovery transients.
- `present-timeout` sets
  `GAMESCOPE_DEBUG_PRESENT_WAIT_INJECT=timeout:40`. It requires the injection,
  stall, and recovery markers. The 250 ms slices keep the waiter stalled for
  roughly ten seconds. Every five-second metrics window lying wholly inside
  that interval must have `real=0 gen=0 dreal=0`, demonstrating that no present
  completion was fabricated while the driver wait was timing out. `rep` may be
  non-zero because hardware repeats are legitimate while completions are
  withheld, and growing `fbdisc` is also expected. Overlapping boundary windows
  are excluded because they may contain valid completions from outside the
  stall.
- `present-error` sets
  `GAMESCOPE_DEBUG_PRESENT_WAIT_INJECT=error:-1000001004:1`, injecting one
  `VK_ERROR_OUT_OF_DATE_KHR`. It requires a swapchain-recreation request,
  subsequent present-wait re-enable, and generation recovery from the
  re-enable marker.
  The request match requires `present_wait:` and `requesting swapchain
  recreation` on the same line, allowing the `VkResult` name or number between
  them.
  - Episode cost: with the default spec, exactly one
    `Creating Gamescope nested swapchain` and exactly one
    `framegen: reset history reason=` may be logged from the recreation request
    until SIGTERM; a failure lists the reset reasons. The count starts at the
    request, not at the injection marker, because the injection marker precedes
    the startup swapchain creations. It runs to SIGTERM, not to the re-enable
    marker, because a redundant reset can land after the waiter is re-enabled.
    An override spec skips this check.
  - Worst case (GMR-4): a persistently failing present wait is retried, not
    backed off. It costs one swapchain recreation and one framegen history
    reset (one re-prime) every `k_ulSwapchainRecoveryBudgetNs` (2 s) for as
    long as it fails.
- `sdl-error` sets `GAMESCOPE_DEBUG_SDL_WAIT_INJECT=error:30` and
  `sdl-error-pumped` sets `GAMESCOPE_DEBUG_SDL_WAIT_INJECT=error-pumped:30`.
  Both require the injection marker and between one and five
  `SDL_WaitEvent… failed: injected` lines during the first three seconds.
  Recovery is measured from `SDL event loop: fault injection exhausted` for a
  finite spec (that marker is then required after the injection marker), or
  from the last logged failure for an endless one: the last `window not mapped;
  presentation deferred` marker must be followed by a later `window mapped;
  presentation resumed` marker; the last resume must come within one second of
  the last failure; and a `framegen-metrics:` window whose start (log time
  minus five seconds) is at or after the last failure must have `real > 0`
  (presentation recovery); generation recovery is judged from the last resume
  marker.
  - For `sdl-error` this proves that the map gate keeps the unpumped startup
    interval from wedging WSI and that presentation and generation recover
    afterward.
  - For `sdl-error-pumped` it proves that a live nested window takes the error
    path at a deterministic 100 ms cadence even while SDL keeps pumping its
    Wayland connection; the log bound checks deduplication and back-off, and
    the later metrics windows check that presentation and generation recover
    once queued-event dispatch resumes.
- `bidir` validates the bidirectional (`GAMESCOPE_FRAMEGEN_BIDIR=1`) delayed
  timeline against its logged target delay D, across a focus-change
  invalidation.
  - Host: bidir needs an inner that owns its output ring, which the SDL
    backend never does, so the inner runs `--backend wayland`. gamescope's
    headless outer does not advertise `wl_subcompositor` or `wp_viewporter`,
    which the Wayland backend requires; the inner would silently fall back to
    SDL and log that `GAMESCOPE_FRAMEGEN_BIDIR` is ignored. The host is
    therefore `gnome-shell --headless --wayland --no-x11 --virtual-monitor
    640x480@60` under `dbus-run-session`. It gets a private bus listening in
    the runtime directory with no service directory (nothing is
    bus-activated), in-memory GSettings, no GVFS FUSE mount, and private XDG
    config, data, cache, and state directories under the scenario's state
    directory. Its output goes to `outer.log`. It is killed as a process group
    after the inner has exited.
  - The inner runs with `GAMESCOPE_FRAMEGEN_BIDIR=1` and
    `GAMESCOPE_FRAMEGEN_DEBUG_EVERY=1`, so every sampled bidir log line is
    printed. A second `vkcube` starts 11 seconds after the first and stays
    until shutdown.
  - `framegen: bidirectional interpolation requested` is required; without it
    the only error is that the inner did not run bidir. Exactly one
    `framegen: history valid=false reason=focus_change` must follow it: zero
    means the second client did not take focus, and more than one is listed.
  - Windows: fill and generation rate mean something only while the client
    runs at its cap, so they are judged per metrics window. A window is
    settled when it is complete, starts at least `BIDIR_SETTLE_SECONDS`
    (H, 2 s) after the bidir marker, ends before SIGTERM, and has no
    transition (see below) from H before its start to its end. A settled
    window is judged when the client held `--framerate-limit` 30 +-10% inside
    it. The client's rate is the sum of the id steps over the sum of the
    `time=` steps between consecutive `framegen: real frame id=N time=NS`
    lines of one history-reset segment, both recorded inside the window.
    - Each side of the focus change needs one judged window. A side with no
      settled window fails with
      `no settled window before|after the focus change at Xs`: too many
      transitions or too short a hold, which is a controller or harness
      fault. A side whose settled windows all ran off the cap fails with
      `harness invalid: no settled window before|after the focus change at Xs held --framerate-limit 30 +-10%`.
      This catches hosts where the nested Wayland backend's vblank clock runs
      ahead of the output (measured 42-56 fps against the 30 fps cap) and any
      regression that lets the client run past its cap.
    - An off-cap window is excluded from the fill and slow checks only;
      every other check judges the whole run at any cadence.
  - Generation: every judged window must have
    `slow <= 0.05 x (real+dreal)` and a fill of at least `MIN_FILL` (0.95):
    distinct new content (a real or generated frame, each distinct id
    counted once) per refresh, 5 s x 60 Hz. The fill is counted from the
    present lines, so a re-flip of a held frame never counts and the nested
    host's wake rate does not matter. At x2 and a 30 fps cap a fill of 0.95
    needs at least 135 generated frames per window, so `gen > 0` follows.
  - D is read from `bidir_delay_ms` on every periodic metrics line: the
    largest target delay in force during that window. T is one refresh
    (1/60 s). The bound for a span is the largest D of the windows that
    overlap it, plus T. If any window lacks the field, the run fails with
    `N/N metrics windows lack bidir_delay_ms` and the gates bounded by D (the
    prime, delay, and gap checks) are skipped. A window with `dreal > 0` and
    `bidir_delay_ms=0` is reported.
  - Two sets of lines from the log decide what is excused:
    - Transitions: a cut (`framegen: history valid=false reason=` or
      `framegen: reset history reason=`) and every timeline rebuild
      (`framegen: bidir ... re-primed`, `framegen: bidir timeline stale`, or
      `framegen: bidir timeline snapped to live`). A transition unsettles the
      windows it falls in or up to H before, and opens an H settling
      allowance for the delay and gap checks.
    - Drop excuses: a cut or a backlog drop
      (`framegen: bidir dropped N queued real(s) (backlog did not drain)`).
      Outside a cut, production drops queued real frames of a source no
      faster than the display only for a backlog whose lateness neither
      shrank nor grew for a second, and then only the real frames of its
      excess; queue capacity alone never drops them. A delay re-prime keeps
      the queued real frames; a shorter delay moves them earlier. Real
      frames dropped by any of the following outside a cut count as unseen:
      - a stale abandon;
      - a snap to live;
      - a replaced queued base
        (`framegen: bidir replaced queued base behind`);
      - a ring-full drop
        (`framegen: bidir dropped N queued real(s) (ring full)`), which only a
        source faster than the display causes and the capped client never
        does;
      - a target collision;
      - a failed present image.
  - Real frames are judged from H after the bidir marker to the end of the
    last metrics window before SIGTERM. Ids are keyed by history-reset
    segment, because frame ids restart at a reset. A real frame is shown by
    its first `framegen: presented delayed real frame id=N.100` or
    `framegen: presented live real frame id=N.100` line, which is logged at
    the compositor opportunity that consumes it; its delay runs from its
    `framegen: real frame` line to that present. Log times carry 1 ms
    resolution and up to 1.4 ms of log-pump lag, so each time comparison
    allows 2 ms (`LOG_SLACK_NS`). The checks:
    - Rebuilds: at most `BIDIR_MAX_REPRIMES` (1) stray rebuilds per run. A
      rebuild is stray when it is logged after H from the bidir marker
      (warm-up) and outside a cut's own priming, which runs from the cut to
      the first real frame presented after it. `reprime=` on the `TOTAL`
      line may not exceed the number of re-prime lines, so every kind of
      re-prime must be logged. One sustained change re-establishes the
      timeline once, so a delay that shortens and lengthens back across one
      transient counts as two rebuilds.
    - Unseen: a real frame that was never shown although a later one was,
      with no drop excuse between its record and that later present, is
      unseen. Any unseen real frame fails the run.
    - Order: within a reset segment every real frame, delayed or live, must
      be presented after every lower id and never again. Any violation
      fails, with the count and the first three offenders.
    - Delay: every shown real frame with no transition from H before its
      record to its present must be shown within D + T of its record. The
      error gives the count and the three worst. There must be at least one
      such frame.
    - Spacing: take consecutive real frames of one reset segment that are
      both shown, with no transition from H before the earlier one's record
      to the later one's present. Their present spacing must match their
      `time=` spacing within T plus `LOG_SLACK_NS` for at least
      `BIDIR_MIN_EVEN_SPACING` (99%) of those pairs.
      - The error gives the count and the three worst offsets.
      - There must be at least one such pair.
      - Production places every real frame at its own record time plus D,
        so the record spacing is the reference.
      - How far the remaining 1% may stray is bounded by the delay and gap
        checks.
    - Gaps: consecutive presents of new content, including from the start
      and to the end of the span, may be at most D + T apart. A gap that
      contains a transition may be 2(D + T): up to D + T since the last
      present before it, then up to D + T from the prime's record to its
      present, which holds because the prime records at the cut.
  - After the focus change: the first
    `framegen: priming history with real frame id=ID`, then the first real
    frame present (delayed or live) logged after it with an id of at least
    `ID` (the prime, or a newer base that replaced it while queued), within
    D + T of the priming line. No `framegen: bidir timeline snapped to live`
    may appear from the focus change until SIGTERM. Together these prove
    that the prime did not discard the delayed queue and was shown in
    bounded time.
  - The summary prints the judged and settled window counts; one row per
    metrics window after the bidir marker with its end, client fps, fill,
    `gen`, `slow`, D, and the peak `envelope_ms` of its
    `framegen: bidir epoch=` lines, marked `unsettled` or `off cap` when it
    is not judged; stray and total rebuilds; unseen and judged real frames;
    order violations; uneven spacing pairs over judged pairs with the worst
    offset; delay min/p50/p90/max; the gap count; and the prime.
  - The default 27-second hold leaves one settled window before the focus
    change and two after it. The intended bar is every check passing in three
    of three runs, but the case currently fails about one run in three on a
    single real frame shown about 4.7 ms past `bidir_delay_ms + T` (bug-178;
    commit 16862c2 shows the same, so it predates the current changes). Treat that one
    signature as a known flake; any other failure is real. A judged window
    below the fill floor is a controller defect; the floor stays.
  - Re-judging a saved bidir log:

    ```bash
    python3 scripts/framegen-nested-smoke.py bidir --check-log <logdir>/bidir/inner.log
    ```

    This applies the same bidir gates offline, the validity check included.
    It takes the `framegen-metrics: TOTAL` line as SIGTERM and does not
    re-check the exit status or shutdown timing.
- `all` runs those six cases sequentially and returns non-zero if any setup,
  lifecycle, or case-specific assertion fails.

The metrics parser reads the leading `real=`, `gen=`, `dreal=`, and `rep=`
fields of periodic `framegen-metrics:` lines and skips the `total` line; the
format itself lives in `src/rendervulkan.cpp`. The throughput gate also reads
`slow=` by name, anywhere on the line; `fill` is derived from the four leading
counters, not parsed. The bidir gates read `bidir_delay_ms` by name from every
periodic line and `reprime=` from the `TOTAL` line.

The latency families `src_flip_ms_*` (live real frames, from source ready to
the actual flip) and `dreal_flip_ms_*` (delayed real frames, from their record
to the actual flip) are metrics only for this harness: none of its gates reads
them (the showcase compare gates their medians, see
[End-to-end showcase and baseline compare](#end-to-end-showcase-and-baseline-compare)).
Each has `_p50`, `_p90`, `_p95`, and the histogram's full scale `_fs` (256 ms); a
percentile at full scale only says the value is somewhere beyond it. An empty
distribution prints 0.000. `src_rec_ms_*` and `rec_flip_ms_*` split
`src_flip_ms_*` at the latch ([Latency gate](#latency-gate-framegen-on-vs-off)).

`ring_ovf` counts the bidir queue fronts that the output-ring ceiling forced
out ahead of their planned slot in the window. Each one is an overflow that
the bidir ring bound (`D_ring`) learns from; it stays 0 outside bidir.

The `q_*` fields come from the quality meter. It only measures and never
changes a generated frame. It runs for every motion pipeline when
`GAMESCOPE_FRAMEGEN_METRICS=1` or `--framegen-debug` is set; on `learned` and
above with adaptation on, the adaptation probe supplies the same counters. The
meter is recorded in the generation batch, so with it on every batch completes
a little later (about 20 µs per real frame on a Radeon 890M at 1440p); metrics
runs carry that cost and production runs do not.
Each real frame grades the field that was estimated for it by warping the
previous frame's block luma along the final field and comparing it with the
real frame. Only changed texels count: where holding the previous frame would
mispredict the real one (the same normalized residual and threshold as
`q_bad_mv`), whatever the field says. A static background predicts itself, and
a vector near zero is not motion evidence. Older builds counted the texels the
field calls moving instead, so their `q_*` values are not comparable. Values are
ratios of sums over the window, so every changed texel weighs the same:

- `q_resid`: mean normalized luma residual of the warp against the real frame.
- `q_bad_mv`: share of the changed texels that survived the checks whose
  residual still exceeds the mispredict threshold (bad vectors that were kept).
- `q_killed_mv`: share of changed texels whose final confidence fell below the
  survive threshold.
- `q_killed_match`: the part of `q_killed_mv` the matcher had already left
  unconfident.
- `q_killed_fb`: the rest of `q_killed_mv`, cut by the forward-backward
  consistency check. On `learned` and above it also includes confidence the
  refiner cut.
- `q_samples`: the number of frames measured. With 0 (meter off, or no fresh
  field in the window) every `q_*` ratio prints 0.

These are five-second window counters. `real`, `gen`, and `dreal` count
presented fresh content, while `rep` counts repeated scanout. A window with
zero `real`, `gen`, and `dreal` inside the injected timeout therefore provides
the completion-publication check without treating legitimate hardware repeats
or frame-generation dispatch logs as presentation proof.

## Fault-injection grammar

Both knobs share the strict grammar of `src/Utils/FaultInjection.h`:

- `kind` must match the whole token.
- `N` is a positive decimal count with no sign, whitespace, or trailing text;
  `0` is rejected. Without `N` the injection never ends.
- An invalid spec is logged and ignored, so the run has no injection. An
  active spec is logged once.

Only the markers quoted under [Case catalogue](#case-catalogue) are parsed; other log
wording may change.

`GAMESCOPE_DEBUG_PRESENT_WAIT_INJECT`:

| Spec | Exact behavior |
| --- | --- |
| `timeout[:N]` | The next `N` waits return `VK_TIMEOUT` without calling the driver. Each backs off one 250 ms slice, so `timeout:40` stalls the waiter for about ten seconds. |
| `error:CODE[:N]` | The next `N` waits return `VkResult` `CODE`, a negative decimal int32 such as `-1000001004` (`VK_ERROR_OUT_OF_DATE_KHR`). Zero and positive codes are rejected because they would fabricate completions. |

Finite budgets span swapchain recreation.

`GAMESCOPE_DEBUG_SDL_WAIT_INJECT`:

| Spec | Exact behavior |
| --- | --- |
| `error[:N]` | For `N` consecutive event-loop iterations (or forever when `N` is omitted), skip both `SDL_PumpEvents()` and `SDL_WaitEventTimeout()`, set the SDL error to `injected`, and return zero into the error path. Queued events are neither pumped nor dispatched. |
| `error-pumped[:N]` | For `N` consecutive event-loop iterations (or forever when `N` is omitted), call `SDL_PumpEvents()` to service the connection and queue incoming events, skip `SDL_WaitEventTimeout()`, set the SDL error to `injected`, and return zero into the error path. Queued events, including `GAMESCOPE_SDL_EVENT_VISIBLE`, are not dispatched until injection ends. |

Both modes therefore exercise the same error handling, time-bounded log
deduplication, exit checks, and 100 ms retry back-off.

## Adding a fault-injection case

Add the subcommand name to `SCENARIOS`, return its environment value from
`scenario_injection()`, and add its marker ordering and metrics assertions to
`validate_scenario()`. Keep injection opt-in and remove inherited
`GAMESCOPE_FRAMEGEN*` and fault variables before launching the outer
compositor. A new C++ knob parses its spec with `ParseFaultInjection` or
`ParseFaultInjectionCode` from `src/Utils/FaultInjection.h`, never its own
parser. Prefer stable observer or lifecycle markers over driver-specific
messages. If timing matters, use the elapsed-time prefix written to
`inner.log`; do not infer a busy-loop bound from total log count over the whole
run. A recovery case judges generation with `generation_errors()` from its
recovery marker; the throughput gate applies to `baseline` and to `bidir`
(without `fill`). The clean-shutdown marker depends on the backend: SDL cases
use `present_wait: waiter joined`, Wayland-backend cases
`framegen-metrics: TOTAL`. Finally, include the case in `all` and document the
exact behavior it proves.

## Scope limits

This is a nested smoke test, not display-quality certification. It cannot
validate native DRM/KMS or VRR pacing, HDR output, real-game behavior, or
multi-GPU import and synchronization. It also does not assess image quality,
latency by instrument, or ReShade reload/unload behavior. The `baseline`
throughput gate checks steady-state nested-SDL throughput; it is not native
pacing certification, and it does not judge the warm-up window at enable.

`bidir` runs nested Wayland under gnome-shell's headless virtual monitor, not
native DRM/KMS, so it does not certify native pacing. It does not cover the
overlay-toggle (RefreshLive) check, the double-present check, ring-pressure
release, VRR, or HDR, and it runs with `GAMESCOPE_FRAMEGEN_DEBUG_EVERY=1`
logging, which changes timing. Run the native and hardware-specific validation
separately before making those claims.

The bidir gates have known limits. The delay is measured from the record, so a
regression that delays the record itself is not caught. The bound for a span
is the largest D of every window that overlaps it, so it is lenient while D
falls within a window. The 2(D + T) allowance across a transition assumes the
prime records at the cut. The fill floor assumes x2 at a source interval of two
refreshes: cap jitter into three-refresh intervals leaves a slot that x2 cannot
fill; a change to the multiplier or the cap needs the fill floor revisited.
Off-cap and unsettled windows are left out of the fill and slow checks, so a
regression confined to them shows only in the per-window rows. `H`
(`BIDIR_SETTLE_SECONDS`) must cover production's envelope seed window
(`k_ulBidirWindowNs`). H is longer than the one second after a transition that
the spacing invariant allows, so the gate does not judge that second strictly.

## Shutdown signal

This is a measurement record that explains why the harness signals the pid and not the group; it is not needed to run the harness.

The harness sends its measured SIGTERM to the inner gamescope pid only, the way
a session manager does (`kill -TERM $pid`, or `timeout --foreground`). The
inner wrapper runs the binary directly and gamescope never calls `setsid` or
`setpgid`, so that pid is gamescope itself. The process-group kill runs only as
cleanup afterwards. Signalling the whole group delivers SIGTERM to Xwayland
concurrently with gamescope, and Xwayland's exit can trip upstream's fatal
`xwm: X11 I/O error! This is fatal. Aborting...` handler.

Measured on 2026-10-14 with debug builds of upstream `a563226` (built from a
`git archive` of that commit plus its pinned submodules) and of this release
candidate. Framegen was off (`GAMESCOPE_FRAMEGEN*` unset); gamescope ran
headless at 1280x720@60 with `xlogo`, was signalled after 4 seconds, and its
own exit status was collected with `wait`. "pid" sends SIGTERM to gamescope
alone; "group" runs it as a `set -m` job and signals the job's process group.

| Build | GPU | Signal | Runs | `X11 I/O error` aborts (rc 134) | Other crashes | Runtime dir left non-empty |
| --- | --- | --- | --- | --- | --- | --- |
| a563226 | NVIDIA RTX PRO 500 | pid | 10 | 0 | 10 (rc 139, bug-064) | 10 |
| a563226 | NVIDIA RTX PRO 500 | group | 10 | 2 | 8 (rc 139, bug-064) | 10 |
| a563226 | AMD 890M | pid | 10 | 0 | 0 | 0 |
| a563226 | AMD 890M | group | 10 | 2 | 0 | 2 |
| release candidate | NVIDIA RTX PRO 500 | pid | 30 | 0 | 0 | 0 |
| release candidate | NVIDIA RTX PRO 500 | group | 30 | 7 | 0 | 7 |
| release candidate | AMD 890M | pid | 30 | 0 | 0 | 0 |
| release candidate | AMD 890M | group | 30 | 6 | 0 | 6 |

The AMD rows selected the GPU with `--prefer-vk-device 1002:150e` (the 890M), the NVIDIA
rows with `10de:2db9`; each log confirms the selected
device. "bug-064" is the known NVIDIA SIGSEGV from static Vulkan destructors at
exit (exit status 139 about 1.2 s after `Primary child shut down!`), which this
fork avoids by exiting without running them; it is unrelated to the X11 I/O
abort and is counted separately. A zero is a count over the stated runs, not
proof of absence. The X11 I/O abort occurs only under group signalling, in
upstream and in the release candidate at similar rates (4 in 20 and 13 in 60),
and never under pid-only signalling (0 in 20 and 0 in 60).

One cell's loop, run from the repository root (build tree `build`, and `GAMESCOPE_LOCAL_PREFIX` set as for the smoke harness):

```bash
for i in $(seq 10); do
  rd=$(mktemp -d -p /tmp fgtXXXX)
  env -u DISPLAY -u WAYLAND_DISPLAY XDG_RUNTIME_DIR=$rd GAMESCOPE_ZENITY_DISABLE=1 \
    ./scripts/env-gamescope-local.sh build/src/gamescope --prefer-vk-device <vid:did> \
    --backend headless -W 1280 -H 720 -r 60 -- xlogo > term-$i.log 2>&1 &
  pid=$!; sleep 4
  kill -TERM $pid
  wait $pid; echo "run $i rc=$?"; rm -rf "$rd"
done
grep -l 'X11 I/O error' term-*.log
```

Replace `<vid:did>` with the present GPU's id (`scripts/test-framegen.sh gpus`
lists them). Logs `term-*.log` are written to the current directory. Group
mode: run `set -m` before the loop and replace the `kill` line with
`kill -TERM -- -$pid`.

## End-to-end showcase and baseline compare

`scripts/framegen-showcase.sh` runs a fixed table of ten framegen
configurations (framegen off, predictive, bidir, each pipeline, base layer)
against one client, with the HUD and `GAMESCOPE_FRAMEGEN_METRICS=1` on, and judges a
candidate against a saved baseline. It is the per-change gate: a change passes
when no factor regresses on any combination it touches, and an output change
stays opt-in until such a compare shows a win on its factor with no regression
on any other.

```text
framegen-showcase.sh [--list] [--baseline FILE] [--compare FILE] [--logs DIR] [RUN...]
  --list            print the run table and settings, then exit
  --baseline FILE   also save this session's results.json as FILE
  --compare FILE    judge this session against FILE: exit 0 pass, 1 regression, 2 not comparable
  --logs DIR        run nothing; summarise (and compare) the run*-r*.log already in DIR
  RUN...            table indices to run (default: all)
```

Setup and usage errors exit 2. The summary and the compare run in
`scripts/framegen-showcase-report.py` (`python3`, standard library only).
`GAMESCOPE_LOCAL_PREFIX` is passed to `scripts/env-gamescope-local.sh` as for
the smoke harness.

Run table (`--list` prints it with the effective settings):

| Index | Name | Flags and environment |
|---|---|---|
| 1 | off | no framegen |
| 2 | extrapolate x2 | `--framegen-mode extrapolate`, multiplier 2 |
| 3 | motion x2 | `--framegen-mode motion`, multiplier 2 |
| 4 | motion x3 | motion, multiplier 3 |
| 5 | bidir x2 | motion x2 with `GAMESCOPE_FRAMEGEN_BIDIR=1` |
| 6 | best low-latency | motion x2, `--framegen-pipeline checked` |
| 7 | best quality | checked pipeline with `GAMESCOPE_FRAMEGEN_BIDIR=1` |
| 8 | learned x2 + net | `--framegen-pipeline learned` with `GAMESCOPE_FRAMEGEN_NET_ONLINE=1 GAMESCOPE_FRAMEGEN_NET_EVERY=2` |
| 9 | guided x2 | `--framegen-pipeline guided` |
| 10 | base layer | checked pipeline with `GAMESCOPE_FRAMEGEN_BASE=1` |

The examples below select runs by these indices. The authoritative table is
the `RUNS` array in `scripts/framegen-showcase.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `REPS` | 3 | Whole repetitions, run rep-major (every selected run once, then again), so thermal and background drift spread over all runs. |
| `SECONDS_PER_RUN` | 45 | Run length; gamescope gets 5 s more. Must be at least 5 × (`WARMUP_WINDOWS` + 2). |
| `WARMUP_WINDOWS` | 1 | Leading 5 s metrics windows treated as client startup. Raise it for a game with a loading screen. |
| `BACKEND` | `wayland` | Gamescope backend; `drm` from a text VT for the native verdict. |
| `W`, `H`, `HZ` | 2560, 1440, 120 | Output size and refresh rate. |
| `RENDER_DEV` | empty | The client's `MESA_VK_DEVICE_SELECT` (`vid:did[!]`). Empty: the client's default device. |
| `PRESENT_DEV` | empty | Gamescope's `--prefer-vk-device`. Empty: gamescope's default device. |
| `CLIENT` | GravityMark | Shell command run inside gamescope through `sh -c`. The default runs `GravityMark.x64` from `GM_DIR` with `ASTEROIDS` asteroids at `W`x`H`. |
| `ASTEROIDS` | 500000 | Asteroid count of the default GravityMark client. |
| `GM_DIR` | `$HOME/GravityMark/bin` | GravityMark's `bin` directory for the default client. |
| `CLIENT_PROC` | `GravityMark.x64` without `CLIENT`, else empty | The client's kernel process name (its first 15 characters), used by the render device check. |
| `GAMESCOPE_BIN` | `build/src/gamescope` | Binary under test. Its build directory is passed as `GAMESCOPE_BUILD_DIR`, so the WSI layer and the reaper come from the same build. |
| `OUT` | `$XDG_RUNTIME_DIR/framegen-showcase/<date-time>` | Session directory: `run<i>-r<rep>.log` and `results.json`. |
| `DEBUG` | 0 | `1` adds `--framegen-debug` to framegen runs. Diagnosis only; refused together with `--baseline` or `--compare`. |
| `EXTRA_ENV` | empty | Extra gamescope environment for every run. |
| `HOST_DISPLAY` | unset | The host's modes, for a caller that nests in its own host; skips the xrandr probe and the locked-session check. `scripts/framegen-headless-host.py` sets it to `headless <mode>`. |

Each log starts with `framegen-showcase: key=value` lines: `run`, `index`,
`rep`, `kind`, `flags`, `run_env`, `backend`, `size`, `hz`, `render`,
`present`, `client`, `warmup`, `extra_env`, `debug`, `seconds` and `bin`.
The driver appends `render_gpu=` (`seen`, `not_seen` or `unverified`) once the
first metrics window appears, and `exit=` with the `timeout` status at the end.
The render check looks for the client process (`CLIENT_PROC`) in the run's own
process tree. It uses DRM fdinfo engine time on `RENDER_DEV` when the device's
driver reports it, and the driver's process list (`nvidia-smi`, by pid) when it
does not. Cleanup signals only processes from the run's own process tree.

Aggregation:

- Only windows with `w` greater than `warmup` count. The `TOTAL` line is never
  used: it includes client startup.
- Frame counts are summed over the steady windows. `real/s` and `shown/s` divide
  by 5 s per window. `fill` is fresh / (fresh + repeated) over the summed
  frames, the same definition the compositor uses per window.
- `jit_ms` is the pooled population standard deviation of the flip interval,
  from each window's `flip_ms_avg` and `flip_ms_sd`, weighted by its presented
  frames.
- `lat_p50` is the median over windows of `src_flip_ms_p50` (live reals,
  from source-ready to flip), or of `dreal_flip_ms_p50` (delayed reals, from the
  record, marked `d`) when a run shows more delayed than live reals. One start
  point is chosen per run; windows without samples (0) are skipped.
- `lat_p95` is the same median over the p95 fields (`src_flip_ms_p95`, or
  `dreal_flip_ms_p95` for a run that shows delayed reals), with the same start
  point. It is the tail, which is what reads as lag, so it has its own gate
  (below).
- `dl_hit` and `bias_ms` are medians over windows that generated or delayed a
  frame; framegen off shows `-`. `reprime`, `ring_ovf` and `q_samples` are
  summed. Each `q_*` is the median over windows with `q_samples` > 0.
- The run's value is the median over its reps that produced a steady window.

A rep is complete when `exit=124` (the run lasted its full length) and it has a
steady window. A missing field is shown as `-` (table) or `n/a` (compare): older
binaries do not print `q_*` or `ring_ovf`. A gated field the baseline measured
but the candidate lacks makes the sessions not comparable.

Compare gates, per run present in both sessions:

| Factor | Regression when the candidate |
|---|---|
| `real_s` | falls by more than 10% of the baseline (the game itself got slower) |
| `fill` | falls by more than 0.01 |
| `flip_ms_sd` (pooled) | rises by more than 0.5 ms |
| `lat_p50` | rises by more than 1 ms |
| `lat_p95` | rises by more than 2 ms |
| completeness | has an incomplete rep where the baseline run had none |

`shown_s`, `dl_hit`, `bias_ms`, `reprime`, `ring_ovf`, the `q_*`
fields and `q_samples` are reported, not gated. The `q_*` fields measure the
content as much as the code: with identical code, GravityMark's `q_killed_mv`
spans 0.63-0.97 between reps, so judge quality on recorded frames instead. Exit 0 is a pass, 1 a regression. Exit 2 means the
sessions are not comparable: a context key differs (`backend`, `size`, `hz`,
`render`, `present`, `client`, `seconds`, `warmup`, `debug`, and `host_display`
when both sides recorded it, since a nested run paces against its host's
refresh), either side is a DEBUG session, a client was not seen on its render
device, the sessions have no run in common, a run's flags, run environment or
latency start point differ, or the candidate lacks a gated field the baseline
measured (a regression found elsewhere still exits 1). The treatment (`bin`,
gamescope `version`, `EXTRA_ENV`) is printed for both sides and never compared: it is
what is under test, so an opt-in knob can be judged with one binary. One
session holds one treatment; summarising logs of mixed binaries is refused.

`max_ms` (`flip_max`) is the longest single flip interval of the run's steady
windows: a stutter that the pooled `flip_ms_sd` averages away. It is reported,
not gated.

Background runs: `scripts/framegen-headless-host.py` runs a command nested in a
private headless GNOME Shell (one virtual monitor, private bus and runtime
directory) and kills its whole process group on exit. The showcase then shows
no window, and the desktop's monitors, mouse and lock state do not touch it; it
records the host as `host_display=headless <mode>`. The host renders on the GPU
its EGL picks, which must be the present device, or it drops gamescope's
connection (on a hybrid machine whose dGPU EGL loads first, set
`__EGL_VENDOR_LIBRARY_FILENAMES` to the iGPU vendor's file):

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
  scripts/framegen-headless-host.py --mode 2560x1440@60 -- \
  env HZ=60 scripts/framegen-showcase.sh --baseline a.json 6 7
```

The host's private runtime directory is deleted on exit, so `OUT` defaults to
`$XDG_RUNTIME_DIR/framegen-showcase/<time>` of the calling session, outside it,
and the results survive; an explicit `OUT` is kept. SIGTERM (`kill`, `timeout`)
takes the same cleanup path as Ctrl-C and kills the host's and the command's
process groups.

Examples (`--compare` needs the baseline's context, so repeat its variables, and the numbers are run-table indices):

```bash
# Baseline from one build, then a candidate from another.
GAMESCOPE_BIN=../build-base/src/gamescope scripts/framegen-showcase.sh --baseline base.json
GAMESCOPE_BIN=build/src/gamescope scripts/framegen-showcase.sh --compare base.json

# Opt-in A/B on one binary: the knob is the treatment (adaptation needs the
# learned pipeline, run 8).
scripts/framegen-showcase.sh --baseline a.json 8
EXTRA_ENV=GAMESCOPE_FRAMEGEN_ADAPT=0 scripts/framegen-showcase.sh --compare a.json 8

# Hybrid: render on the dGPU, composite, generate and present on the iGPU.
RENDER_DEV='<dgpu vid:did>!' PRESENT_DEV='<igpu vid:did>' scripts/framegen-showcase.sh --baseline hybrid.json

# Single GPU, at 60 Hz.
RENDER_DEV='<gpu vid:did>!' PRESENT_DEV='<gpu vid:did>' HZ=60 scripts/framegen-showcase.sh --baseline single60.json

# A game with a loading screen instead of GravityMark.
CLIENT='exec /path/to/game' CLIENT_PROC=game.exe WARMUP_WINDOWS=6 SECONDS_PER_RUN=90 \
	scripts/framegen-showcase.sh 1 3 6

# Native verdict from a text VT.
BACKEND=drm scripts/framegen-showcase.sh --compare native-base.json

# Re-judge saved logs without running anything.
scripts/framegen-showcase.sh --logs "$XDG_RUNTIME_DIR/framegen-showcase/<session>" --compare base.json
```

A full A/B covers two contents, two topologies and two refresh rates, one
compare per combination:

```bash
for content in gm game; do
	for topo in hybrid single; do
		for hz in 60 120; do
			# set CLIENT/CLIENT_PROC for $content and RENDER_DEV/PRESENT_DEV for $topo
			HZ=$hz OUT=ab/$content-$topo-$hz scripts/framegen-showcase.sh \
				--compare base/$content-$topo-$hz.json
		done
	done
done
```

The full table takes about 10 runs × 3 reps × 57 s, roughly 30 minutes per
combination; select runs by index for a focused check.

Limits:

- Nested backends carry the parent compositor's frame and jitter. Nested
  results never decide jitter; the final verdict is a native DRM run.
- DEBUG numbers are never comparable: per-frame logging perturbs pacing.
- Latency percentiles are the upper edges of 1 ms buckets, so the +1 ms gate on
  `lat_p50` tolerates a one-bucket shift, the +2 ms gate on `lat_p95` a
  two-bucket shift, and a sub-millisecond regression is invisible.
- The `off` run checks pacing and latency parity only, not idle CPU or timer
  wake-ups.
- A baseline binary loads this checkout's Lua scripts (`scripts/`), not its own.
- There is no HDR content gate: the default content is SDR. Run an HDR client
  through `CLIENT` to cover it.
- The NVIDIA render check matches the client by pid, not by GPU; with two GPUs
  of the same `vid:did`, either counts. The check runs once, at the first
  metrics window.
- `CLIENT_PROC` is the 15-character kernel process name, not the full name.
- `timeout` reports 124 for a run that lasted its full length even when
  gamescope then crashed during shutdown; shutdown is the smoke suite's job.

## How to judge a setting

No single number ranks a setting. What the viewer sees is the product of five
factors, and each has a limit past which it spoils everything else:

| Factor | What it is | Where to look |
|---|---|---|
| Real-frame latency | How long a real frame takes to reach the screen; judged on the p95 tail, not the median | Showcase `lat_p95` and `lat_p50` |
| Frame cadence | How evenly frames flip on the display (flip timing) | Showcase `jit_ms` and `max_ms` |
| Motion continuity | Whether the content advances smoothly; rejected or held motion reads as choppy | By eye |
| Image integrity | No invented structure | By eye |
| Sharpness | No blur | By eye |

Inside the limits the priority is latency first, then cadence, then the rest by
eye. Late-frame counts did not decide in practice. Far Cry Primal is the example
([Real-game A/B](#real-game-ab-far-cry-primal)): bidirectional x2 had the lowest
jitter and about 30 ms more latency, and the player ranked it last.

Every knob trades one factor for another, so name the trade before judging the
result:

| Knob | Trades | Seen so far |
|---|---|---|
| Motion rejection threshold | Image integrity against motion continuity | Layers that reject or alter motion felt choppy: the vector checks of `checked`, the acceleration of `predict` |
| Field resolution (`GAMESCOPE_FRAMEGEN_FIELD_FINE`) | Detail against robustness on swarms | The /4 field looked crisper on Xonotic and blocky on GravityMark |
| Fallback shape (`GAMESCOPE_FRAMEGEN_SHARP_FALLBACK`) | Sharpness against edge stability | The sharp fallback frays edges on dense particle swarms; the crossfade it replaced left double edges and blur |
| Bidirectional / run-ahead (`GAMESCOPE_FRAMEGEN_BIDIR`, `GAMESCOPE_FRAMEGEN_BATCH_REPAINT`) | Cadence or throughput against latency | Bidirectional mode adds about 20 to 45 ms more than the default modes (more at lower frame rates); batch repaint raised real frames from 34 to 44 FPS but the source-to-flip p95 from about 11 to 25 ms |

The one change that improved both sides was evidence-based rescue: the learned
net restores a rejected vector only where the refined vector predicts the pair
better than holding (zero motion), which helped smoothness and image quality
together.

Three content classes span the space. A change that wins one of them is not a
win, and a default has to stay inside every limit on all three:

| Class | Tested with | Looks like |
|---|---|---|
| Simple, sharp-edged | Xonotic | An older shooter with sharp edges |
| Adversarial swarm on high contrast | GravityMark, 1M asteroids | A dense particle swarm and large round bodies on black; the hardest case |
| Realistic | Far Cry Primal, Far Cry New Dawn, Far Cry 3 Blood Dragon | Modern game content |

### Blind A/B protocol

The blind A/Bs in this guide ([results](#blind-ab-results-laptop-panel-2026-10-1617))
follow one protocol:

1. Both clips show the same content.
2. A and B are presented in random hidden order, with the HUD off.
3. The viewer says which one they would rather play with.
4. A second round repeats the comparison with the order re-randomized.
5. The assignment is revealed only afterwards.

When an arm uses in-game learning, start every such arm, blind or real-game,
from its own copy of the same `GAMESCOPE_FRAMEGEN_NET_PROFILE` file, for example
`cp fg-game.bin fg-game-armA.bin`. The profile keeps training across runs, so a
shared file gives later arms a better starting point. One file holds the sets
for every input encoding, so one copy per arm is enough.

### The blind verdict outranks the metrics

The showcase columns and the offline gates narrow the candidates; the viewer's
blind verdict decides, and when the two disagree the verdict stands. Example: the
temporal gate ([Offline gates on captured content](#offline-gates-on-captured-content))
scored the /4 motion field better than /8 on flicker on every content class, yet
the viewer found /4 blocky on GravityMark in both rounds. The flicker metric
over-weights judder relative to this viewer, so /4 stays opt-in
(`GAMESCOPE_FRAMEGEN_FIELD_FINE=1`).

The verdicts in this guide are one viewer's, on one laptop. The README's
[Choosing settings](../README.md#choosing-settings) turns them into starting
points for players.

## Real-game A/B: Far Cry Primal

A hand-run A/B on a real game (2026-10-17), complementing the scripted showcase.
It is indicative, not a gate: the scene differed between runs and the judgement
is partly subjective.

Setup:

- Far Cry Primal (DX11 via DXVK, GE-Proton11-9, umu), started through Ubisoft
  Connect from a native Lutris 0.5.22 install.
- Hybrid laptop: the game renders on the NVIDIA RTX PRO 500 Blackwell laptop GPU
  (`10de:2db9`); gamescope composites, generates and presents on the AMD Radeon
  890M (`1002:150e`) through `scripts/gamescope-hybrid.sh --steer-child`.
- Nested in GNOME (Wayland backend), 2560x1600 at 120 Hz, fullscreen.
  `GAMESCOPE_FRAMEGEN_METRICS=1`, `GAMESCOPE_FRAMEGEN_HUD=2`, `--framegen-debug`
  and a per-game `GAMESCOPE_FRAMEGEN_NET_ONLINE` profile.
- Per configuration: fresh game start, about 45 s of loading, then 60 s of free
  play. Values are medians of the 5 s framegen-metrics windows (rates are
  totals divided by seconds).

**Cap the game below the output rate.** Uncapped, the game already delivered
118-120 real frames/s to gamescope at 120 Hz, so there was nothing to fill
(`gen` 0-2/s, fill about 1.0). A real-game A/B must cap the game below the
output rate; this run used `DXVK_FRAME_RATE=60` (40 for x3).

### Mode comparison

Learned pipeline unless noted; game capped as shown.

| Configuration | Game fps | Gen fps | Output fps | `flip_ms_sd` | Latency p50 | `q_resid` | Slow drops |
|---------------|---------:|--------:|-----------:|-------------:|-------------|----------:|-----------:|
| forward x2, cap 60 | 55 | 57 | 112 | 1.98 ms | 16 ms (`src_flip`) | 0.027 | 0 |
| bidir x2, cap 60 | 58 | 58 | 115 | 1.50 ms | 46 ms (`dreal_flip`) | 0.028 | 0 |
| bidir x3, cap 40 | 39 | 72 | 111 | 2.29 ms | 63 ms (`dreal_flip`) | 0.032 | 0 |
| checked forward x2, cap 60 | 59 | 56 | 115 | 1.90 ms | 15 ms (`src_flip`) | 0.027 | 14 |

`q_killed_fb` was 0.018 for forward learned and 0.18 for bidir. Generation GPU
time on the 890M was about 3-4.6 ms per batch.

The player ranked forward x2 best and bidir x2 worst, although bidir x2 had the
lowest jitter. Bidir shows each real frame about 30 ms later (p50 46 ms against
16 ms), and that latency cost outweighed the smoother cadence. `flip_ms_sd` and
fill alone therefore mis-rank configurations; weigh latency with them. This
agrees with the earlier blind A/B in which quick and sharp beat smooth.

### Pipeline comparison

Forward x2, game capped at 60.

| Pipeline | `flip_ms_sd` | Latency p50 (`src_flip`) | Slow / disc | `q_bad_mv` | `q_resid` | Gen GPU per batch |
|----------|-------------:|-------------------------:|------------:|-----------:|----------:|------------------:|
| warp | 1.89 ms | 15 ms | 14 / 44 | 0.030 | 0.028 | 1.3 ms |
| checked | 1.36 ms | 19 ms | 13 / 13 | 0.024 | 0.028 | 1.6 ms |
| learned | 1.18 ms | 16 ms | 0 / 0 | 0.014 | 0.027 | 3.0 ms |
| predict | 1.81 ms | 15 ms | 0 / 2 | 0.013 | 0.027 | 3.1 ms |
| guided | 1.65 ms | 16 ms | 0 / 0 | 0.011 | 0.026 | 3.8 ms |

Player ranking: learned > guided > warp = predict > checked. Learned was the
best overall; warp felt fast and snappy but looked worse; checked and predict
felt choppy; predict tied with warp because it looked better but felt less
snappy. Result for this hybrid laptop:
forward (no bidir) x2 with the learned pipeline. Because the scene differed
between runs, the `q_*` values are indicative only.

### Far Cry New Dawn, uncapped

Same setup, no frame cap. New Dawn renders about 68 fps natively on the
NVIDIA card, an uneven ratio to 120 Hz, so generation fills irregular gaps.
The first ~20 s of each measured minute were still loading screens, so later
sequences wait 65 s before measuring.

| config | game fps | gen fps | output fps | `flip_ms_sd` | latency p50 | slow / disc | `q_bad_mv` | `q_resid` | gen GPU per batch |
|---|---|---|---|---|---|---|---|---|---|
| off | ~68 | 0 | ~68 | - | - | - | - | - | - |
| forward x2, learned | 67 | 48 | 115 | 1.56 ms | 17 ms | 1 / 22 | 0.013 | 0.025 | 3.1 ms |
| forward x2, guided | 69 | 48 | 117 | 1.28 ms | 17 ms | 0 / 0 | 0.013 | 0.027 | 3.6 ms |
| forward x2, warp | 68 | 49 | 117 | 0.99 ms | 17 ms | 0 / 10 | 0.028 | 0.029 | 1.3 ms |
| bidir x2, learned | 70 | 43 | 113 | 1.34 ms | 38 ms (`dreal_flip`) | 0 / 45 | 0.019 | 0.028 | 4.2 ms |

Player ranking: learned > guided = bidir > warp, with off clearly worst;
guided was close to learned. The player noted the choice depends on the use
case. The learned profile for this game started from a neutral prior in this
session. Across both games, forward x2 with the learned pipeline is the default
and warp keeps the lowest jitter and GPU cost but the worst motion vectors.

Launcher note: Ubisoft Connect's own login fails with a generic "An error
occurred while trying to send your request" when a DNS ad-blocker blocks
`dd.ubisoft.com`, `js.datadome.co` or `api-js.datadome.co`. The login is
protected by DataDome and the blocked hosts return HTTP 403 (`x-datadome:
protected`).

### Far Cry 3 Blood Dragon, uncapped

Same setup, no frame cap (Ubisoft 205, `fc3_blooddragon_d3d11.exe`, a 32-bit
DirectX 11 game). It renders about 55 fps natively; each run waited 65 s for
loading, then measured 60 s.

| config | game fps | gen fps | output fps | `flip_ms_sd` | latency p50 | slow / disc | `q_bad_mv` | `q_resid` | gen GPU per batch |
|---|---|---|---|---|---|---|---|---|---|
| off | 55 | 0 | 55 | 3.87 ms | 15 ms | - | - | - | - |
| forward x2, learned | 57 | 60 | 117 | 1.50 ms | 17 ms | 9 / 2 | 0.018 | 0.032 | 1.2 ms |
| forward x2, guided | 54 | 62 | 116 | 1.43 ms | 17 ms | 14 / 4 | 0.014 | 0.030 | 1.2 ms |
| forward x2, warp | 54 | 64 | 118 | 1.15 ms | 17 ms | 0 / 11 | 0.026 | 0.031 | 1.1 ms |
| bidir x2, learned | 52 | 50 | 102 | 2.98 ms | 54 ms (`dreal_flip`) | 0 / 16 | 0.021 | 0.032 | 4.0 ms |

Player ranking: learned > guided > bidir; warp was not ranked. Forward x2 more
than doubles the output rate (55 to about 117 fps) and cuts the frame-time
jitter from 3.87 ms to about 1.5 ms, for about 2 ms more latency, the cost the
[latency gate](#latency-gate-framegen-on-vs-off) measures. Bidirectional was the
weakest here: the lowest output rate (fill about 0.87), the most jitter of the
framegen runs and 54 ms latency. Learned has now ranked first in all three Far
Cry games.

### Far Cry Primal, HD textures, cap 60

Far Cry Primal again, with the HD texture pack installed and higher in-game
graphics settings, so these numbers do not compare one-to-one with the first
Primal table. Game capped at 60 with `DXVK_FRAME_RATE=60`, 65 s load, 60 s
measured. The game still held its 60 fps cap.

| config | game fps | gen fps | output fps | `flip_ms_sd` | latency p50 | slow / disc | `q_bad_mv` | `q_resid` | gen GPU per batch |
|---|---|---|---|---|---|---|---|---|---|
| off | 60 (cap) | 0 | 60 | - | - | - | - | - | - |
| forward x2, learned | 60 | 59 | 119 | 1.08 ms | 18 ms | 1 / 0 | 0.013 | 0.027 | 3.1 ms |
| forward x2, guided | 56 | 58 | 114 | 1.15 ms | 18 ms | 1 / 0 | 0.010 | 0.025 | 3.9 ms |
| forward x2, warp | 59 | 57 | 116 | 1.60 ms | 16 ms | 0 / 10 | 0.033 | 0.029 | 1.4 ms |
| bidir x2, learned | 59 | 53 | 113 | 1.60 ms | 54 ms (`dreal_flip`) | 0 / 15 | 0.029 | 0.030 | 4.2 ms |

Player ranking: learned >= guided > bidir > warp ("difficult" to separate
learned and guided). Bidirectional felt poor at the start of the run; warp
showed too many artifacts. Learned gave the lowest jitter of any Far Cry run
(1.08 ms) with essentially no drops; warp had the highest `q_bad_mv` measured
(0.033).

### Far Cry New Dawn, HD textures, low frame rate

Far Cry New Dawn with the HD texture pack and higher settings renders about
41 fps here, dipping to 30-40 (off: 42 fps, `flip_ms_sd` 4.63 ms). Four rounds
in the same scene, 65 s load and 60 s measured per run, fixed run order, not
blind. Columns: game fps / output fps / `fill` / `flip_ms_sd` / latency p50
(`src_flip`, or `dreal_flip` for bidir) / slow / disc / highest degrade level /
`q_bad_mv`.

| config | x2 | x3 | x4 |
|---|---|---|---|
| learned | 37 / 113 / 0.95 / 2.15 / 17 / 30 / 96 / 3 / 0.018 | 42 / 117 / 0.99 / 1.87 / 17 / 0 / 10 / 0 / 0.015 | 43 / 117 / 0.99 / 1.60 / 17 / 0 / 7 / 0 / 0.012 |
| guided | 42 / 116 / 0.98 / 1.72 / 17 / 1 / 6 / 1 / 0.015 | 41 / 116 / 0.98 / 1.65 / 17 / 0 / 9 / 0 / 0.013 | 41 / 113 / 0.97 / 1.84 / 17 / 4 / 9 / 3 / 0.013 |
| warp | 41 / 116 / 0.99 / 1.60 / 17 / 2 / 29 / 1 / 0.033 | 42 / 117 / 0.99 / 1.38 / 17 / 0 / 4 / 0 / 0.028 | 43 / 117 / 0.99 / 1.43 / 17 / 0 / 19 / 0 / 0.033 |
| bidir, learned | 42 / 84 / 0.71 / 2.74 / 54 / 0 / 9 / 3 / - | 40 / 110 / 0.98 / 2.05 / 54 / 0 / 36 / 1 / 0.021 | 42 / 118 / 1.00 / 1.23 / 54 / 0 / 28 / 1 / 0.018 |

On the default forward path the multiplier does not set how many frames are
generated: causal deadline scheduling (`plan_next_causal_slot`) fills every
free vblank until the next real frame and never reads it; the multiplier only
sizes the generated pool and output ring. Forward `gen` and `fill` are indeed
flat across x2/x3/x4. The weak x2 learned run is a game-rate confound: the game
itself ran at 37 fps in that window against 41-42 in every other forward run.
The multiplier does bound bidirectional mode (`plan_bidir_slots` generates at
most multiplier - 1 frames per real frame): at about 41 fps bidir x2 filled
only 71% of the slots, x3 98%, x4 100%.

Player verdict on the x2/x3 rounds: at 30-40 fps "x2 is too low, all are bad";
once the multiplier safely reaches the refresh rate the known ranking returns.
Warp is quick and smooth but shows many artifacts at this low input rate;
bidirectional "surprisingly shines", almost artifact-free at x4, but has lag
and a specific stutter. This verdict stands as given, with the confounds above.

**Anti-lag round (x4).** `DXVK_CONFIG="dxgi.maxFrameLatency = 1;
dxvk.latencySleep = True"` caps the game's render queue at one frame and lets
DXVK pace frame starts; on this driver DXVK reports `Latency control enabled,
using VK_NV_low_latency2` even though the game has no Reflex support. It cost no
frame rate (game 40-42 fps). Its gain happens before gamescope, so `src_flip`
cannot show it; DXVK's `DXVK_HUD=latency` readout and feel are the only measure
here.

| config (x4) | without anti-lag | with anti-lag | with anti-lag, `dxgi.maxFrameRate = 40` |
|---|---|---|---|
| learned | 43 / 117 / 0.99 / 1.60 / 17 / 0 / 7 / 0 / 0.012 | 41 / 116 / 0.99 / 1.67 / 17 / 0 / 6 / 0 / 0.010 | 39 / 117 / 0.99 / 1.39 / 18 / 2 / 4 / 1 / 0.010 |
| guided | 41 / 113 / 0.97 / 1.84 / 17 / 4 / 9 / 3 / 0.013 | 41 / 116 / 0.98 / 1.50 / 17 / 3 / 8 / 1 / 0.011 | 38 / 115 / 0.98 / 1.62 / 17 / 8 / 55 / 4 / 0.009 |
| warp | 43 / 117 / 0.99 / 1.43 / 17 / 0 / 19 / 0 / 0.033 | 42 / 118 / 0.99 / 1.42 / 17 / 1 / 10 / 1 / 0.025 | 39 / 117 / 0.99 / 1.59 / 17 / 8 / 72 / 1 / 0.032 |
| bidir, learned | 42 / 118 / 1.00 / 1.23 / 54 / 0 / 28 / 1 / 0.018 | 41 / 118 / 1.00 / 1.23 / 63 / 0 / 28 / 1 / 0.018 | 35 / 115 / 0.97 / 1.67 / 63 / 1 / 30 / 1 / 0.022 |

Player ranking for the anti-lag round, on two axes: responsiveness guided >=
learned > warp > bidir; visual quality bidir > guided > learned >= warp. Guided
placing level with learned is one confounded ranking (its degrade ladder
settled at 1 here instead of 3), not a rule for low frame rates. Warp measured
the same 17 ms as learned and guided yet felt less responsive, which points at
motion-vector error (its `q_bad_mv` is two to three times the others) being
perceived as lag; `src_flip` alone does not capture responsiveness.

The 40 fps cap did not help: the game could not hold 40 in this scene (35-39
fps), so the cadence became less even, and guided and warp discarded far more
frames. A cap only helps when the game holds it with margin; here that would be
30 fps (exactly four vblanks at 120 Hz). The player confirmed the measurement
and described bidirectional as having "a strange sluggish stutter inside the
pictures". Bidirectional added about 37 ms at this frame rate and about 46 ms
with anti-lag (`dreal_flip` 54 and 63 ms against 17 ms forward). Its delay
controller held `delay_ms` at 41.7 ms without anti-lag, moved between 41.7 and
66.7 ms with anti-lag (7 reprimes), and sat at 50 ms with the cap, while real
frames arrived 3, 4 or 5 vblanks apart. Uneven arrival under a fixed or
re-planned delay is the leading candidate for the in-picture stutter; a
phase/timestamp mismatch is not excluded. Bidirectional interpolation phases
follow source-ready timestamps, so render-time jitter that shifts source-ready
without moving the game's simulation step would vary the on-screen velocity per
refresh even at a constant delay. The x4 run without anti-lag already showed the
stutter with a constant 41.7 ms delay and 39 of 43 sampled gaps at 3 vblanks,
which fits this second explanation. A 30 fps cap evens out both and cannot tell
them apart; an A/B of frame-index phase against source-time phase at uneven
cadence can.

## Blind A/B results (laptop panel, 2026-10-16/17)

Blind A/Bs by one viewer on the laptop's 120 Hz panel, in random hidden order
([protocol](#blind-ab-protocol)). Like the Far Cry runs they are indicative, not
a gate. Xonotic is an older, sharp-edged shooter. GravityMark with 1M asteroids
is a dense particle swarm plus large round bodies on black, the hardest case.

| Content | Compared | Verdict |
|---|---|---|
| Xonotic | Sharp fallback (the default) against framegen off | Sharp fallback preferred |
| Xonotic | Crossfade (`GAMESCOPE_FRAMEGEN_SHARP_FALLBACK=0`) against framegen off | Off preferred: "off faster, sharper" |
| Xonotic | `GAMESCOPE_FRAMEGEN_FIELD_FINE=1` against the default | `FIELD_FINE` looked crisper (2 of 2 rounds) |
| GravityMark, 1M asteroids | `GAMESCOPE_FRAMEGEN_FIELD_FINE=1` against the default | `FIELD_FINE` looked blocky (2 of 2 rounds) |
| GravityMark, 1M asteroids | Default (/8 field, sharp fallback) against framegen off | Default preferred: "clearly smoother, frayed edges" |
| vkmark desktop | `GAMESCOPE_FRAMEGEN_FIELD_FINE=1` against the default | Tie |

**Not yet tested.** All of these ran on the `checked` pipeline. `learned` has not
been blind-tested on Xonotic or GravityMark, and the sharp fallback has not yet
been verified on `learned`.

### Sharp fallback

Where the motion evidence is weak, the predictive warp moves the real frame only
as far as the confidence allows (the displacement is scaled by the final
confidence, with one final colour sample) instead of crossfading the motion warp
with the colour extrapolation. It is on by default;
`GAMESCOPE_FRAMEGEN_SHARP_FALLBACK=0` restores the crossfade, which leaves double
edges and blur wherever the confidence changes. It shows only real pixels, so
HDR is unaffected.

It applies to the predictive warp shader, which the `warp`, `checked` and
`learned` pipelines use, and `predict` while it has no history; `predict` and
`guided` otherwise use the acceleration shader, and bidirectional mode its own.
Compared with the crossfade, the cost per generated frame on the Radeon 890M is
equal at 1080p and about 10% lower at 1440p and 2160p.

Known limit: it frays edges on dense particle swarms. Dropping the per-pixel
agreement removed the fraying but made Xonotic choppier in a blind test, so the
agreement stays.

### Finer field

`GAMESCOPE_FRAMEGEN_FIELD_FINE=1` (off by default) uses a finer /4 motion field:
one more pyramid level, four times the field texels, and one more pyramid
dispatch and refinement per direction. It applies to the `warp` and `checked`
pipelines only and is ignored for `learned`, `predict` and `guided`.
`GAMESCOPE_FRAMEGEN_RECORD` is ignored while it is on, because the recorded
samples are /8. It looked crisper on Xonotic and tied on vkmark but blocky on
GravityMark, so it stays opt-in.

### Batch repaint and the August build

`GAMESCOPE_FRAMEGEN_BATCH_REPAINT=1` (off by default; causal mode only, not
bidirectional, not VRR hybrid) requests a composite after each generated batch
instead of only waking the compositor, which is how builds behaved before commit
518498f. On a GPU-bound client (GravityMark 1M, 2048x1280@120, headless) real
frames rose from 34 to 44 FPS with steadier flips, but the source-to-flip p95
grew from about 11 to about 25 ms, and the viewer rated it laggier in a blind
A/B. That is why it is off.

The same trade explains an earlier comparison in which an August build "felt
smoother but laggier": that build still had the behaviour this option restores.

## Latency gate: framegen on vs off

The p95 tail of real-frame latency is what reads as lag (a longer tail was rated
laggier in the batch-repaint A/B above), so the cost of turning framegen on is
measured and gated. Measured headless on two GPUs, with GravityMark at 1M
asteroids and with Xonotic, on debug and optimized builds alike: framegen on adds
about 2 ms from source to flip over framegen off, at p50 and at p95.

All of it sits after the latch. Framegen off with composition forced
(`--force-composition`, the `composite_force` convar) matches framegen on, so the
cost is the full composite that both backends force for every real frame while
framegen is on (`src/Backends/WaylandBackend.cpp` and
`src/Backends/DRMBackend.cpp`). Framegen's own work adds 0 to 0.5 ms.

### Measuring it

With `GAMESCOPE_FRAMEGEN_METRICS=1`, four fields appended at the end of each
`framegen-metrics:` line split `src_flip_ms_*` at the latch:

| Fields | Interval |
|---|---|
| `src_rec_ms_p50`, `src_rec_ms_p95` | Source ready to the first paint or latch of that real frame |
| `rec_flip_ms_p50`, `rec_flip_ms_p95` | Latch to the actual flip |

They are recorded with framegen off too, so the same fields compare off with on.
With framegen on, the difference shows in `rec_flip_ms_*`; `src_rec_ms_*` does not
move.

The showcase reports the end-to-end percentiles as `lat_p50` and `lat_p95` for
every run, `off` included, so one session shows the cost as the difference
between the `off` row and a forward framegen row. Compare like with like: a
bidirectional run is measured from the record, not from source-ready, and is
marked `d`. `--compare` gates a candidate against a saved baseline: `lat_p95` may
rise by at most 2 ms and `lat_p50` by at most 1 ms (see the
[compare gates](#end-to-end-showcase-and-baseline-compare)).

```bash
# Framegen off (run 1) against forward framegen (run 6: motion x2, checked),
# rendering on the dGPU and presenting on the iGPU, in the background.
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
  scripts/framegen-headless-host.py --mode 2560x1440@120 -- \
  env HZ=120 RENDER_DEV='<dgpu vid:did>!' PRESENT_DEV='<igpu vid:did>' \
  scripts/framegen-showcase.sh 1 6
```

### Why the composite stays

A design that presented each real frame directly, as framegen off does, and
composited the history off the critical path would have removed the 2 ms. It was
rejected after review, for three reasons:

- The client image has exclusive ownership and uses FOREIGN queue-ownership
  transfers, so a host or KMS read that runs concurrently with the history
  composite is not portable.
- The dual-GPU staging path replaces the commit's texture with a non-flippable
  image.
- Recording the history after the commit would break the present tags and the
  timing anchors.

The composite stays.

## Offline prediction-error split (S1/S2)

`scripts/framegen-hindsight-eval.py` splits the predictive path's error on a
recorded capture into the residual left by the next pair's own estimated field
and the continuation gap on top of it. Capture in motion mode with
`GAMESCOPE_FRAMEGEN_RECORD=<dir>`, then run (numpy required):

```bash
python3 scripts/framegen-hindsight-eval.py --data <dir>
```

It reuses the trainer's GSFD parser and bilinear sampler
(`scripts/framegen-net-train.py`) and streams the capture, holding two samples
at a time. Unreadable samples are skipped with a message and break the chain of
consecutive frames. For each pair of consecutive real frames N and N+1, over the
texels moving in either field (at least 0.25 field texels per frame):

- `hold`: the error of showing frame N again.
- `cont`: constant-motion continuation, frame N warped along its own field (what
  the predictive path does).
- `cont*`: the same with the best scale per pair; `cadence-aware gain` is what
  scaling the motion by frame timing could win.
- `oracle`: frame N warped along the next pair's estimated field. This is S1,
  the retrospective-field residual: an upper bound on what better matching
  could remove, since it also holds disocclusion, shading change and
  resampling error that no motion vector recovers.
- S2 = `cont` − `oracle`, the continuation gap, and its share of `cont`.

The rejection block looks at the texels moving in the judged field only (the
field generation uses, so its moving set is narrower than the split's): of the
moving texels the field rejects (confidence below 0.5), how often warping would
have beaten holding, the mean gain, the p95 loss, how often warping wins on the
kept texels, and the shares of confidence below 0.1 and between 0.1 and 0.5.

Scope: field-resolution luma and absolute error, not a colour-domain perceptual
score, the same limit as `framegen-net-eval.py`. A continuation teacher is
worth building only when S2 is at least 30% of the predictive error on two
contents.

## Offline gates on captured content

Two numpy tools judge a motion-field or output change on recorded frames, so that
candidates are narrowed offline before a blind A/B decides.
`scripts/framegen-capture.sh` records the frames, `scripts/framegen-field-gate.py`
scores one generated frame at a time, and `scripts/framegen-temporal-gate.py`
replays the displayed sequence. Capture needs a gamescope built with PipeWire and
`gst-launch-1.0` with `pipewiresrc`; the gates need `numpy` and read
`NAME-WxH.gray` from the capture directory. The capture script writes to
`${XDG_CACHE_HOME:-$HOME/.cache}/gamescope-framegen-captures` and the gates read
`~/.cache/gamescope-framegen-captures`; pass `--captures` when `OUT` or
`XDG_CACHE_HOME` moves it.

### Capture

```text
framegen-capture.sh NAME W H WARMUP_S SECONDS -- client command...
```

The client runs in a headless gamescope at exactly W x H with framegen off. After
WARMUP_S seconds the script records SECONDS seconds of its PipeWire stream as raw
8-bit luma into `$OUT/NAME-WxH.gray` (frames of W x H bytes) and writes the
gamescope log to `NAME-WxH.log`. It fails on a failed or partial capture.

| Variable | Default | Meaning |
|---|---|---|
| `OUT` | `${XDG_CACHE_HOME:-$HOME/.cache}/gamescope-framegen-captures` | Capture directory |
| `RATE` | 60 | Output refresh rate |
| `PRESENT_DEV` | empty | gamescope's `--prefer-vk-device` |
| `GAMESCOPE_BIN` | `build/src/gamescope` | Binary to run |

The field gate drops repeated frames, so the client's frame rate does not matter
to it. For ground truth at in-between phases (the temporal gate), record with
`RATE=120` and a vsynced client, so every captured frame is a distinct, evenly
spaced moment:

```bash
# 20 s warm-up, 30 s recorded.
scripts/framegen-capture.sh xonotic 1280 720 20 30 -- /path/to/client
RATE=120 scripts/framegen-capture.sh xonotic120 1280 720 20 30 -- /path/to/vsynced-client
```

### Field gate

```text
framegen-field-gate.py --size WxH [--stride N] [--arms a,b,...] [--domain coded|linear] [--net weights.bin] [--captures DIR] NAME...
framegen-field-gate.py --parity DIR
```

A numpy port of the production field chain (luma pair, pyramid, matcher,
confidence, forward-backward check) and of the warps. It scores triplets A, B, C
of distinct consecutive frames, every `--stride`-th (default 6): the predictive
output predicts C from B along A to B, the bidirectional output rebuilds B from A
and C. Per capture and method it prints, for each arm, the paired change in error
against the first arm with a 95% bootstrap interval, and three more columns:

| Column | Meaning |
|---|---|
| paired change `[lo,hi]` | Change in the error where the frame changed, against the first arm |
| `sta` | Error on static pixels: drag |
| `lost` | Edges in the truth that the output lacks: smear |
| `false` | Edges the output invents: tearing, block shear |

Arms (`--arms`, default `checked/8,checked/4`; the first one is the reference):

| Arm | Meaning |
|---|---|
| `checked/8`, `checked/4` | The production field at /8 or /4 |
| `...@xf` | The predictive output with the crossfade fallback (`GAMESCOPE_FRAMEGEN_SHARP_FALLBACK=0`). Without the suffix the port models the sharp fallback, as production does by default |
| `...+vm`, `...+nc` | Variants of the field: `+vm` takes the 3x3 vector median of the final field; `+nc` keeps confidence only where a vector agrees with its median |
| `learned/8` | `checked/8` refined by a GSFR net (`--net`) |

`--domain coded` (the default) matches on gamma-coded code values, like
production for 8- and 10-bit outputs; production matches float outputs on linear
values. `--domain linear` decodes sRGB first and is a what-if.

The port's numbers count only while it reproduces the GPU. `--parity DIR` checks
that: it runs the `checked/8` chain on the lumas of `GAMESCOPE_FRAMEGEN_RECORD`
samples (captured as in [Offline prediction-error split](#offline-prediction-error-split-s1s2))
and compares both fields, forward and reverse, with the ones the GPU recorded. It
fails below 99.9% identical vectors, above a mean confidence error of 0.01 in any
field, or on non-finite values. The last result was 99.997% identical vectors and
a worst-field mean confidence error of 0.00019.

Parity reaches only the `checked/8` field chain. The samples are /8 and hold the
checked fields before refinement (`GAMESCOPE_FRAMEGEN_RECORD` is ignored while
`GAMESCOPE_FRAMEGEN_FIELD_FINE=1`), so it does not cover the /4 arm, the warps or
the refiner.

### Temporal gate

```text
framegen-temporal-gate.py --size WxH [--k 2,3] [--arms hold,checked/8,checked/4] [--stride N] [--domain coded|linear] [--methods predictive,bidir] [--captures DIR] NAME...
```

Tearing and flicker are temporal: a generated frame can be close to the truth on
its own and still jump against its neighbours. The temporal gate therefore
replays the displayed sequence on a capture made faster than the simulated source
(`RATE=120` and a vsynced client). It treats every k-th captured frame as a real
frame (k = 2 simulates a 60 FPS source, k = 3 a 40 FPS source), and the frames
between two reals are the ground truth for the generated frames at phases j/k:

- `predictive`: from the reals f[i-k] and f[i], it shows f[i], the predicted
  frames, then f[i+k].
- `bidir`: from the reals f[i] and f[i+k], it shows f[i], the interpolated
  frames, then f[i+k].
- `hold` (an arm, framegen off): it shows f[i] until f[i+k].

The metrics are taken on the pixels the truth changes within the interval:

| Metric | Meaning |
|---|---|
| `E` | Mean error of the generated frames |
| `flick` | Per-step difference between the shown change and the true change, including the error that appears after a real frame and snaps away at the next one; `hold` scores its judder here. Printed with the paired change against the first arm and a 95% bootstrap interval |
| `eflick` | The same on gradient magnitude, positive part only: edges that move or pop when the truth's do not (tearing, shear) |
| `lost`, `false` | Edges of the truth missing from the generated frames (smear, blur); edges in the generated frames that the truth lacks (ghosts, double edges) |
| `flip` | Predictive only: share of confident vectors that change by more than 2 px between consecutive real intervals after motion compensation |

Arms are as in the field gate, plus `hold`, and `@xf` selects the crossfade
fallback as there. It takes no `--net`, so there is no `learned` arm. It warns
when more than 2% of the captured frames repeat: the client was then slower than
the capture and the truth is not evenly spaced.

### What the gates predicted and what they missed

- The port reproduces the GPU: the last `--parity` run gave 99.997% identical
  vectors and a worst-field mean confidence error of 0.00019, well inside the
  limits.
- The temporal gate did not predict the eye on swarms. It scored /4 better than
  /8 on flicker on every content class, yet in the blind A/Bs /4 looked crisper on
  Xonotic, tied on vkmark and looked blocky on GravityMark. The metric
  over-weights judder relative to this viewer, so the blind A/B outranks it
  ([How to judge a setting](#how-to-judge-a-setting)) and /4 stays opt-in.

Both gates score luma from 8-bit captures, so they say nothing about colour or
HDR. Offline numbers guide; the blind A/B decides.

## Offline refiner and colour evaluators

Both need `numpy` and read a capture directory. Neither runs gamescope.

### `framegen-net-eval.py`

Capture field-resolution samples with `GAMESCOPE_FRAMEGEN_RECORD=<dir>` (the
same capture the hindsight evaluator reads), then:

```bash
python3 scripts/framegen-net-eval.py --data <dir>
python3 scripts/framegen-net-eval.py --data <dir> --net weights.bin
```

It grades the field's photometric prediction (the source luma warped along the
field, compared with the destination luma) with structural and temporal
metrics instead of a single residual. With `--net` it also runs a trained
`GSFR` weights blob and reports the neutral (no refiner) against refined
deltas, so a blob can be judged before it is shipped. Weights come from
`scripts/framegen-net-train.py --data <dir> --out weights.bin`; see
[the AI refiner section of the user guide](framegen-howto.md#f-ai-refiner-in-game-learning).
It shares the trainer's parser and net, which `framegen-net-train.py` must stay
next to. Scope: field-resolution luma, not a colour-domain perceptual score.

### `framegen-color-eval.py`

Capture full-colour held-out triplets with
`GAMESCOPE_FRAMEGEN_RECORD_COLOR=<dir>` (optional `_MAX` and `_SKIP`, see the
[Configuration reference](framegen-reference.md)). The capture predicts frame B
from A and C without presenting the prediction, and stores it beside the real
B in a `color_*.gscf` file. Capture writes perturb pacing, so this measures
image quality, not pacing.

```bash
python3 scripts/framegen-color-eval.py --data <dir>
python3 scripts/framegen-color-eval.py --data <dir> --per-frame   # every sample
python3 scripts/framegen-color-eval.py --data <dir> --lpips       # optional, needs torch and lpips
```

Higher PSNR and SSIM, and lower MAE, bad, edge and residual-change values, are
better. Metrics grade the captured output code values; EOTF metadata is
reported and no HDR display transform is applied.

## Reference presets

Ready-made, benchmark-oriented configurations. They are not required for normal
play. The default causal fixed-cadence behavior is described in the architecture
document ([framegen-architecture.md](framegen-architecture.md)).

### Preferred visual baseline: GravityMark x4

The accepted, most visually appealing GravityMark configuration from live
testing: Motion Guided, three generated frames per real frame, bidirectional
presentation, strength `0.5`, and the frozen learned profile. It prioritizes
visual smoothness and quality. Bidirectional mode shows real frames later than
live, so this is not the competitive-latency preset.

The frozen learned profile is not shipped with the repository. It is a
`.bin` file produced by in-situ learning on the author's machine. To make your
own, run GravityMark with `GAMESCOPE_FRAMEGEN_NET_ONLINE=1` and
`GAMESCOPE_FRAMEGEN_NET_PROFILE=<path>` for several minutes (see
[the AI refiner section of the user guide](framegen-howto.md#f-ai-refiner-in-game-learning)), then use that file as
`FRAMEGEN_BEST_PROFILE`. The command serves it read-only through
`GAMESCOPE_FRAMEGEN_NET_PROFILE` (without `NET_ONLINE` nothing is trained or
saved). To run the preset without a net instead, delete the
`GAMESCOPE_FRAMEGEN_NET_PROFILE=...` line from the command. The preset is a
benchmark-only configuration and is not validated on every topology.

`build-perf` below is just a second meson build directory used for
benchmarks; any name works if you use the same one everywhere (see
[Validation by change class](framegen-maintenance.md#validation-by-change-class)).

Set the two PCI IDs and paths for the machine, then run:

```bash
export PRESENT_DEV=1002:5678          # GPU physically connected to the display
export RENDER_DEV=10de:1234!          # GPU rendering GravityMark
export GRAVITYMARK_DIR=/path/to/GravityMark/bin
export FRAMEGEN_BEST_PROFILE="$HOME/.cache/gamescope-fg-gravitymark-best-<id>.bin"

test -r "$FRAMEGEN_BEST_PROFILE" # must succeed; see the note above if the file does not exist
unset GAMESCOPE_FRAMEGEN_VRR_HYBRID GAMESCOPE_FRAMEGEN_BASE \
  GAMESCOPE_FRAMEGEN_NET_ONLINE \
  GAMESCOPE_FRAMEGEN_NET GAMESCOPE_FRAMEGEN_RECORD_COLOR \
  GAMESCOPE_FRAMEGEN_BIDIR_TRACE

GAMESCOPE_BUILD_DIR=build-perf \
GAMESCOPE_FRAMEGEN_BIDIR=1 \
GAMESCOPE_FRAMEGEN_NET_PROFILE="$FRAMEGEN_BEST_PROFILE" \
GAMESCOPE_FRAMEGEN_BIDIR_OCCLUSION=0 \
GAMESCOPE_FRAMEGEN_DEBUG_EVERY=60 \
./scripts/env-gamescope-local.sh \
gamescope --expose-wayland --backend wayland \
  --prefer-vk-device "$PRESENT_DEV" \
  -W 2560 -H 1440 -r 120 \
  --experimental-framegen \
  --framegen-mode motion \
  --framegen-multiplier 4 \
  --framegen-pipeline guided \
  --framegen-strength 0.5 \
  --framegen-debug \
  -- env -C "$GRAVITYMARK_DIR" MESA_VK_DEVICE_SELECT="$RENDER_DEV" \
    ./GravityMark.x64 -vk -temporal 0 -benchmark 0 -fps 1 \
    -asteroids 1000000 -width 2560 -height 1440 -vsync 0
```

The frozen command is repeatable and cannot modify the accepted profile.
Reservoir and shading are omitted on purpose: bidirectional mode forces both
off, so setting them here would be inert.

To continue in-situ learning without touching the frozen profile, make a
writable copy:

```bash
export FRAMEGEN_WORKING_PROFILE="$HOME/.cache/gamescope-fg-gravitymark-working.bin"
cp -a "$FRAMEGEN_BEST_PROFILE" "$FRAMEGEN_WORKING_PROFILE"
chmod u+w "$FRAMEGEN_WORKING_PROFILE"
```

Then remove the frozen command's `GAMESCOPE_FRAMEGEN_NET_PROFILE=...` line and export:

```bash
export GAMESCOPE_FRAMEGEN_NET_ONLINE=1
export GAMESCOPE_FRAMEGEN_NET_PROFILE="$FRAMEGEN_WORKING_PROFILE"
```

### Optional cadence-locked variant

At 120 Hz, x4 can fully populate the display only while the real-frame cadence
stays at or above 30 fps. A workload that changes from a short interval to five
or more vblanks when camera motion begins cannot be covered retroactively by
bidirectional generation: it has already drained the slots planned from the
previous interval, and x4 can insert at most three new frames. The result is an
honest repeat of the newest real frame, often perceived as a brief onset
stutter.

For a smoothness-first 120 Hz test, add this option before
`--experimental-framegen`:

```bash
  --framerate-limit 30 \
```

This asks gamescope to release client frame callbacks on every fourth display
vblank. When the client and compositor meet those callbacks, each real frame plus
its three generated frames aligns to four display vblanks. It is vendor-agnostic
and does not change reconstruction quality, but it deliberately caps the game, so
it adds input latency when the game could otherwise render above 30 fps. Keep it
optional rather than treating it as a general default.

The simple limit persists across repaints and display changes. A combined refresh
override takes precedence while it also controls the FPS cap; clearing that
override restores the simple limit. `--framerate-limit 0` disables it. The
`GAMESCOPE_FPS_LIMIT` X11 property updates this same limit.

The limiter cannot recover a missed client render or a late compositor wakeup. A
nested test running without `CAP_SYS_NICE` may still measure three- or five-vblank
pairs or longer gaps around the intended four-vblank cadence. At x4 a five-vblank
gap necessarily leaves one repeat, because only three generated frames are
available. Treat the limiter as a controlled pacing test, not as the frozen
visual baseline or a guaranteed stutter fix.
