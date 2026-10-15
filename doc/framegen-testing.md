# Frame-generation nested runtime testing

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

## Fault-injection grammar

Both knobs share the strict grammar of `src/Utils/FaultInjection.h`:

- `kind` must match the whole token.
- `N` is a positive decimal count with no sign, whitespace, or trailing text;
  `0` is rejected. Without `N` the injection never ends.
- An invalid spec is logged and ignored, so the run has no injection. An
  active spec is logged once.

Only the markers quoted under "What each case proves" are parsed; other log
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

## What each case proves

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
    change and two after it. A correct build passes every check in three of
    three runs. A judged window below the fill floor is a controller defect;
    the floor stays.
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
to the actual flip) are metrics only; no gate reads them. Each has `_p50`,
`_p90`, `_p95`, and the histogram's full scale `_fs` (256 ms); a percentile at
full scale only says the value is somewhere beyond it. An empty distribution
prints 0.000.

These are five-second window counters. `real`, `gen`, and `dreal` count
presented fresh content, while `rep` counts repeated scanout. A window with
zero `real`, `gen`, and `dreal` inside the injected timeout therefore provides
the completion-publication check without treating legitimate hardware repeats
or frame-generation dispatch logs as presentation proof.

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

## Shutdown signal

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

The AMD rows select the GPU with `--prefer-vk-device 1002:150e`, the NVIDIA
rows with `--prefer-vk-device 10de:2db9`; each log confirms the selected
device. "bug-064" is the known NVIDIA SIGSEGV from static Vulkan destructors at
exit (exit status 139 about 1.2 s after `Primary child shut down!`), which this
fork avoids by exiting without running them; it is unrelated to the X11 I/O
abort and is counted separately. A zero is a count over the stated runs, not
proof of absence. The X11 I/O abort occurs only under group signalling, in
upstream and in the release candidate at similar rates (4 in 20 and 13 in 60),
and never under pid-only signalling (0 in 20 and 0 in 60).

One cell's loop, run from the repository root:

```bash
for i in $(seq 10); do
  rd=$(mktemp -d -p /tmp fgtXXXX)
  env -u DISPLAY -u WAYLAND_DISPLAY XDG_RUNTIME_DIR=$rd GAMESCOPE_ZENITY_DISABLE=1 \
    ./scripts/env-gamescope-local.sh build/src/gamescope --prefer-vk-device 1002:150e \
    --backend headless -W 1280 -H 720 -r 60 -- xlogo > term-$i.log 2>&1 &
  pid=$!; sleep 4
  kill -TERM $pid    # group mode: run "set -m" first, then kill -TERM -- -$pid
  wait $pid; echo "run $i rc=$?"; rm -rf "$rd"
done
grep -l 'X11 I/O error' term-*.log
```

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
