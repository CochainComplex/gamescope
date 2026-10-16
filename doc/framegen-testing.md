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

The last two sections cover the end-to-end showcase compare
(`scripts/framegen-showcase.sh`), which judges a candidate build against a
baseline on real content, and the offline split of prediction error into
matching and continuation (`scripts/framegen-hindsight-eval.py`).

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

`ring_ovf` counts the bidir queue fronts that the output-ring ceiling forced
out ahead of their planned slot in the window. Each one is an overflow that
the bidir ring bound (`D_ring`) learns from; it stays 0 outside bidir.

The `q_*` fields come from the quality meter. It only measures and never
changes a generated frame. It runs for every motion pipeline when
`GAMESCOPE_FRAMEGEN_METRICS=1` or `--framegen-debug` is set; on `learned` and
above with adaptation on, the adaptation probe supplies the same counters.
Each real frame grades the field that was estimated for it by warping the
previous frame's block luma along the final field and comparing it with the
real frame. Only moving field texels count (|mv| of at least 0.25 field
texels); a static background predicts itself whatever the field says. Values
are ratios of sums over the window, so every moving texel weighs the same:

- `q_resid`: mean normalized luma residual of the warp against the real frame.
- `q_bad_mv`: share of the moving texels that survived the checks whose
  residual still exceeds the mispredict threshold (bad vectors that were kept).
- `q_killed_mv`: share of moving texels whose final confidence fell below the
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

## End-to-end showcase and baseline compare

`scripts/framegen-showcase.sh` runs a fixed table of ten framegen
configurations (framegen off, predictive, bidir, each pipeline) against one
client, with the HUD and `GAMESCOPE_FRAMEGEN_METRICS=1` on, and judges a
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
| `GM_DIR` | `$HOME/GravityMark/bin` | GravityMark's `bin` directory for the default client. |
| `CLIENT_PROC` | `GravityMark.x64` without `CLIENT`, else empty | The client's kernel process name (its first 15 characters), used by the render device check. |
| `GAMESCOPE_BIN` | `build/src/gamescope` | Binary under test. Its build directory is passed as `GAMESCOPE_BUILD_DIR`, so the WSI layer and the reaper come from the same build. |
| `OUT` | `$XDG_RUNTIME_DIR/framegen-showcase/<date-time>` | Session directory: `run<i>-r<rep>.log` and `results.json`. |
| `DEBUG` | 0 | `1` adds `--framegen-debug` to framegen runs. Diagnosis only; refused together with `--baseline` or `--compare`. |
| `EXTRA_ENV` | empty | Extra gamescope environment for every run. |

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
- `dl_hit` and `bias_ms` are medians over windows that generated or delayed a
  frame; framegen off shows `-`. `reprime`, `ring_ovf` and `q_samples` are
  summed. Each `q_*` is the median over windows with `q_samples` > 0.
- The run's value is the median over its reps that produced a steady window.

A rep is complete when `exit=124` (the run lasted its full length) and it has a
steady window. A missing field is shown as `-` (table) or `n/a` (compare) and is
never gated: older binaries do not print `q_*` or `ring_ovf`.

Compare gates, per run present in both sessions:

| Factor | Regression when the candidate |
|---|---|
| `fill` | falls by more than 0.01 |
| `flip_ms_sd` (pooled) | rises by more than 0.5 ms |
| `lat_p50` | rises by more than 1 ms |
| each `q_*` | exceeds the baseline by more than 5% of it (a zero baseline flags any increase) |
| completeness | has an incomplete rep where the baseline run had none |

`real_s`, `shown_s`, `dl_hit`, `bias_ms`, `reprime`, `ring_ovf` and `q_samples`
are reported, not gated. Exit 0 is a pass, 1 a regression. Exit 2 means the
sessions are not comparable: a context key differs (`backend`, `size`, `hz`,
`render`, `present`, `client`, `seconds`, `warmup`, `debug`), either side is a
DEBUG session, a client was not seen on its render device, or a run's flags,
run environment or latency start point differ. The treatment (`bin`, gamescope
`version`, `EXTRA_ENV`) is printed for both sides and never compared: it is
what is under test, so an opt-in knob can be judged with one binary. One
session holds one treatment; summarising logs of mixed binaries is refused.

Examples (`--compare` needs the baseline's context, so repeat its variables):

```bash
# Baseline from one build, then a candidate from another.
GAMESCOPE_BIN=../build-base/src/gamescope scripts/framegen-showcase.sh --baseline base.json
GAMESCOPE_BIN=build/src/gamescope scripts/framegen-showcase.sh --compare base.json

# Opt-in A/B on one binary: the knob is the treatment.
scripts/framegen-showcase.sh --baseline a.json 3 6
EXTRA_ENV=GAMESCOPE_FRAMEGEN_ADAPT=0 scripts/framegen-showcase.sh --compare a.json 3 6

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
- Latency percentiles are the upper edges of 1 ms buckets, so the +1 ms gate
  tolerates a one-bucket shift and a sub-millisecond regression is invisible.
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

## Offline prediction-error split (S1/S2)

`scripts/framegen-hindsight-eval.py` splits the predictive path's error on a
recorded capture into the part a better matcher could remove and the part only
a better predictor could. Capture in motion mode with
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
- `oracle`: frame N warped along the true next field. This is S1, the matching
  error.
- S2 = `cont` − `oracle`, the continuation error, and its share of `cont`.

The rejection block looks at the texels moving in the judged field only (the
field generation uses, so its moving set is narrower than the split's): of the
moving texels the field rejects (confidence below 0.5), how often warping would
have beaten holding, the mean gain, the p95 loss, how often warping wins on the
kept texels, and the shares of confidence below 0.1 and between 0.1 and 0.5.

Scope: field-resolution luma and absolute error, not a colour-domain perceptual
score, the same limit as `framegen-net-eval.py`. A continuation teacher is
worth building only when S2 is at least 30% of the predictive error on two
contents.
