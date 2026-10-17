#!/usr/bin/env python3
"""Nested frame-generation runtime smoke and fault-injection harness."""

from __future__ import annotations

import argparse
import ctypes
import functools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import BinaryIO, Callable, Iterable, TextIO


SCENARIOS = (
    "baseline",
    "present-timeout",
    "present-error",
    "sdl-error",
    "sdl-error-pumped",
    "bidir",
)
FRAMEGEN_ENV_PREFIX = "GAMESCOPE_FRAMEGEN"
PRESENT_INJECT_ENV = "GAMESCOPE_DEBUG_PRESENT_WAIT_INJECT"
SDL_INJECT_ENV = "GAMESCOPE_DEBUG_SDL_WAIT_INJECT"
FAULT_ENV = (PRESENT_INJECT_ENV, SDL_INJECT_ENV)
DISABLE_WSI_ENV = "DISABLE_GAMESCOPE_WSI"
SELF = (sys.executable, str(Path(__file__).resolve()))
METRIC_RE = re.compile(
    r"framegen-metrics:\s+(?P<label>\S+)\s+real=(?P<real>\d+)\s+"
    r"gen=(?P<gen>\d+)\s+dreal=(?P<dreal>\d+)\s+rep=(?P<rep>\d+)"
)
TIMESTAMP_RE = re.compile(r"^\[(?P<elapsed>\d+(?:\.\d+)?)\]\s(?P<text>.*)$")
BAD_WORD_RE = re.compile(r"abort|terminate|corrupted", re.IGNORECASE)
METRIC_WINDOW_SECONDS = 5.0
FRAMEGEN_ENABLED_MARKER = "framegen: enabled mode="
# Baseline throughput gate: the gen > 0 floor passes 10-30% judder.
MAX_SLOW_DROP_RATIO = 0.05  # generation_too_slow drops per real frame, per window
MIN_FILL = 0.95  # fresh / all presents; framegen_metrics_fill_rate()
SLOW_RE = re.compile(r"\sslow=(?P<slow>\d+)")  # the total; never slow_ev= etc.
DEFAULT_HOLD_SECONDS = 18.0
FRAMERATE_LIMIT = 30  # the inner's --framerate-limit
REFRESH_HZ = 60  # every output in the nested stack runs at this rate
# Shutdown proof. The present waiter exists only on Vulkan-swapchain (SDL)
# backends; vulkan_framegen_shutdown() logs the TOTAL line on any backend.
WAITER_JOINED_MARKER = "present_wait: waiter joined"
FRAMEGEN_SHUTDOWN_MARKER = "framegen-metrics: TOTAL"
# present-error: the cost of one injected failure episode.
REMAKE_MARKER = "Creating Gamescope nested swapchain"
RESET_RE = re.compile(r"framegen: reset history reason=(?P<reason>\S+)")
# bidir: a judged window before the focus change, the settling window, two after.
BIDIR_HOLD_SECONDS = 27.0
BIDIR_SECOND_CLIENT_AFTER = 11.0  # seconds after the first vkcube
# A bidir run is judged only while the client holds its --framerate-limit.
FRAMERATE_CAP_TOLERANCE = 0.10
# H: the warm-up after the bidir enable, and the settling allowance after a
# timeline transition. It must cover production's envelope seed window
# (k_ulBidirWindowNs, 1 s) plus the establishment that follows it.
BIDIR_SETTLE_SECONDS = 2.0
BIDIR_MAX_REPRIMES = 1  # stray timeline rebuilds per run
BIDIR_DELAY_FIELD = "bidir_delay_ms"  # the largest target delay D in force per window
# Share of consecutive settled real frames whose present spacing must match
# their record spacing within T; the rest absorbs a lone early or late show.
BIDIR_MIN_EVEN_SPACING = 0.99
# The output's refresh interval T in nanoseconds.
VBLANK_NS = round(1e9 / REFRESH_HZ)
# Log time against the logged event: 1 ms timestamp resolution plus the log
# pump's lag, measured at most 1.4 ms.
LOG_SLACK_NS = 2_000_000
# A process still running this long into a shutdown wait has its stacks saved,
# ahead of gamescope's own five-second shutdown watchdog.
STACK_CAPTURE_AFTER_SECONDS = 3.0
STACK_CAPTURE_TIMEOUT_SECONDS = 20
PR_SET_PTRACER = 0x59616D61  # prctl(): Yama's ptracer exception
HOST_DISPLAY = "fgsmoke-host"
BIDIR_MARKER = "framegen: bidirectional interpolation requested"
FOCUS_MARKER = "framegen: history valid=false reason=focus_change"
PRIME_RE = re.compile(r"framegen: priming history with real frame id=(?P<id>\d+)")
SDL_WAIT_FAILURE_RE = re.compile(r"\bSDL_WaitEvent(?:Timeout)? failed: injected$")
REAL_FRAME_RE = re.compile(r"framegen: real frame id=(?P<id>\d+) time=(?P<ns>\d+)")
PRESENTED_CONTENT_RE = re.compile(
    r"framegen: presented (?P<kind>delayed real|live real|generated) frame id=(?P<id>\d+)"
    r"\.(?P<phase>\d+)"
)
REPRIME_RE = re.compile(r"\sreprime=(?P<n>\d+)")
# Cuts: a history invalidation or reset. Only a cut or a backlog drop may drop
# a queued real frame; every cut and rebuild opens a settling allowance.
DISCONTINUITY_RE = re.compile(r"framegen: (?:history valid=false|reset history) reason=")
SNAP_MARKER = "framegen: bidir timeline snapped to live"
REBUILD_RE = re.compile(r"framegen: bidir (?:.*re-primed|timeline stale|timeline snapped to live)")
REPRIME_LINE_RE = re.compile(r"framegen: bidir .*re-primed")
# The only drop of queued real frames outside a cut that a source no faster
# than the display may cause: a backlog whose lateness neither shrank nor grew
# for a second drops the real frames of its excess. It keeps D and is no
# rebuild. Queue capacity alone never drops a real frame of such a source.
BACKLOG_DROP_RE = re.compile(
    r"framegen: bidir dropped \d+ queued real\(s\) \(backlog did not drain\)"
)
EPOCH_MARKER = "framegen: bidir epoch="  # one line per bidir plan
METRIC_FIELD_RE = re.compile(r"\s(?P<name>\w+)=(?P<value>-?\d+(?:\.\d+)?)(?=\s|$)")


@dataclass
class Result:
    scenario: str
    passed: bool
    evidence: str
    errors: list[str] = field(default_factory=list)
    shutdown_evidence: str | None = None


def shell_join(parts: Iterable[str]) -> str:
    return shlex.join([str(part) for part in parts])


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_text(temporary, json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid_file(path: Path, deadline: float) -> int:
    while time.monotonic() < deadline:
        try:
            pid = int(path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)
            continue
        if pid_alive(pid):
            return pid
        raise RuntimeError(f"recorded process {pid} from {path} is no longer running")
    raise TimeoutError(f"timed out waiting for {path}")


def poll_until(done: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if done():
            return True
        time.sleep(0.05)
    return done()


def allow_stack_capture(tracer_pid: int) -> None:
    """Let `tracer_pid` and its descendants ptrace the calling process.

    Under Yama ptrace_scope 1 only an ancestor may attach otherwise. The
    setting survives exec; without Yama the call fails harmlessly.
    """
    ctypes.CDLL(None, use_errno=True).prctl(
        ctypes.c_int(PR_SET_PTRACER),
        ctypes.c_ulong(tracer_pid),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )


def gdb_stacks(pid: int) -> str | None:
    """Every thread's backtrace from gdb, or None when gdb is absent or cannot attach."""
    gdb = shutil.which("gdb")
    if gdb is None:
        return None
    try:
        completed = subprocess.run(
            [gdb, "-nx", "-batch", "-p", str(pid), "-ex", "thread apply all bt"],
            env={**os.environ, "DEBUGINFOD_URLS": ""},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=STACK_CAPTURE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None
    output = completed.stdout + completed.stderr
    # A failed attach still exits 0, so success is a printed frame.
    return output if "#0 " in output else None


def proc_stacks(pid: int) -> str:
    """Every thread's state from /proc, for when gdb cannot attach."""
    tasks = Path(f"/proc/{pid}/task")
    try:
        tids = sorted(os.listdir(tasks), key=int)
    except OSError as exc:
        return f"{tasks}: <unreadable: {exc.errno}>\n"
    lines = [f"gdb could not attach to {pid}; /proc task state follows"]
    for tid in tids:
        lines.append(f"Thread {tid}")
        for name in ("comm", "stat", "wchan", "syscall", "stack"):
            try:
                value = (tasks / tid / name).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                value = f"<unreadable: {exc.errno}>"
            lines.append(f"  {name}: " + value.strip().replace("\n", "\n    "))
    return "\n".join(lines) + "\n"


def capture_stacks(pid: int, path: Path) -> str:
    write_text(path, gdb_stacks(pid) or proc_stacks(pid))
    return f"stacks saved to {path}"


def await_or_capture(
    done: Callable[[], bool], pid: int | None, path: Path, grace: float
) -> tuple[bool, str | None]:
    """Wait up to `grace` seconds for `done()`.

    If it is not done after STACK_CAPTURE_AFTER_SECONDS, save the stacks of
    `pid` (when it is still running) before waiting out the rest. The capture
    counts toward `grace`. Returns whether it finished and the capture note.
    """
    started = time.monotonic()
    if poll_until(done, min(STACK_CAPTURE_AFTER_SECONDS, grace)):
        return True, None
    capture = capture_stacks(pid, path) if pid is not None and pid_alive(pid) else None
    return poll_until(done, grace - (time.monotonic() - started)), capture


def timestamped_copy(stream: TextIO, output: TextIO, started: float) -> None:
    for line in stream:
        output.write(f"[{time.monotonic() - started:09.3f}] {line}")
        output.flush()


def env_assignments(env: dict[str, str]) -> list[str]:
    return [f"{name}={value}" for name, value in env.items()]


def client_command(
    state_dir: Path | str,
    client_log: Path | str,
    vkcube: str,
    second_client_after: float | None = None,
) -> list[str]:
    command = [
        *SELF,
        "--internal-client",
        "--state-dir", str(state_dir),
        "--client-log", str(client_log),
        "--vkcube", vkcube,
    ]
    if second_client_after is not None:
        command.extend(("--second-client-after", str(second_client_after)))
    return command


def inner_command(
    binary: str, state_dir: Path | str, client_log: Path | str, vkcube: str, bidir: bool
) -> list[str]:
    # Bidir needs an inner that owns its output ring; SDL never does.
    second_client_after = BIDIR_SECOND_CLIENT_AFTER if bidir else None
    return [
        binary,
        "--backend", "wayland" if bidir else "sdl",
        "-w", "640", "-h", "480", "-W", "640", "-H", "480", "-r", str(REFRESH_HZ),
        "--experimental-framegen", "--framegen-mode", "motion", "--framegen-pipeline", "checked",
        "--framegen-multiplier", "2", "--framerate-limit", str(FRAMERATE_LIMIT),
        "--framegen-debug",
        "--", *client_command(state_dir, client_log, vkcube, second_client_after),
    ]


def inner_env(present_inject: str | None, sdl_inject: str | None, bidir: bool) -> dict[str, str]:
    env = {
        "SDL_VIDEODRIVER": "wayland",
        DISABLE_WSI_ENV: "1",
        "GAMESCOPE_ZENITY_DISABLE": "1",
        "GAMESCOPE_FRAMEGEN_METRICS": "1",
        "GAMESCOPE_FRAMEGEN_DEBUG_EVERY": "60",
    }
    if bidir:
        # The real-frame, present, prime, and rebuild lines are sampled by
        # FramegenDebugShouldLog(); the bidir gates need every one of them.
        env["GAMESCOPE_FRAMEGEN_BIDIR"] = "1"
        env["GAMESCOPE_FRAMEGEN_DEBUG_EVERY"] = "1"
    if present_inject:
        env[PRESENT_INJECT_ENV] = present_inject
    if sdl_inject:
        env[SDL_INJECT_ENV] = sdl_inject
    return env


def internal_client(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    log_path = Path(args.client_log)
    write_text(state_dir / "client.pid", f"{os.getpid()}\n")
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    env = os.environ.copy()
    env.pop(DISABLE_WSI_ENV, None)
    if args.second_client_after is not None and os.fork() == 0:
        # gamescope never setsids its children, so this second client stays in
        # the supervisor's process group and needs no extra cleanup.
        time.sleep(args.second_client_after)
        log_fd = os.open(
            log_path.with_name("client2.log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(log_fd)
    os.execvpe(args.vkcube, [args.vkcube], env)
    return 127


def internal_inner(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    log_path = Path(args.inner_log)
    status_path = state_dir / "inner-status.json"
    command = inner_command(args.binary, state_dir, args.client_log, args.vkcube, args.bidir)
    env = {**os.environ, **inner_env(args.present_inject, args.sdl_inject, args.bidir)}

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            preexec_fn=(
                None if args.ptracer is None
                else functools.partial(allow_stack_capture, args.ptracer)
            ),
        )
        write_text(state_dir / "inner.pid", f"{process.pid}\n")

        def forward(signum: int, _frame: object) -> None:
            if process.poll() is None:
                process.send_signal(signum)

        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        assert process.stdout is not None
        pump = threading.Thread(
            target=timestamped_copy, args=(process.stdout, output, started), daemon=True
        )
        pump.start()
        returncode = process.wait()
        pump.join(timeout=1.0)

    write_json_atomic(
        status_path,
        {
            "returncode": returncode,
            "runtime_seconds": time.monotonic() - started,
            "finished_monotonic": time.monotonic(),
        },
    )
    return returncode if returncode >= 0 else 128 - returncode


def log_records(text: str) -> list[tuple[float, str]]:
    records: list[tuple[float, str]] = []
    for line in text.splitlines():
        match = TIMESTAMP_RE.match(line)
        if match:
            records.append((float(match.group("elapsed")), match.group("text")))
    return records


def marker_record(records: list[tuple[float, str]], marker: str) -> tuple[float, str] | None:
    return next(((elapsed, text) for elapsed, text in records if marker in text), None)


def last_log_line(text: str) -> str | None:
    lines = text.splitlines()
    return lines[-1] if lines else None


def format_shutdown_evidence(inner_log: str, last_pre_sigterm: str | None) -> str:
    tail = inner_log.splitlines()[-15:]
    before = last_pre_sigterm or "<inner log was empty before SIGTERM>"
    rendered_tail = "\n".join(f"    {line}" for line in tail)
    if not rendered_tail:
        rendered_tail = "    <inner log is empty>"
    return (
        f"last inner log line before SIGTERM: {before}\n"
        f"last 15 inner log lines:\n{rendered_tail}"
    )


def metric_records(records: list[tuple[float, str]]) -> list[tuple[float, dict[str, int], str]]:
    parsed: list[tuple[float, dict[str, int], str]] = []
    for elapsed, text in records:
        match = METRIC_RE.search(text)
        if not match or match.group("label") == "total":
            continue
        counts = {name: int(match.group(name)) for name in ("real", "gen", "dreal", "rep")}
        parsed.append((elapsed, counts, text.strip()))
    return parsed


def metric_fields(line: str) -> dict[str, float]:
    """Every name=number field of a metrics line, by name."""
    return {m.group("name"): float(m.group("value")) for m in METRIC_FIELD_RE.finditer(line)}


def nearest_rank(values: list[int], percent: int) -> int:
    """The nearest-rank percentile, the rule MetricsDistribution_t::percentile() uses."""
    ordered = sorted(values)
    return ordered[max(0, (len(ordered) * percent + 99) // 100 - 1)]


def log_cutoff(text: str) -> float:
    """Elapsed time of the last timestamped line in `text` (0.0 if none)."""
    records = log_records(text)
    return records[-1][0] if records else 0.0


def judged_windows(
    metrics: list[tuple[float, dict[str, int], str]],
    since: float,
    sigterm_elapsed: float,
) -> list[tuple[float, dict[str, int], str]]:
    """Complete windows that start at or after `since` and end before SIGTERM.

    The window in progress at `since` may hold frames from before it and is
    the settling window, so it is never judged.
    """
    return [
        item
        for item in metrics
        if item[0] <= sigterm_elapsed and item[0] - METRIC_WINDOW_SECONDS >= since
    ]


def generation_errors(
    metrics: list[tuple[float, dict[str, int], str]],
    event: str,
    since: float,
    sigterm_elapsed: float,
) -> list[str]:
    """Require gen > 0 in every judged window from `since` until SIGTERM.

    Every window logged before SIGTERM is judged, so generation must both
    resume and still run in the last of them.
    """
    judged = judged_windows(metrics, since, sigterm_elapsed)
    if not judged:
        return [
            f"no complete metrics window starts at or after {since:.3f}s and ends "
            "before SIGTERM (hold too short or run ended early; raise --hold-seconds)"
        ]
    return [
        f"framegen generation stayed at 0 after {event} ({since:.3f}s): "
        f"the window ending at {elapsed:.3f}s has real={counts['real']} gen=0"
        for elapsed, counts, _ in judged
        if counts["gen"] == 0
    ]


def throughput_errors(
    metrics: list[tuple[float, dict[str, int], str]],
    since: float,
    sigterm_elapsed: float,
    min_fill: float | None = MIN_FILL,
) -> list[str]:
    """Require full-rate generation in every judged window.

    Drops are judged per fresh real frame, real + dreal (bidir presents its
    reals as dreal). `min_fill=None` skips the fill gate. An empty selection
    is reported by the caller.
    """
    errors: list[str] = []
    fill_gate = "" if min_fill is None else f", fill >= {min_fill:g}"
    for elapsed, counts, line in judged_windows(metrics, since, sigterm_elapsed):
        match = SLOW_RE.search(line)
        if match is None:
            errors.append(f"metrics window ending at {elapsed:.3f}s has no slow= field")
            continue
        slow = int(match.group("slow"))
        reals = counts["real"] + counts["dreal"]
        # Same definition as framegen_metrics_fill_rate() / the HUD fill.
        fresh = reals + counts["gen"]
        presented = fresh + counts["rep"]
        fill = fresh / presented if presented else 0.0
        if slow > MAX_SLOW_DROP_RATIO * reals or (min_fill is not None and fill < min_fill):
            errors.append(
                "framegen throughput below the gate "
                f"(slow/(real+dreal) <= {MAX_SLOW_DROP_RATIO:g}{fill_gate}): "
                f"the window ending at {elapsed:.3f}s has real={counts['real']} "
                f"dreal={counts['dreal']} gen={counts['gen']} slow={slow} fill={fill:.3f}"
            )
    return errors


@dataclass(frozen=True)
class RealFrame:
    """One recorded real frame of the delayed timeline."""

    key: tuple[int, int]  # (history-reset segment, id): ids restart at a reset
    recorded: float  # log time of its "real frame" line
    time_ns: int  # its time= stamp
    shown: float | None  # log time of its first present, delayed or live (None: never)

    @property
    def delay_ns(self) -> int | None:
        return None if self.shown is None else round((self.shown - self.recorded) * 1e9)


@dataclass(frozen=True)
class BidirTimeline:
    """What a bidir log recorded, presented and declared."""

    reals: list[RealFrame]  # in record order
    contents: list[float]  # first present of each distinct real or generated frame
    order_errors: list[tuple[float, int, int]]  # (log time, id, highest id already shown)
    cuts: list[float]  # history invalidations and resets
    rebuilds: list[float]  # re-primes, stale timelines and snaps to live
    reprimes: list[float]  # the re-prime lines among the rebuilds
    backlog_drops: list[float]  # backlogs that did not drain dropping their excess

    @property
    def transitions(self) -> list[float]:
        """Where the timeline restarts; each opens a settling allowance."""
        return sorted(self.cuts + self.rebuilds)

    @property
    def drop_excuses(self) -> list[float]:
        """The only lines that may drop a queued real frame."""
        return sorted(self.cuts + self.backlog_drops)


def bidir_timeline(records: list[tuple[float, str]]) -> BidirTimeline:
    """Scan a bidir log once.

    Frame ids restart at a history reset, so frames are keyed by reset segment.
    Within a segment every real frame, delayed or live, must be presented after
    every lower id and never again; a present that breaks that is an order error.
    """
    segment = 0
    highest: int | None = None
    recorded: dict[tuple[int, int], tuple[float, int]] = {}
    shown: dict[tuple[int, int], float] = {}
    contents: dict[tuple[int, bool, int, str], float] = {}
    order_errors: list[tuple[float, int, int]] = []
    cuts: list[float] = []
    rebuilds: list[float] = []
    reprimes: list[float] = []
    backlog_drops: list[float] = []
    for elapsed, text in records:
        if DISCONTINUITY_RE.search(text):
            cuts.append(elapsed)
            if RESET_RE.search(text):
                segment, highest = segment + 1, None
        elif BACKLOG_DROP_RE.search(text):
            backlog_drops.append(elapsed)
        elif REBUILD_RE.search(text):
            rebuilds.append(elapsed)
            if REPRIME_LINE_RE.search(text):
                reprimes.append(elapsed)
        elif match := REAL_FRAME_RE.search(text):
            recorded.setdefault(
                (segment, int(match.group("id"))), (elapsed, int(match.group("ns")))
            )
        elif match := PRESENTED_CONTENT_RE.search(text):
            generated, frame_id = match.group("kind") == "generated", int(match.group("id"))
            contents.setdefault((segment, generated, frame_id, match.group("phase")), elapsed)
            if generated:
                continue
            if highest is not None and frame_id <= highest:
                order_errors.append((elapsed, frame_id, highest))
            else:
                highest = frame_id
            shown.setdefault((segment, frame_id), elapsed)
    return BidirTimeline(
        reals=[
            RealFrame(key, at, time_ns, shown.get(key))
            for key, (at, time_ns) in sorted(recorded.items())
        ],
        contents=sorted(contents.values()),
        order_errors=order_errors,
        cuts=cuts,
        rebuilds=rebuilds,
        reprimes=reprimes,
        backlog_drops=backlog_drops,
    )


def measured_client_fps(reals: list[RealFrame], since: float, end: float) -> float:
    """The client's frame rate from the time= stamps of its recorded real frames.

    Sums the id and time= steps between consecutive real frames of one reset
    segment that were both recorded in [since, end].
    """
    frames = elapsed_ns = 0
    for earlier, later in zip(reals, reals[1:]):
        if (
            earlier.key[0] == later.key[0]
            and since <= earlier.recorded
            and later.recorded <= end
            and later.time_ns > earlier.time_ns
        ):
            frames += later.key[1] - earlier.key[1]
            elapsed_ns += later.time_ns - earlier.time_ns
    return frames * 1e9 / elapsed_ns if elapsed_ns else 0.0


def window_fps(reals: list[RealFrame], window_end: float) -> float:
    """The client's frame rate in the metrics window ending at window_end."""
    return measured_client_fps(reals, window_end - METRIC_WINDOW_SECONDS, window_end)


def holds_cap(reals: list[RealFrame], window_end: float) -> bool:
    """The client held FRAMERATE_LIMIT within the tolerance in the window ending at window_end."""
    fps = window_fps(reals, window_end)
    return abs(fps - FRAMERATE_LIMIT) <= FRAMERATE_CAP_TOLERANCE * FRAMERATE_LIMIT


def window_fill(timeline: BidirTimeline, window_end: float) -> tuple[int, float]:
    """Distinct new frames presented in the window ending at window_end, and that per refresh.

    Counted from the present lines, so a re-flip of a held frame never counts
    and the nested host's wake rate does not matter.
    """
    count = sum(
        window_end - METRIC_WINDOW_SECONDS < at <= window_end for at in timeline.contents
    )
    return count, count / (METRIC_WINDOW_SECONDS * REFRESH_HZ)


def settled_windows(
    metrics: list[tuple[float, dict[str, int], str]],
    timeline: BidirTimeline,
    since: float,
    sigterm_elapsed: float,
) -> list[tuple[float, dict[str, int], str]]:
    """Judged windows with no transition from H before their start to their end.

    H is BIDIR_SETTLE_SECONDS; the transitions are the cuts and rebuilds, the
    same set that opens the delay and gap allowances.
    """
    return [
        item
        for item in judged_windows(metrics, since, sigterm_elapsed)
        if not any(
            item[0] - METRIC_WINDOW_SECONDS - BIDIR_SETTLE_SECONDS < at <= item[0]
            for at in timeline.transitions
        )
    ]


def delay_windows(
    metrics: list[tuple[float, dict[str, int], str]],
) -> tuple[list[tuple[float, float]], list[str]]:
    """(window end, D in ms) for every periodic metrics window, and contract errors.

    D is the window's BIDIR_DELAY_FIELD. If any window lacks it, no window is
    returned and the gates bounded by D are skipped.
    """
    windows: list[tuple[float, float]] = []
    lacking = 0
    unset: list[float] = []
    for window_end, counts, line in metrics:
        delay_ms = metric_fields(line).get(BIDIR_DELAY_FIELD)
        if delay_ms is None:
            lacking += 1
            continue
        if counts["dreal"] > 0 and delay_ms == 0:
            unset.append(window_end)
        windows.append((window_end, delay_ms))
    errors: list[str] = []
    if lacking:
        errors.append(f"{lacking}/{len(metrics)} metrics windows lack {BIDIR_DELAY_FIELD}")
        windows = []
    if unset:
        errors.append(
            f"{len(unset)} metrics windows presented delayed reals with {BIDIR_DELAY_FIELD}=0, "
            f"first ending at {unset[0]:.3f}s"
        )
    return windows, errors


def delay_bound_ns(windows: list[tuple[float, float]], start: float, end: float) -> int | None:
    """D + T over [start, end]: the largest D of the windows overlapping it, plus T.

    A window covers the METRIC_WINDOW_SECONDS up to its end. None when no
    window overlaps the span, which is then not judged.
    """
    values = [
        delay_ms
        for window_end, delay_ms in windows
        if window_end - METRIC_WINDOW_SECONDS < end and window_end >= start
    ]
    return round(max(values) * 1e6) + VBLANK_NS if values else None


def settling(transitions: list[float], start: float, end: float) -> bool:
    """A transition lies from BIDIR_SETTLE_SECONDS before start to end."""
    return any(start - BIDIR_SETTLE_SECONDS <= at <= end for at in transitions)


def bidir_sequence_errors(
    records: list[tuple[float, str]],
    focus: float,
    cutoff: float,
    windows: list[tuple[float, float]],
) -> tuple[list[str], str]:
    """Judge the bidir timeline across one focus-change invalidation.

    The prime must be shown, delayed or live, within D + T of the priming line,
    with no snap to live after the focus change. Its present is the first real
    frame present logged after the priming line with an id at or past the
    prime's. If a newer base replaced the queued prime, content order makes
    that the first real frame shown; the unseen gate judges the replaced prime.
    """
    window = [(elapsed, text) for elapsed, text in records if focus <= elapsed <= cutoff]
    errors = [
        f"bidir timeline snapped to live at {elapsed:.3f}s, after the focus change at {focus:.3f}s"
        for elapsed, text in window
        if SNAP_MARKER in text
    ]
    prime = next(
        (
            (index, elapsed, match)
            for index, (elapsed, text) in enumerate(window)
            if (match := PRIME_RE.search(text))
        ),
        None,
    )
    if prime is None:
        errors.append(f"no prime after the focus change at {focus:.3f}s")
        return errors, f"focus {focus:.3f}s; no prime"
    prime_index, prime_at, prime_id = prime[0], prime[1], int(prime[2].group("id"))
    shown_at = next(
        (
            elapsed
            for elapsed, text in window[prime_index + 1 :]
            if (match := PRESENTED_CONTENT_RE.search(text))
            and match.group("kind") != "generated"
            and int(match.group("id")) >= prime_id
        ),
        None,
    )
    if shown_at is None:
        errors.append(f"bidir prime id {prime_id} primed at {prime_at:.3f}s was never presented")
        return errors, f"focus {focus:.3f}s; prime id {prime_id} never shown"
    delay = shown_at - prime_at
    bound = delay_bound_ns(windows, prime_at, shown_at)
    if bound is not None and delay * 1e9 > bound + LOG_SLACK_NS:
        errors.append(
            f"bidir prime id {prime_id} was presented {delay * 1000:.0f} ms after it was primed "
            f"(maximum {BIDIR_DELAY_FIELD} + T = {bound / 1e6:.1f} ms)"
        )
    return errors, f"focus {focus:.3f}s; prime id {prime_id} shown +{delay * 1000:.0f}ms"


def bidir_reprime_errors(
    timeline: BidirTimeline, enable: float, total_reprimes: int | None
) -> tuple[list[str], str]:
    """At most BIDIR_MAX_REPRIMES stray timeline rebuilds per run.

    A rebuild is stray when it is logged after the warm-up (BIDIR_SETTLE_SECONDS
    after the bidir enable) and outside a cut's own priming, which runs from the
    cut to the first real frame presented after it. Every re-prime counted by
    the TOTAL line's reprime= must also be logged.
    """
    shown = sorted(real.shown for real in timeline.reals if real.shown is not None)
    priming = [
        (cut, next((at for at in shown if at >= cut), math.inf)) for cut in timeline.cuts
    ]
    stray = [
        at
        for at in timeline.rebuilds
        if at > enable + BIDIR_SETTLE_SECONDS
        and not any(cut <= at <= first for cut, first in priming)
    ]
    errors: list[str] = []
    if len(stray) > BIDIR_MAX_REPRIMES:
        errors.append(
            f"bidir timeline rebuilt {len(stray)} times after warm-up outside a cut's priming "
            f"(maximum {BIDIR_MAX_REPRIMES}) at " + ", ".join(f"{at:.3f}s" for at in stray[:5])
        )
    if total_reprimes is not None and total_reprimes > len(timeline.reprimes):
        errors.append(
            f"TOTAL reprime={total_reprimes} but only {len(timeline.reprimes)} re-prime lines "
            "were logged"
        )
    return errors, f"rebuilds {len(stray)} stray/{len(timeline.rebuilds)}"


def bidir_unseen_errors(
    timeline: BidirTimeline, since: float, last: float
) -> tuple[list[str], str]:
    """Judge the real frames recorded in [since, last] that were never shown.

    One is unseen when a later real frame was shown and no cut or backlog
    drop line lies between its record and that present. Any unseen real
    frame fails the run.
    """
    excuses = timeline.drop_excuses
    judged = 0
    unseen: list[RealFrame] = []
    next_shown = math.inf
    for real in reversed(timeline.reals):
        if since <= real.recorded <= last:
            judged += 1
            if (
                real.shown is None
                and next_shown < math.inf
                and not any(real.recorded <= at <= next_shown for at in excuses)
            ):
                unseen.append(real)
        if real.shown is not None:
            next_shown = min(next_shown, real.shown)
    if not judged:
        return [f"no real frame was recorded from {since:.3f}s to {last:.3f}s"], "unseen 0/0"
    unseen.reverse()
    errors: list[str] = []
    if unseen:
        errors.append(
            f"bidir passed over {len(unseen)}/{judged} real frames with no cut or backlog drop "
            "line before a later real frame was shown; first at "
            + ", ".join(f"{real.recorded:.3f}s" for real in unseen[:5])
        )
    return errors, f"unseen {len(unseen)}/{judged}"


def bidir_order_errors(timeline: BidirTimeline) -> tuple[list[str], str]:
    """Delayed reals must be presented in increasing id order, each once."""
    offenders = timeline.order_errors
    errors = []
    if offenders:
        errors.append(
            f"bidir presented {len(offenders)} delayed real frames out of order or again: "
            + ", ".join(
                f"id {frame_id} after id {highest} at {at:.3f}s"
                for at, frame_id, highest in offenders[:3]
            )
        )
    return errors, f"order {len(offenders)}"


def bidir_spacing_errors(
    timeline: BidirTimeline, since: float, last: float
) -> tuple[list[str], str]:
    """Consecutive settled real frames are shown at their record spacing, within T.

    Judged are consecutive real frames of one reset segment, both recorded in
    [since, last] and both shown, with no transition from BIDIR_SETTLE_SECONDS
    before the earlier record to the later present. Their present spacing may
    differ from their time= spacing by more than T for at most
    1 - BIDIR_MIN_EVEN_SPACING of the pairs; the delay and gap gates bound how
    far.
    """
    transitions = timeline.transitions
    offsets: list[tuple[int, RealFrame]] = []
    for earlier, later in zip(timeline.reals, timeline.reals[1:]):
        if (
            earlier.key[0] != later.key[0]
            or earlier.shown is None
            or later.shown is None
            or earlier.recorded < since
            or later.recorded > last
            or settling(transitions, earlier.recorded, later.shown)
        ):
            continue
        offset = round((later.shown - earlier.shown) * 1e9) - (later.time_ns - earlier.time_ns)
        offsets.append((offset, later))
    if not offsets:
        return [
            "no settled pair of shown real frames to judge spacing by"
        ], "spacing: nothing judged"
    uneven = sorted(
        (item for item in offsets if abs(item[0]) > VBLANK_NS + LOG_SLACK_NS),
        key=lambda item: abs(item[0]),
        reverse=True,
    )
    errors: list[str] = []
    if len(uneven) > len(offsets) * (1 - BIDIR_MIN_EVEN_SPACING):
        errors.append(
            f"bidir spaced {len(uneven)}/{len(offsets)} consecutive real frames more than T "
            f"from their record spacing (at most {1 - BIDIR_MIN_EVEN_SPACING:.0%}); worst "
            + ", ".join(
                f"{offset / 1e6:+.1f} ms at {real.recorded:.3f}s" for offset, real in uneven[:3]
            )
        )
    worst = max(abs(offset) for offset, _ in offsets)
    return errors, f"spacing {len(uneven)} uneven/{len(offsets)} (worst {worst / 1e6:.1f} ms)"


def bidir_delay_errors(
    timeline: BidirTimeline, windows: list[tuple[float, float]], since: float, last: float
) -> tuple[list[str], str]:
    """Every settled real frame is shown within D + T of its record.

    Judged are the real frames recorded in [since, last] and shown with no
    transition from BIDIR_SETTLE_SECONDS before their record to their present.
    The delay runs from the record line to the first present line of that
    real frame, delayed or live, which is logged at the compositor opportunity
    that consumes it. D is the largest of the windows overlapping that span.
    """
    transitions = timeline.transitions
    delays: list[int] = []
    late: list[tuple[int, RealFrame]] = []
    for real in timeline.reals:
        if real.shown is None or not since <= real.recorded <= last:
            continue
        if settling(transitions, real.recorded, real.shown):
            continue
        bound = delay_bound_ns(windows, real.recorded, real.shown)
        if bound is None:
            continue
        delays.append(real.delay_ns)
        if real.delay_ns > bound + LOG_SLACK_NS:
            late.append((real.delay_ns - bound, real))
    if not delays:
        return ["no settled shown real frame to judge the delay by"], "delay: nothing judged"
    errors: list[str] = []
    if late:
        late.sort(key=lambda item: item[0], reverse=True)
        errors.append(
            f"bidir showed {len(late)}/{len(delays)} real frames later than "
            f"{BIDIR_DELAY_FIELD} + T after their record; worst "
            + ", ".join(
                f"+{over / 1e6:.1f} ms over {(real.delay_ns - over) / 1e6:.1f} ms "
                f"at {real.recorded:.3f}s"
                for over, real in late[:3]
            )
        )
    summary = "/".join(
        f"{value / 1e6:.0f}"
        for value in (
            min(delays),
            nearest_rank(delays, 50),
            nearest_rank(delays, 90),
            max(delays),
        )
    )
    return errors, f"delay min/p50/p90/max {summary} ms ({len(delays)} judged)"


def bidir_gap_errors(
    timeline: BidirTimeline, windows: list[tuple[float, float]], since: float, last: float
) -> tuple[list[str], str]:
    """No new content may be missing for longer than D + T from since to last.

    Across a transition the allowance is 2(D + T): up to D + T since the last
    present before it, then up to D + T from the prime's record to its present.
    """
    transitions = timeline.transitions
    edges = [since, *(at for at in timeline.contents if since <= at <= last), last]
    gaps: list[tuple[float, float]] = []
    for earlier, later in zip(edges, edges[1:]):
        bound = delay_bound_ns(windows, earlier, later)
        if bound is None:
            continue
        if any(earlier <= at <= later for at in transitions):
            bound *= 2
        if (later - earlier) * 1e9 > bound + LOG_SLACK_NS:
            gaps.append((later - earlier, earlier))
    errors: list[str] = []
    if gaps:
        errors.append(
            f"bidir showed no new content for longer than {BIDIR_DELAY_FIELD} + T "
            f"{len(gaps)} times; worst "
            + ", ".join(
                f"{gap * 1000:.0f} ms at {at:.3f}s" for gap, at in sorted(gaps, reverse=True)[:3]
            )
        )
    return errors, f"{len(gaps)} gaps"


def bidir_fill_errors(
    timeline: BidirTimeline, judged: list[tuple[float, dict[str, int], str]]
) -> list[str]:
    """Distinct frames per refresh must reach MIN_FILL in every judged window."""
    errors: list[str] = []
    for window_end, _, _ in judged:
        count, fill = window_fill(timeline, window_end)
        if fill < MIN_FILL:
            errors.append(
                f"bidir fill {fill:.3f} < {MIN_FILL:g}: the window ending at {window_end:.3f}s "
                f"presented {count} distinct frames in "
                f"{METRIC_WINDOW_SECONDS * REFRESH_HZ:.0f} refreshes"
            )
    return errors


def bidir_window_rows(
    records: list[tuple[float, str]],
    metrics: list[tuple[float, dict[str, int], str]],
    timeline: BidirTimeline,
    enable: float,
    settled: list[tuple[float, dict[str, int], str]],
) -> str:
    """One row per metrics window after the enable: end, client fps, fill, gen, slow, D, peak E.

    A window outside the settled set is marked "unsettled" and a settled one
    off the client's cap "off cap"; neither is judged for fill or slow.
    """
    envelopes = [
        (elapsed, fields["envelope_ms"])
        for elapsed, text in records
        if EPOCH_MARKER in text and "envelope_ms" in (fields := metric_fields(text))
    ]
    settled_ends = {item[0] for item in settled}
    rows: list[str] = []
    for window_end, counts, line in metrics:
        if window_end < enable:
            continue
        start = window_end - METRIC_WINDOW_SECONDS
        fields = metric_fields(line)
        peak = max((value for at, value in envelopes if start < at <= window_end), default=0.0)
        mark = ""
        if window_end not in settled_ends:
            mark = " unsettled"
        elif not holds_cap(timeline.reals, window_end):
            mark = " off cap"
        rows.append(
            f"{window_end:.1f}s {window_fps(timeline.reals, window_end):.1f}fps "
            f"fill {window_fill(timeline, window_end)[1]:.3f} gen {counts['gen']} "
            f"slow {fields.get('slow', 0):.0f} D {fields.get(BIDIR_DELAY_FIELD, 0):.1f} "
            f"E {peak:.1f}{mark}"
        )
    return "windows [" + " | ".join(rows) + "]"


def bidir_errors(
    records: list[tuple[float, str]],
    metrics: list[tuple[float, dict[str, int], str]],
    sigterm_elapsed: float,
) -> tuple[list[str], str]:
    """Judge a bidir run against its logged target delay D, across one focus change.

    Fill and slow are judged per settled window in which the client held its
    frame cap; each side of the focus change needs one such window, or the run
    is invalid. Every other gate judges the whole run at any cadence: real
    frames from BIDIR_SETTLE_SECONDS after the bidir enable to the end of the
    last metrics window before SIGTERM.
    """
    requested = marker_record(records, BIDIR_MARKER)
    if requested is None:
        return [
            "inner did not run bidir (Wayland backend fell back to SDL or BIDIR was ignored)"
        ], "no bidir marker"
    focus = [
        elapsed for elapsed, text in records if elapsed >= requested[0] and FOCUS_MARKER in text
    ]
    if not focus:
        return ["second vkcube did not trigger a focus change"], "no focus change"
    if len(focus) > 1:
        return [
            "expected exactly one focus change after the bidir marker, got "
            + ", ".join(f"{elapsed:.3f}s" for elapsed in focus)
        ], "several focus changes"

    enable = requested[0]
    since = enable + BIDIR_SETTLE_SECONDS
    timeline = bidir_timeline(records)
    errors: list[str] = []
    settled = settled_windows(metrics, timeline, since, sigterm_elapsed)
    judged = [item for item in settled if holds_cap(timeline.reals, item[0])]
    for side, before in (("before", True), ("after", False)):
        if not any((item[0] < focus[0]) == before for item in settled):
            errors.append(
                f"no settled window {side} the focus change at {focus[0]:.3f}s (too many "
                "timeline transitions or too short a hold)"
            )
        elif not any((item[0] < focus[0]) == before for item in judged):
            errors.append(
                f"harness invalid: no settled window {side} the focus change at "
                f"{focus[0]:.3f}s held --framerate-limit {FRAMERATE_LIMIT} "
                f"+-{FRAMERATE_CAP_TOLERANCE:.0%}"
            )
    errors.extend(throughput_errors(judged, since, sigterm_elapsed, min_fill=None))
    errors.extend(bidir_fill_errors(timeline, judged))
    windows, window_errors = delay_windows(metrics)
    errors.extend(window_errors)
    last = max((elapsed for elapsed, _, _ in metrics if elapsed <= sigterm_elapsed), default=since)
    total = marker_record(records, FRAMEGEN_SHUTDOWN_MARKER)
    total_reprimes = (
        int(match.group("n")) if total and (match := REPRIME_RE.search(total[1])) else None
    )

    gates = [
        bidir_reprime_errors(timeline, enable, total_reprimes),
        bidir_unseen_errors(timeline, since, last),
        bidir_order_errors(timeline),
        bidir_spacing_errors(timeline, since, last),
    ]
    if windows:
        gates.append(bidir_delay_errors(timeline, windows, since, last))
        gates.append(bidir_gap_errors(timeline, windows, since, last))
    gates.append(bidir_sequence_errors(records, focus[0], sigterm_elapsed, windows))
    evidence = [
        f"judged {len(judged)}/{len(settled)} settled windows",
        bidir_window_rows(records, metrics, timeline, enable, settled),
    ]
    for gate_errors, gate_evidence in gates:
        errors.extend(gate_errors)
        evidence.append(gate_evidence)
    return errors, "; ".join(evidence)


def episode_cost_errors(
    records: list[tuple[float, str]], requested: float, sigterm_elapsed: float
) -> list[str]:
    """One present-error episode costs one recreation and one history reset.

    The count starts at the recreation request, not at the injection marker,
    which precedes startup recreations, and runs to SIGTERM so a redundant
    reset landing after the waiter is re-enabled is still charged.
    """
    window = [text for elapsed, text in records if requested <= elapsed <= sigterm_elapsed]
    remakes = sum(REMAKE_MARKER in text for text in window)
    reasons = [match.group("reason") for text in window if (match := RESET_RE.search(text))]
    if remakes == 1 and len(reasons) == 1:
        return []
    return [
        "one present-error episode must cost exactly one swapchain recreation and one "
        f"framegen history reset; after the request: {remakes} recreation(s), "
        f"{len(reasons)} reset(s) ({', '.join(reasons) or 'none'})"
    ]


def signal_if_alive(pid: int | None, signum: int) -> None:
    if pid is None:
        return
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        pass


def terminate_group(group_id: int, grace: float = 2.0) -> None:
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass


def scenario_injection(scenario: str) -> tuple[str | None, str | None]:
    if scenario == "present-timeout":
        return "timeout:40", None
    if scenario == "present-error":
        return "error:-1000001004:1", None
    if scenario == "sdl-error":
        return None, "error:30"
    if scenario == "sdl-error-pumped":
        return None, "error-pumped:30"
    return None, None


def effective_injection(args: argparse.Namespace, scenario: str) -> tuple[str | None, str | None]:
    if args.no_inject:
        return None, None
    if args.present_inject is not None or args.sdl_inject is not None:
        return args.present_inject, args.sdl_inject
    return scenario_injection(scenario)


def validate_scenario(
    scenario: str,
    inner_log: str,
    status: dict[str, object] | None,
    shutdown_elapsed: float | None,
    sigterm_grace: float,
    present_inject: str | None,
    sdl_inject: str | None,
    sigterm_elapsed: float,
    shutdown_marker: str,
) -> Result:
    errors: list[str] = []
    records = log_records(inner_log)
    metrics = metric_records(records)
    joined = marker_record(records, shutdown_marker)
    if BAD_WORD_RE.search(inner_log):
        errors.append('inner log contains "abort", "terminate", or "corrupted"')
    if joined is None:
        errors.append(f'missing "{shutdown_marker}"')
    if status is None:
        errors.append("inner exit status was not recorded")
    elif status.get("returncode") != 0:
        errors.append(f"inner exit code was {status.get('returncode')}, expected 0")
    if shutdown_elapsed is None or shutdown_elapsed > sigterm_grace:
        errors.append(f"inner did not exit within {sigterm_grace:g} seconds of SIGTERM")

    evidence = joined[1].strip() if joined else "no shutdown evidence"
    if scenario == "baseline":
        enabled = marker_record(records, FRAMEGEN_ENABLED_MARKER)
        if enabled is None:
            errors.append(f'missing "{FRAMEGEN_ENABLED_MARKER}" marker')
        else:
            errors.extend(
                generation_errors(metrics, "framegen enable", enabled[0], sigterm_elapsed)
            )
            errors.extend(throughput_errors(metrics, enabled[0], sigterm_elapsed))
        if metrics:
            evidence = metrics[-1][2]

    elif scenario == "present-timeout":
        active = marker_record(records, "present_wait: fault injection active")
        stalled = marker_record(records, "present_wait: no completion for")
        recovered = marker_record(records, "present_wait: recovered after stall")
        for description, found in (("fault-injection marker", active), ("stall marker", stalled), ("recovery marker", recovered)):
            if found is None:
                errors.append(f"missing present-wait {description}")
        if stalled and recovered and inner_log.find("present_wait: no completion for") >= inner_log.find("present_wait: recovered after stall"):
            errors.append("present-wait recovery marker precedes the stall marker")
        quiet_candidates: list[tuple[float, dict[str, int], str]] = []
        if active and recovered:
            quiet_candidates = [
                item
                for item in metrics
                if item[0] - METRIC_WINDOW_SECONDS >= active[0]
                and item[0] <= recovered[0]
            ]
            if not quiet_candidates:
                errors.append("no metrics window lies entirely inside the present-wait stall")
            fabricated = [
                item
                for item in quiet_candidates
                if any(item[1][name] != 0 for name in ("real", "gen", "dreal"))
            ]
            if fabricated:
                errors.append(
                    "fully stalled metrics window has non-zero real, gen, or dreal: "
                    + fabricated[0][2]
                )
        if quiet_candidates:
            evidence = quiet_candidates[-1][2]

    elif scenario == "present-error":
        active = marker_record(records, "present_wait: fault injection active")
        requested = next(
            (
                (elapsed, text)
                for elapsed, text in records
                if "present_wait:" in text and "requesting swapchain recreation" in text
            ),
            None,
        )
        enabled = marker_record(records, "present_wait: re-enabled after swapchain recreation")
        if active is None:
            errors.append("missing present-wait fault-injection marker")
        if requested is None:
            errors.append("missing swapchain-recreation request marker")
        if enabled is None:
            errors.append("missing present-wait re-enabled marker")
        if requested and enabled and requested[0] >= enabled[0]:
            errors.append("present waiter was re-enabled before recreation was requested")
        if enabled:
            errors.extend(generation_errors(metrics, "recovery", enabled[0], sigterm_elapsed))
        if requested and present_inject == scenario_injection("present-error")[0]:
            errors.extend(episode_cost_errors(records, requested[0], sigterm_elapsed))
        if metrics:
            evidence = metrics[-1][2]

    elif scenario in ("sdl-error", "sdl-error-pumped"):
        active = marker_record(records, "SDL event loop: fault injection active")
        if active is None:
            errors.append("missing SDL fault-injection marker")
        # Match both SDL_WaitEvent and SDL_WaitEventTimeout spellings, after
        # the "sdl:" log-scope prefix.
        all_failures = [
            (elapsed, text)
            for elapsed, text in records
            if SDL_WAIT_FAILURE_RE.search(text)
            and active is not None
            and elapsed >= active[0]
        ]
        failures_in_first_three_seconds = [
            item for item in all_failures if active is not None and item[0] <= active[0] + 3.0
        ]
        if not failures_in_first_three_seconds:
            errors.append("no injected SDL wait failure was logged in the first 3 seconds")
        if len(failures_in_first_three_seconds) > 5:
            errors.append(
                f"SDL wait failure logged {len(failures_in_first_three_seconds)} times "
                "in 3 seconds (maximum 5)"
            )
        # Identical SDL errors are logged at most once per second, so the last
        # logged failure can precede the last injected one by up to a second. A
        # finite injection count must mark its exhaustion after activation; only
        # an endless spec falls back to the last logged failure.
        exhausted_marker = "SDL event loop: fault injection exhausted"
        exhausted = marker_record(records, exhausted_marker)
        if sdl_inject and ":" in sdl_inject:
            if exhausted is None:
                errors.append("missing SDL fault-injection exhausted marker")
            elif active and inner_log.find(exhausted_marker) < inner_log.find("SDL event loop: fault injection active"):
                errors.append("SDL fault injection exhausted before it was active")
            last_failure = exhausted
        else:
            last_failure = all_failures[-1] if all_failures else None
        # A window overlapping the failure may hold frames from before it, so
        # only windows starting at or after the last failure count.
        metrics_after_failure = [
            item
            for item in metrics
            if last_failure
            and item[0] - METRIC_WINDOW_SECONDS >= last_failure[0]
            and item[1]["real"] > 0
        ]
        if last_failure and not metrics_after_failure:
            errors.append(
                "no framegen-metrics window starting at or after the last injected SDL "
                f"failure ({last_failure[0]:.3f}s) has real > 0"
            )

        # Judge the last map-gate episode: an earlier deferral/resume pair
        # proves nothing about a later deferral that never resumed.
        deferred_marker = "window not mapped; presentation deferred"
        resumed_marker = "window mapped; presentation resumed"
        latest_first = records[::-1]
        deferred = marker_record(latest_first, deferred_marker)
        resumed = marker_record(latest_first, resumed_marker)
        if deferred is None:
            errors.append("missing SDL map-gate deferral marker")
        if resumed is None:
            errors.append("missing SDL map-gate resume marker")
        if deferred and resumed and inner_log.rfind(deferred_marker) > inner_log.rfind(resumed_marker):
            errors.append(
                f"SDL presentation still deferred: last deferral at {deferred[0]:.3f}s "
                f"is not followed by a resume (last resume {resumed[0]:.3f}s)"
            )
        if last_failure and resumed:
            resume_delay = resumed[0] - last_failure[0]
            if resume_delay < 0.0 or resume_delay > 1.0:
                errors.append(
                    "SDL presentation did not resume within 1 second after the "
                    f"last injected failure (delay {resume_delay:.3f}s)"
                )
        # Presenting real frames again is not enough: generation must recover
        # too, and must not stop again before SIGTERM.
        if resumed:
            errors.extend(generation_errors(metrics, "recovery", resumed[0], sigterm_elapsed))
        if last_failure:
            evidence = (
                f"last failure at {last_failure[0]:.3f}s; "
                + (metrics_after_failure[0][2] if metrics_after_failure else "no later metrics window")
            )

    elif scenario == "bidir":
        scenario_errors, evidence = bidir_errors(records, metrics, sigterm_elapsed)
        errors.extend(scenario_errors)

    return Result(scenario, not errors, evidence, errors)


HOST_BUS_CONFIG = """<!DOCTYPE busconfig PUBLIC
 "-//freedesktop//DTD D-Bus Bus Configuration 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
  <type>session</type>
  <listen>unix:dir={runtime_dir}</listen>
  <auth>EXTERNAL</auth>
  <policy context="default">
    <allow send_destination="*"/>
    <allow receive_sender="*"/>
    <allow own="*"/>
  </policy>
</busconfig>
"""


HOST_MODE = f"640x480@{REFRESH_HZ}"


def host_command(bus_config: Path, mode: str = HOST_MODE) -> list[str]:
    return [
        "dbus-run-session", f"--config-file={bus_config}", "--",
        "gnome-shell", "--headless", "--wayland", "--no-x11",
        "--virtual-monitor", mode, "--wayland-display", HOST_DISPLAY,
    ]


def start_host(
    bus_config: Path, state_dir: Path, runtime_dir: Path, env: dict[str, str], log: BinaryIO,
    mode: str = HOST_MODE,
) -> subprocess.Popen[bytes]:
    """Start the bidir Wayland host and wait for its display socket.

    gamescope's headless outer lacks wl_subcompositor and wp_viewporter, which
    the Wayland backend requires. The private bus has no service directory,
    so nothing is ever bus-activated.
    """
    if shutil.which("gnome-shell") is None or shutil.which("dbus-run-session") is None:
        raise RuntimeError(
            "bidir needs gnome-shell and dbus-run-session: gamescope's headless outer lacks "
            "wl_subcompositor/wp_viewporter, so --backend wayland would fall back to SDL"
        )
    write_text(bus_config, HOST_BUS_CONFIG.format(runtime_dir=runtime_dir))
    host_env = {
        name: value
        for name, value in env.items()
        if name not in ("DBUS_SESSION_BUS_ADDRESS", "DISPLAY", "WAYLAND_DISPLAY")
    }
    for kind in ("config", "data", "cache", "state"):
        host_env[f"XDG_{kind.upper()}_HOME"] = str(state_dir / "host" / kind)
    host_env.update(
        GSETTINGS_BACKEND="memory", GIO_USE_VFS="local", GVFS_DISABLE_FUSE="1", NO_AT_BRIDGE="1"
    )
    host = subprocess.Popen(
        host_command(bus_config, mode),
        env=host_env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if not poll_until((runtime_dir / HOST_DISPLAY).exists, 20.0) or host.poll() is not None:
        terminate_group(host.pid)
        raise RuntimeError(f"bidir host did not create the {HOST_DISPLAY} Wayland display")
    return host


def hold_seconds(args: argparse.Namespace, scenario: str) -> float:
    if args.hold_seconds is not None:
        return args.hold_seconds
    return BIDIR_HOLD_SECONDS if scenario == "bidir" else DEFAULT_HOLD_SECONDS


def run_scenario(args: argparse.Namespace, scenario: str, root_logdir: Path) -> Result:
    run_dir = root_logdir / scenario
    suffix = 2
    while run_dir.exists():
        run_dir = root_logdir / f"{scenario}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    state_dir = run_dir / "state"
    state_dir.mkdir(mode=0o700, exist_ok=False)
    # Fixed under /tmp: the Wayland and X socket paths must fit sun_path (108 bytes).
    runtime_dir = Path(tempfile.mkdtemp(prefix="fgsmoke-", dir="/tmp"))
    write_text(run_dir / "runtime-dir.txt", f"{runtime_dir}\n")
    outer_log_path = run_dir / "outer.log"
    inner_log_path = run_dir / "inner.log"
    client_log_path = run_dir / "client.log"
    bus_config = state_dir / "host-bus.conf"
    present_inject, sdl_inject = effective_injection(args, scenario)
    bidir = scenario == "bidir"
    shutdown_marker = FRAMEGEN_SHUTDOWN_MARKER if bidir else WAITER_JOINED_MARKER

    inner_wrapper = [
        *SELF,
        "--internal-inner",
        "--state-dir", str(state_dir),
        "--inner-log", str(inner_log_path),
        "--client-log", str(client_log_path),
        "--binary", args.binary,
        "--vkcube", args.vkcube,
        "--ptracer", str(os.getpid()),
    ]
    if present_inject:
        inner_wrapper.extend(("--present-inject", present_inject))
    if sdl_inject:
        inner_wrapper.extend(("--sdl-inject", sdl_inject))
    if bidir:
        inner_wrapper.append("--bidir")
        outer_command = [
            args.env_wrapper, "env", "-u", "DISPLAY", f"WAYLAND_DISPLAY={HOST_DISPLAY}",
            *inner_wrapper,
        ]
    else:
        outer_command = [
            args.env_wrapper,
            "env", "-u", "DISPLAY", "-u", "WAYLAND_DISPLAY",
            args.binary,
            "--backend", "headless", "--expose-wayland", "-W", "640", "-H", "480",
            "-r", str(REFRESH_HZ),
            "--", *inner_wrapper,
        ]

    # Pin the wrapper to this checkout and build so the binary, PATH, and the
    # WSI layer on VK_LAYER_PATH all come from --build-dir.
    outer_env = {
        "GAMESCOPE_LOCAL_PREFIX": args.prefix,
        "GAMESCOPE_REPO": args.repo,
        "GAMESCOPE_BUILD_DIR": os.path.relpath(args.build_dir, args.repo),
        "GAMESCOPE_ZENITY_DISABLE": "1",
        "XDG_RUNTIME_DIR": str(runtime_dir),
    }
    print(
        f"\n[{scenario}] injection: "
        f"present={present_inject or '<none>'} sdl={sdl_inject or '<none>'}"
    )
    print(f"[{scenario}] commands:")
    if bidir:
        print(shell_join(host_command(bus_config)))
    print(shell_join([*env_assignments(outer_env), *outer_command]))
    print(shell_join([
        *env_assignments(inner_env(present_inject, sdl_inject, bidir)),
        *inner_command(args.binary, state_dir, client_log_path, args.vkcube, bidir),
    ]))
    print(shell_join(["env", "-u", DISABLE_WSI_ENV, args.vkcube]))

    env = os.environ.copy()
    for name in list(env):
        if name.startswith(FRAMEGEN_ENV_PREFIX) or name in FAULT_ENV:
            env.pop(name, None)
    env.update(outer_env)

    host: subprocess.Popen[bytes] | None = None
    outer: subprocess.Popen[bytes] | None = None
    inner_pid: int | None = None
    client_pid: int | None = None
    shutdown_elapsed: float | None = None
    last_pre_sigterm: str | None = None
    sigterm_elapsed = 0.0
    inner_sigterm_sent = False
    setup_error: str | None = None
    status: dict[str, object] | None = None
    captures: list[str] = []
    allow_harness = functools.partial(allow_stack_capture, os.getpid())
    with outer_log_path.open("wb") as outer_log:
        try:
            if bidir:
                host = start_host(bus_config, state_dir, runtime_dir, env, outer_log)
            outer = subprocess.Popen(
                outer_command,
                env=env,
                stdout=outer_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=allow_harness,
            )
            deadline = time.monotonic() + 20.0
            inner_pid = wait_for_pid_file(state_dir / "inner.pid", deadline)
            client_pid = wait_for_pid_file(state_dir / "client.pid", deadline)
            readiness_started = time.monotonic()
            time.sleep(2.0)
            if (
                outer.poll() is not None
                or (host is not None and host.poll() is not None)
                or not pid_alive(inner_pid)
                or not pid_alive(client_pid)
            ):
                raise RuntimeError(
                    "host, outer, inner, or vkcube exited during the readiness interval"
                )
            print(f"[{scenario}] ready: outer={outer.pid} inner={inner_pid} client={client_pid}")
            time.sleep(
                max(0.0, hold_seconds(args, scenario) - (time.monotonic() - readiness_started))
            )
            shutdown_started = time.monotonic()
            pre_sigterm_log = read_text(inner_log_path)
            last_pre_sigterm = last_log_line(pre_sigterm_log)
            sigterm_elapsed = log_cutoff(pre_sigterm_log)
            signal_if_alive(inner_pid, signal.SIGTERM)
            inner_sigterm_sent = True
            status_path = state_dir / "inner-status.json"
            exited, capture = await_or_capture(
                status_path.exists, inner_pid, run_dir / f"stacks-inner-{inner_pid}.txt",
                args.sigterm_grace,
            )
            captures.extend([capture] if capture else [])
            if exited:
                shutdown_elapsed = time.monotonic() - shutdown_started
                status = json.loads(read_text(status_path))
            else:
                setup_error = (
                    "inner did not record an exit status within "
                    f"{args.sigterm_grace:g} seconds of SIGTERM"
                )
            exited, capture = await_or_capture(
                lambda: outer.poll() is not None, outer.pid,
                run_dir / f"stacks-outer-{outer.pid}.txt", args.sigterm_grace,
            )
            captures.extend([capture] if capture else [])
            if not exited and setup_error is None:
                setup_error = "outer compositor did not exit after the inner compositor stopped"
        except TimeoutError as exc:  # startup: save the stacks of whatever is stuck
            setup_error = str(exc)
            if inner_pid is not None and pid_alive(inner_pid):
                name, stuck = "inner", inner_pid
            else:
                name, stuck = "outer", outer.pid if outer and outer.poll() is None else None
            if stuck is not None:
                captures.append(capture_stacks(stuck, run_dir / f"stacks-{name}-{stuck}.txt"))
        except Exception as exc:  # cleanup and report supervisor/setup failures uniformly
            setup_error = str(exc)
        finally:
            signal_if_alive(client_pid, signal.SIGTERM)
            if not inner_sigterm_sent:
                pre_sigterm_log = read_text(inner_log_path)
                last_pre_sigterm = last_log_line(pre_sigterm_log)
                sigterm_elapsed = log_cutoff(pre_sigterm_log)
            signal_if_alive(inner_pid, signal.SIGTERM)
            if outer is not None:
                terminate_group(outer.pid, grace=args.sigterm_grace)
                try:
                    outer.wait(timeout=args.sigterm_grace)
                except subprocess.TimeoutExpired:
                    pass
            # After the inner is gone, so it never sees its display vanish. As a
            # group: SIGTERM to dbus-run-session alone orphans gnome-shell.
            if host is not None:
                terminate_group(host.pid, grace=args.sigterm_grace)
                try:
                    host.wait(timeout=args.sigterm_grace)
                except subprocess.TimeoutExpired:
                    pass

    shutil.rmtree(runtime_dir, ignore_errors=True)

    inner_text = read_text(inner_log_path)
    result = validate_scenario(
        scenario, inner_text, status, shutdown_elapsed, args.sigterm_grace, present_inject,
        sdl_inject, sigterm_elapsed, shutdown_marker,
    )
    if setup_error:
        result.errors.insert(0, "; ".join([setup_error, *captures]))
        result.passed = False
    if not read_text(outer_log_path).strip():
        result.errors.append("outer compositor log is empty")
        result.passed = False
    if not read_text(client_log_path).strip() and client_pid is None:
        result.errors.append("vkcube did not start or produce a PID record")
        result.passed = False
    joined = marker_record(log_records(inner_text), shutdown_marker)
    shutdown_failed = (
        status is None
        or status.get("returncode") != 0
        or shutdown_elapsed is None
        or shutdown_elapsed > args.sigterm_grace
        or joined is None
        or (setup_error is not None and ("exit" in setup_error or "stopped" in setup_error))
    )
    if shutdown_failed or captures:
        result.shutdown_evidence = "\n".join(
            [format_shutdown_evidence(inner_text, last_pre_sigterm), *captures]
        )
    print(f"[{scenario}] logs: {run_dir}")
    return result


def check_log(path: Path) -> int:
    """Re-judge a saved bidir inner.log; the TOTAL line stands in for SIGTERM.

    Exit status and shutdown timing are not re-checked.
    """
    records = log_records(read_text(path))
    total = marker_record(records, FRAMEGEN_SHUTDOWN_MARKER)
    cutoff = total[0] if total else (records[-1][0] if records else 0.0)
    errors, evidence = bidir_errors(records, metric_records(records), cutoff)
    print(f"{'PASS' if not errors else 'FAIL'}  {evidence}")
    for error in errors:
        print(f"  - {error}")
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", choices=(*SCENARIOS, "all"))
    parser.add_argument(
        "--logdir", type=Path,
        help="log root (default: a new directory under $TMPDIR or /tmp)",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=Path("build"),
        help="meson build directory, relative to the repository root (default: build)",
    )
    parser.add_argument(
        "--hold-seconds", type=float,
        help="seconds to hold each scenario before SIGTERM "
        f"(default: {BIDIR_HOLD_SECONDS:g} for bidir, {DEFAULT_HOLD_SECONDS:g} otherwise)",
    )
    parser.add_argument(
        "--sigterm-grace", type=float, default=5.0,
        help="seconds to allow clean shutdown after SIGTERM (default: 5)",
    )
    parser.add_argument(
        "--prefix", default=os.environ.get("GAMESCOPE_LOCAL_PREFIX"),
        help="local dependency prefix (default: $GAMESCOPE_LOCAL_PREFIX)",
    )
    parser.add_argument("--internal-inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--state-dir", help=argparse.SUPPRESS)
    parser.add_argument("--inner-log", help=argparse.SUPPRESS)
    parser.add_argument("--client-log", help=argparse.SUPPRESS)
    parser.add_argument("--binary", help=argparse.SUPPRESS)
    parser.add_argument("--vkcube", help=argparse.SUPPRESS)
    parser.add_argument("--bidir", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--second-client-after", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--ptracer", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--present-inject", metavar="SPEC",
        help="replace the scenario default with a present-wait injection spec",
    )
    parser.add_argument(
        "--sdl-inject", metavar="SPEC",
        help="replace the scenario default with an SDL-wait injection spec",
    )
    parser.add_argument(
        "--no-inject", action="store_true",
        help="disable the scenario's default fault injection",
    )
    parser.add_argument(
        "--check-log", type=Path, metavar="INNER_LOG",
        help="re-judge a saved bidir inner.log with the bidir gates and exit",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.internal_client:
        return internal_client(args)
    if args.internal_inner:
        return internal_inner(args)
    if args.scenario is None:
        parser.error("a subcommand is required")
    if args.check_log is not None:
        if args.scenario != "bidir":
            parser.error("--check-log supports the bidir scenario only")
        if not args.check_log.is_file():
            parser.error(f"--check-log: no such file: {args.check_log}")
        return check_log(args.check_log)
    if args.no_inject and (args.present_inject is not None or args.sdl_inject is not None):
        parser.error("--no-inject cannot be combined with --present-inject or --sdl-inject")
    if args.hold_seconds is not None and args.hold_seconds < 0:
        parser.error("--hold-seconds must be non-negative")
    if args.sigterm_grace <= 0:
        parser.error("--sigterm-grace must be greater than zero")

    repo = Path(__file__).resolve().parent.parent
    build_dir = (repo / args.build_dir).resolve()
    binary = build_dir / "src" / "gamescope"
    env_wrapper = repo / "scripts" / "env-gamescope-local.sh"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"gamescope binary is missing or not executable: {binary}")
    if not env_wrapper.is_file() or not os.access(env_wrapper, os.X_OK):
        parser.error(f"dependency wrapper is missing or not executable: {env_wrapper}")
    if not args.prefix:
        parser.error("no local dependency prefix: pass --prefix or set GAMESCOPE_LOCAL_PREFIX")
    prefix = Path(args.prefix).resolve()
    if not prefix.is_dir():
        parser.error(f"local dependency prefix is not a directory: {prefix}")
    lookup_path = os.pathsep.join((str(prefix / "bin"), os.environ.get("PATH", "")))
    vkcube = shutil.which("vkcube", path=lookup_path)
    if vkcube is None:
        parser.error("vkcube was not found in GAMESCOPE_LOCAL_PREFIX/bin or PATH")
    args.prefix = str(prefix)
    args.repo = str(repo)
    args.build_dir = build_dir
    args.binary = str(binary)
    args.env_wrapper = str(env_wrapper)
    args.vkcube = vkcube

    if args.logdir is None:
        root_logdir = Path(tempfile.mkdtemp(prefix="gamescope-framegen-smoke-"))
    else:
        root_logdir = args.logdir.resolve()
        root_logdir.mkdir(parents=True, exist_ok=True)
    print(f"logs: {root_logdir}")

    selected = SCENARIOS if args.scenario == "all" else (args.scenario,)
    results = [run_scenario(args, scenario, root_logdir) for scenario in selected]
    print("\nRESULT             STATUS  EVIDENCE")
    print("-----------------  ------  --------")
    for result in results:
        print(f"{result.scenario:17}  {'PASS' if result.passed else 'FAIL':6}  {result.evidence}")
        for error in result.errors:
            print(f"{'':27}  - {error}")
        if result.shutdown_evidence:
            print(f"{'':27}  shutdown evidence:")
            for line in result.shutdown_evidence.splitlines():
                print(f"{'':29}{line}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
