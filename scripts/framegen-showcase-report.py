#!/usr/bin/env python3
"""Summarise and compare framegen-showcase.sh sessions.

summarize: read the run logs of one session (each starts with the
"framegen-showcase: key=value" header the driver writes), keep the steady
metrics windows (w > warmup; the TOTAL line is never used), print the table and
write results.json.

compare: judge a candidate results.json against a baseline one. Exit 0 when no
gated factor regresses, 1 on a regression, 2 when the sessions are not
comparable (different context, a DEBUG session, a client not seen on its render
device, or a run whose flags or latency start point differ).

Standard library only, so the judge runs anywhere the driver runs.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import NamedTuple

HEADER = "framegen-showcase: "
WINDOW_RE = re.compile(r"framegen-metrics: w=(\d+) (.*)$")
FIELD_RE = re.compile(r"\s(?P<name>\w+)=(?P<value>-?\d+(?:\.\d+)?)(?=\s|$)")
VERSION_RE = re.compile(r"gamescope version (\S+)")

# Length of one compositor metrics window.
WINDOW_SECONDS = 5.0

# Must match between baseline and candidate.
CONTEXT = ("backend", "size", "hz", "render", "present", "client", "seconds", "warmup", "debug")
# Must match when both sides recorded it (older logs lack it).
OPTIONAL_CONTEXT = ("host_display",)
# What is under test: printed for both sides, never compared.
TREATMENT = ("bin", "version", "extra_env")
# Identify a run; must match between the reps of a session and across sessions.
RUN_KEYS = ("flags", "run_env")
REQUIRED = ("run", "index", "rep", *CONTEXT, "bin", "extra_env", *RUN_KEYS)

QUALITY = ("q_resid", "q_bad_mv", "q_killed_mv", "q_killed_match", "q_killed_fb")
SUMMED = ("reprime", "ring_ovf", "q_samples")
REPORTED = ("real_s", "shown_s", "dl_hit", "bias_ms", "reprime", "ring_ovf", "q_samples")


class Gate(NamedTuple):
    """A negative limit is the largest allowed fall, a positive one the largest
    allowed rise. A relative limit is a fraction of the baseline value."""

    field: str
    limit: float
    relative: bool


GATES = (
    Gate("fill", -0.01, False),
    Gate("flip_ms_sd", 0.5, False),
    Gate("lat_p50", 1.0, False),
    *(Gate(q, 0.05, True) for q in QUALITY),
)


def refuse(message):
    print(f"framegen-showcase: {message}", file=sys.stderr)
    sys.exit(2)


def read_log(path):
    header, windows, version = {}, [], None
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(HEADER):
                key, _, value = line[len(HEADER):].partition("=")
                header[key] = value
                continue
            m = WINDOW_RE.search(line)
            if m:
                fields = {f.group("name"): float(f.group("value")) for f in FIELD_RE.finditer(" " + m.group(2))}
                windows.append((int(m.group(1)), fields))
            elif version is None:
                v = VERSION_RE.search(line)
                if v:
                    version = v.group(1)
    missing = [k for k in REQUIRED if k not in header]
    if missing:
        refuse(f"{path}: not a framegen-showcase log (header lacks {', '.join(missing)})")
    warmup = int(header["warmup"])
    return {"path": str(path), "header": header, "version": version,
            "windows": [fields for w, fields in windows if w > warmup]}


def lat_key(reps_windows):
    """Bidir shows its reals as delayed reals, measured from the record; live
    reals are measured from source-ready. One start point per run."""
    real = sum(w.get("real", 0.0) for ws in reps_windows for w in ws)
    dreal = sum(w.get("dreal", 0.0) for ws in reps_windows for w in ws)
    return "dreal_flip_ms_p50" if dreal > real else "src_flip_ms_p50"


def _median(values):
    values = list(values)
    return statistics.median(values) if values else None


def summarize_rep(windows, key):
    n = len(windows)
    if not n:
        return {"windows": 0}

    def total(*names):
        return sum(w.get(k, 0.0) for w in windows for k in names)

    fresh = total("real", "dreal", "gen")
    out = {
        "windows": n,
        "real_s": total("real", "dreal") / (WINDOW_SECONDS * n),
        "shown_s": fresh / (WINDOW_SECONDS * n),
    }
    if fresh + total("rep"):
        out["fill"] = fresh / (fresh + total("rep"))

    # Pooled population sd; the presented count stands in for the interval count.
    pooled = [(w.get("real", 0.0) + w.get("gen", 0.0) + w.get("dreal", 0.0) + w.get("rep", 0.0),
               w["flip_ms_avg"], w["flip_ms_sd"]) for w in windows if "flip_ms_avg" in w and "flip_ms_sd" in w]
    count = sum(n_i for n_i, _, _ in pooled)
    if count:
        mean = sum(n_i * avg for n_i, avg, _ in pooled) / count
        second = sum(n_i * (sd * sd + avg * avg) for n_i, avg, sd in pooled) / count
        out["flip_ms_sd"] = max(second - mean * mean, 0.0) ** 0.5

    optional = {
        "lat_p50": _median(w[key] for w in windows if w.get(key, 0.0) > 0),
        # Without generated or delayed frames there are no deadlines and no bias.
        **{f: _median(w[f] for w in windows if f in w and w.get("gen", 0.0) + w.get("dreal", 0.0) > 0)
           for f in ("dl_hit", "bias_ms")},
        **{f: sum(w[f] for w in windows if f in w) for f in SUMMED if any(f in w for w in windows)},
        **{f: _median(w[f] for w in windows if f in w and w.get("q_samples", 0.0) > 0) for f in QUALITY},
    }
    out.update((k, v) for k, v in optional.items() if v is not None)
    return out


def summarize(paths):
    if not paths:
        refuse("no run logs to summarise")
    logs = [read_log(p) for p in paths]

    for key in (*CONTEXT, *OPTIONAL_CONTEXT, "bin", "extra_env"):
        values = sorted({log["header"].get(key, "") for log in logs})
        if len(values) > 1:
            refuse(f"logs disagree on {key}: {values}")
    versions = sorted({log["version"] for log in logs if log["version"]})
    if len(versions) > 1:
        refuse(f"logs disagree on version: {versions}")

    groups = {}
    for log in logs:
        groups.setdefault((int(log["header"]["index"]), log["header"]["run"]), []).append(log)

    runs = []
    for (index, name), group in sorted(groups.items()):
        for key in RUN_KEYS:
            values = sorted({log["header"][key] for log in group})
            if len(values) > 1:
                refuse(f"run {index} reps disagree on {key}: {values}")
        group.sort(key=lambda log: int(log["header"]["rep"]))
        steady = [log["windows"] for log in group]
        key = lat_key(steady)
        reps = []
        for log in group:
            rep = summarize_rep(log["windows"], key)
            exit_status = log["header"].get("exit", "")
            reps.append({"rep": int(log["header"]["rep"]), "log": log["path"],
                         "render_gpu": log["header"].get("render_gpu", "unverified"), "exit": exit_status,
                         "complete": exit_status == "124" and rep["windows"] > 0, **rep})
        measured = [r for r in reps if r["windows"] > 0]
        fields = {k for r in measured for k in r} - {"rep", "log", "render_gpu", "exit", "complete", "windows"}
        runs.append({
            "index": index, "name": name,
            **{k: group[0]["header"][k] for k in RUN_KEYS},
            # No steady window anywhere: no start point to compare.
            "lat_of": ("delayed" if key.startswith("dreal") else "live") if any(steady) else None,
            "reps": reps,
            "median": {f: statistics.median(r[f] for r in measured if f in r) for f in sorted(fields)},
            "complete_reps": sum(r["complete"] for r in reps),
        })

    header = logs[0]["header"]
    return {"context": {k: header[k] for k in (*CONTEXT, *OPTIONAL_CONTEXT) if k in header},
            "treatment": {"bin": header["bin"], "version": versions[0] if versions else "", "extra_env": header["extra_env"]},
            "runs": runs}


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def rep_problem(rep):
    if rep["exit"] != "124":
        return f"ended early (rc={rep['exit'] or 'unknown'})"
    return "no steady metrics window"


def format_table(results):
    columns = (("real/s", "real_s", ".1f"), ("shown/s", "shown_s", ".1f"), ("fill", "fill", ".3f"),
               ("jit_ms", "flip_ms_sd", ".2f"), ("lat_p50", "lat_p50", ".1f"), ("dl_hit", "dl_hit", ".3f"),
               ("bias_ms", "bias_ms", ".3f"), ("reprime", "reprime", "g"), ("ring_ovf", "ring_ovf", "g"),
               ("q_resid", "q_resid", ".4f"), ("q_bad_mv", "q_bad_mv", ".4f"), ("q_kill_mv", "q_killed_mv", ".4f"))
    lines = ["%-3s %-19s %-5s" % ("run", "name", "reps") + "".join(f" {title:>9}" for title, _, _ in columns)]
    warnings = []
    for run in results["runs"]:
        med = run["median"]
        cells = []
        for _, field, spec in columns:
            cell = _fmt(med.get(field), spec)
            if field == "lat_p50" and field in med and run["lat_of"] == "delayed":
                cell += "d"
            cells.append(f" {cell:>9}")
        reps = f"{run['complete_reps']}/{len(run['reps'])}"
        lines.append(f"{run['index']:<3} {run['name'][:19]:<19} {reps:<5}" + "".join(cells))
        for rep in run["reps"]:
            where = f"run {run['index']} rep {rep['rep']}"
            if not rep["complete"]:
                warnings.append(f"  ! {where}: {rep_problem(rep)} ({rep['log']})")
            if rep["render_gpu"] == "not_seen":
                warnings.append(f"  ! {where}: client NOT seen on the render device")
            elif rep["render_gpu"] == "unverified":
                warnings.append(f"  ! {where}: render device not verified")
    lines += warnings
    lines.append(f"Medians over reps of steady windows (w > warmup={results['context']['warmup']}); counts summed, "
                 "fill over the summed frames, jitter = pooled flip_ms_sd, lat_p50 in ms "
                 "(d = delayed reals, from the record).")
    return "\n".join(lines)


def judge(gate, base, cand):
    if gate.relative:
        # A zero baseline flags any increase.
        return round(cand - base * (1.0 + gate.limit), 6) > 0
    d = round(cand - base, 6)
    return d < gate.limit if gate.limit < 0 else d > gate.limit


def compare(base, cand):
    problems = []
    for key in (*CONTEXT, *OPTIONAL_CONTEXT):
        b, c = base["context"].get(key), cand["context"].get(key)
        if key in OPTIONAL_CONTEXT and (b is None or c is None):
            continue
        if b != c:
            problems.append(f"context {key} differs: baseline {b!r}, candidate {c!r}")
    for side, res in (("baseline", base), ("candidate", cand)):
        if res["context"].get("debug") == "1":
            problems.append(f"{side} is a DEBUG run")
        for run in res["runs"]:
            for rep in run["reps"]:
                if rep["render_gpu"] == "not_seen":
                    problems.append(f"{side} run {run['index']} rep {rep['rep']}: client not seen on the render device")
    for side, res in (("baseline", base), ("candidate", cand)):
        print(f"{side} treatment: " + " ".join(f"{k}={res['treatment'].get(k, '')!r}" for k in TREATMENT))

    base_runs = {r["index"]: r for r in base["runs"]}
    cand_runs = {r["index"]: r for r in cand["runs"]}
    common = sorted(base_runs.keys() & cand_runs.keys())
    if not common:
        problems.append("no run in common")
    for index in common:
        b, c = base_runs[index], cand_runs[index]
        for key in RUN_KEYS:
            if b[key] != c[key]:
                problems.append(f"run {index} {key} differs: baseline {b[key]!r}, candidate {c[key]!r}")
        if b["lat_of"] and c["lat_of"] and b["lat_of"] != c["lat_of"]:
            problems.append(f"run {index} lat_of differs: baseline {b['lat_of']}, candidate {c['lat_of']}")
    if problems:
        for p in problems:
            print(f"NOT COMPARABLE: {p}")
        return 2

    gates = {g.field: g for g in GATES}
    regressions = 0
    missing = []
    for index in common:
        b, c = base_runs[index], cand_runs[index]
        print(f"run {index} {c['name']} (latency of {c['lat_of'] or b['lat_of'] or '-'} reals)")
        if all(r["complete"] for r in b["reps"]):
            for rep in c["reps"]:
                if not rep["complete"]:
                    print(f"  rep {rep['rep']}: REGRESS: {rep_problem(rep)}")
                    regressions += 1
        for field in (*REPORTED[:2], *(g.field for g in GATES), *REPORTED[2:]):
            bv, cv = b["median"].get(field), c["median"].get(field)
            if bv is not None and cv is None and field in gates:
                # The baseline measured it: an absent candidate value is no evidence.
                status = "MISSING"
                missing.append(f"run {index} {field}")
            elif bv is None or cv is None:
                status = "n/a"
            elif field not in gates:
                status = "reported"
            elif judge(gates[field], bv, cv):
                status = "REGRESS"
                regressions += 1
            else:
                status = "ok"
            print(f"  {field:<15} {_fmt(bv, '.6g'):>10} -> {_fmt(cv, '.6g'):<10} ({status})")
    for index in sorted(base_runs.keys() ^ cand_runs.keys()):
        side = "baseline" if index in base_runs else "candidate"
        print(f"run {index}: not compared ({side} only)")

    if regressions:
        print(f"FAIL: {regressions} regression(s)")
        return 1
    if missing:
        print(f"NOT COMPARABLE: the candidate lacks {', '.join(missing)}")
        return 2
    print("PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summarize", help="summarise one session's run logs")
    s.add_argument("--out", required=True, type=Path, help="results.json to write")
    s.add_argument("logs", nargs="*", type=Path)
    c = sub.add_parser("compare", help="judge a candidate results.json against a baseline")
    c.add_argument("baseline", type=Path)
    c.add_argument("candidate", type=Path)
    args = parser.parse_args()

    if args.command == "summarize":
        results = summarize(args.logs)
        args.out.write_text(json.dumps(results, indent=1) + "\n")
        print(format_table(results))
        return 0
    loaded = []
    for path in (args.baseline, args.candidate):
        try:
            loaded.append(json.loads(path.read_text()))
        except (OSError, ValueError) as e:
            refuse(f"cannot read {path}: {e}")
    return compare(*loaded)


if __name__ == "__main__":
    sys.exit(main())
