#!/usr/bin/env python3
"""Offline split of the predictive path's error into matching and continuation.

Works on GAMESCOPE_FRAMEGEN_RECORD (GSFD) captures. Each sample holds the
previous and current real frame's field-resolution luma and the checked motion
field between them (cur(p) = prev(p - F(p)), confidence in channel 2). For each
pair of consecutive samples, real frame N+1 is predicted from real frame N:

  hold    L_N(p)                     the frame is held
  cont    L_N(p - F_N(p))            constant-motion continuation (predictive path)
  cont*   L_N(p - a F_N(p))          best scale a per pair (cadence-aware continuation)
  oracle  L_N(p - F_N+1(p))          the next pair's estimated field: what it leaves

S1 is the oracle error: the residual left by the next pair's own estimated
field, an upper bound on what better matching could remove (it also holds
disocclusion, shading change and resampling error). S2, the continuation gap,
is cont - oracle: what no better matcher can remove, only a better predictor.

The rejection block asks, inside the moving texels the checked field rejects
(confidence below 0.5), whether warping would have beaten holding.

Scope: field-resolution luma, absolute error, moving texels only. It is not a
colour-domain perceptual score (the same limit as framegen-net-eval.py).

Usage:
  framegen-hindsight-eval.py --data /path/to/capture/dir
"""

import argparse
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

# Share the trainer's GSFD parser and bilinear sampler, so the two never drift.
_TRAIN_PATH = Path(__file__).with_name("framegen-net-train.py")
_spec = importlib.util.spec_from_file_location("framegen_net_train", _TRAIN_PATH)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

MOVING = 0.25  # field texels per frame
LOW_CONFIDENCE = 0.5
MIN_MOVING = 50
MIN_KEPT = 10
SCALES = np.linspace(0.0, 2.0, 21)


def iter_consecutive(files, stats):
    """Yield (cur, nxt) when nxt's previous frame is cur's current frame.
    Holds two samples at a time."""
    prev = None
    for path in files:
        try:
            cur = T.load_sample(path)
        except (ValueError, struct.error) as e:
            print(f"skipping {path}: {e}", file=sys.stderr)
            stats["breaks"] += 1
            prev = None
            continue
        stats["samples"] += 1
        if prev is not None:
            if prev[1].shape == cur[0].shape and np.abs(cur[0] - prev[1]).max() <= 1e-3:
                yield prev, cur
            else:
                stats["breaks"] += 1
        prev = cur


def texel_centres(shape):
    cy, cx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return cy + 0.5, cx + 0.5


def warp(luma, field, scale, centres):
    cy, cx = centres
    return T.bilinear(luma, cy - scale * field[..., 1], cx - scale * field[..., 0])[0]


def speed(field):
    return np.hypot(field[..., 0], field[..., 1])


def split_pair(cur, nxt, centres):
    luma, field = cur[1], cur[2].astype(np.float32)
    truth, next_field = nxt[1], nxt[2].astype(np.float32)
    # Motion in either field: the oracle warps with the next one.
    moving = np.maximum(speed(field), speed(next_field)) >= MOVING
    if moving.sum() < MIN_MOVING:
        return None

    def err(prediction):
        return float(np.abs(prediction - truth)[moving].mean())

    hold = err(luma)
    cont = err(warp(luma, field, 1.0, centres))
    oracle = err(warp(luma, next_field, 1.0, centres))
    best_cont, best_scale = min((err(warp(luma, field, a, centres)), a) for a in SCALES)
    low_conf = float((field[..., 2][moving] < LOW_CONFIDENCE).mean())
    return hold, cont, best_cont, best_scale, oracle, low_conf, float(moving.mean())


def rejection_pair(cur, nxt, centres):
    luma, field, truth = cur[1], cur[2].astype(np.float32), nxt[1]
    # Motion in the judged field only: the one generation uses. This is a
    # narrower set than split_pair's, so its rejected share differs slightly.
    moving = speed(field) >= MOVING
    conf = field[..., 2]
    killed = moving & (conf < LOW_CONFIDENCE)
    kept = moving & (conf >= LOW_CONFIDENCE)
    if killed.sum() < MIN_MOVING or kept.sum() < MIN_KEPT:
        return None
    warp_err = np.abs(warp(luma, field, 1.0, centres) - truth)
    hold_err = np.abs(luma - truth)
    gain = hold_err[killed] - warp_err[killed]
    conf_moving = conf[moving]
    return ((gain > 0).mean(), gain.mean(), np.percentile(-gain, 95), (hold_err[kept] > warp_err[kept]).mean(),
            (conf_moving < 0.1).mean(), ((conf_moving >= 0.1) & (conf_moving < LOW_CONFIDENCE)).mean())


def main():
    ap = argparse.ArgumentParser(description="Split predictive error into matching (S1) and continuation (S2).")
    ap.add_argument("--data", required=True, type=Path, help="GAMESCOPE_FRAMEGEN_RECORD capture directory")
    args = ap.parse_args()

    files = sorted(args.data.glob("fg_*.bin"))
    if not files:
        sys.exit(f"no fg_*.bin samples under {args.data}")
    stats = {"samples": 0, "breaks": 0}
    split, rejection, centres, shape = [], [], None, None
    for cur, nxt in iter_consecutive(files, stats):
        if centres is None or cur[1].shape != shape:
            shape = cur[1].shape
            centres = texel_centres(shape)
        row = split_pair(cur, nxt, centres)
        if row:
            split.append(row)
        row = rejection_pair(cur, nxt, centres)
        if row:
            rejection.append(row)
    if not split:
        sys.exit(f"no usable consecutive pair with at least {MIN_MOVING} moving texels under {args.data}")

    R = np.array(split)
    print(f"samples={stats['samples']} pairs_used={len(R)} non_consecutive={stats['breaks']} "
          f"field={shape[1]}x{shape[0]}")
    med = np.median(R, axis=0)
    print(f"median over moving texels (|F| >= {MOVING} field texel, abs luma error):")
    print(f"  hold={med[0]:.4f}  cont(a=1)={med[1]:.4f}  cont(best a)={med[2]:.4f} (median a={med[3]:.2f})"
          f"  oracle(S1)={med[4]:.4f}")
    print(f"  S2 = cont - oracle = {med[1] - med[4]:.4f}  ({(med[1] - med[4]) / max(med[1], 1e-9) * 100:.0f}% of cont"
          f" error);  cadence-aware gain = {med[1] - med[2]:.4f}")
    print(f"  low-conf share of moving field={med[5] * 100:.0f}%  moving share of frame={med[6] * 100:.0f}%")
    for q in (75, 90):
        p = np.percentile(R, q, axis=0)
        print(f"  p{q}: hold={p[0]:.4f} cont={p[1]:.4f} cont*={p[2]:.4f} oracle={p[4]:.4f}")
    print("a distribution (p10 p25 p50 p75 p90):", np.round(np.percentile(R[:, 3], [10, 25, 50, 75, 90]), 2))

    if not rejection:
        print(f"rejection: no pair with {MIN_MOVING} rejected and {MIN_KEPT} kept moving texels")
        return
    print(f"rejection inside the judged field's moving texels ({len(rejection)} pairs):")
    names = ("killed_warp_better", "killed_gain", "killed_warp_worse_p95", "kept_warp_better", "conf_lt", "conf_mid")
    for name, values in zip(names, np.array(rejection).T):
        print(f"  {name:24s} median={np.median(values):.4f}  p10={np.percentile(values, 10):.4f}"
              f"  p90={np.percentile(values, 90):.4f}")


if __name__ == "__main__":
    main()
