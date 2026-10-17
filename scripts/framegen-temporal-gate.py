#!/usr/bin/env python3
"""Temporal gate for motion-field changes: what the eye sees between real frames.

framegen-field-gate.py scores one generated frame at phase 1.0 (predictive) or
0.5 (bidir). Tearing and flicker are temporal: a generated frame can be close to
the truth on its own and still jump against its neighbours. This gate replays
the displayed sequence instead, on the same numpy port of the production chain
(checked with framegen-field-gate.py --parity).

Ground truth at the in-between phases: capture faster than the simulated source
(framegen-capture.sh with RATE=120 and a vsynced client), then treat every k-th
captured frame as a real frame. The k-1 frames in between are the truth for the
generated frames at phases j/k.

  predictive  reals f[i-k], f[i]; shows f[i], P(1/k) .. P((k-1)/k), f[i+k]
  bidir       reals f[i], f[i+k]; shows f[i], I(1/k) .. I((k-1)/k), f[i+k]
  hold        framegen off: shows f[i] until f[i+k] (judder)

With e_j = shown_j - truth_j (e_0 = e_k = 0 at the reals), on pixels the truth
changes within the interval:

  E       mean |e_j| over the generated frames (per-frame error)
  flick   mean |e_(j+1) - e_j| over all k steps = |shown change - true change|,
          including the error appearing after a real and snapping away at the
          next one; hold scores its judder here
  eflick  the same on gradient magnitude, positive part only: edges that move or
          pop when the truth's edges do not (tearing, shear)
  lost    edges of the truth missing in the generated frames (smear, blur)
  false   edges in the generated frames the truth lacks (ghosts, double edges)
  flip    predictive only: share of confident vectors that change by more than
          2 px between consecutive real intervals after motion compensation

Predictive arms use production's warp.comp from framegen-field-gate.py: the
sharp fallback by default; suffix @xf selects the crossfade fallback
(GAMESCOPE_FRAMEGEN_SHARP_FALLBACK=0).

Paired change vs the first arm with a 95% bootstrap interval, for flick.

  framegen-temporal-gate.py --size 1280x720 --k 2,3 [--arms hold,checked/8,checked/4]
                            [--stride 8] [--domain coded|linear] NAME...
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("framegen_field_gate", HERE / "framegen-field-gate.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["framegen_field_gate"] = G
_spec.loader.exec_module(G)

FLIP_PX = 2.0      # a vector change larger than this between intervals is a flip
FLIP_CONF = 0.5    # only vectors both intervals trust


def changed(frames, lo=4 / 255):
    return np.max([np.abs(f - frames[0]) for f in frames[1:]], axis=0) > lo


def fields(arm, a, b, domain):
    lum = G.linear if domain == "linear" else (lambda x: x)
    return G.arm_fields(lum(a), lum(b), arm, None)


def shown(method, arm, f, i, k, domain):
    """The displayed sequence f[i] .. f[i+k] for one arm, and its flip rate (or None)."""
    B, C = f[i], f[i + k]
    if arm == "hold":
        return [B] * k + [C], None
    arm, _, out = arm.partition("@")
    if method == "predictive":
        A = f[i - k]
        X, _, s = fields(arm, A, B, domain)
        Xp, _, _ = fields(arm, f[i - 2 * k], A, domain)
        gen = [G.predict(A, B, X, s, j / k, sharp=out != "xf") for j in range(1, k)]
        return [B] + gen + [C], flip_rate(Xp, X, s)
    X, Y, s = fields(arm, B, C, domain)
    return [B] + [G.interpolate(B, C, X, Y, s, j / k) for j in range(1, k)] + [C], None


def flip_rate(Xp, Xc, s):
    """Share of confident vectors of Xc that disagree with where Xp (one interval earlier) had them."""
    h, w = Xc.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vx, vy = Xc[..., 0], Xc[..., 1]
    sy = np.clip(np.rint(yy - vy), 0, h - 1).astype(np.int64)
    sx = np.clip(np.rint(xx - vx), 0, w - 1).astype(np.int64)
    old = Xp[sy, sx]
    trust = (Xc[..., 2] > FLIP_CONF) & (old[..., 2] > FLIP_CONF) & (np.abs(Xc[..., :2]).sum(-1) * s > 1.0)
    if trust.sum() < 50:
        return None
    jump = np.abs(Xc[..., :2] - old[..., :2]).max(-1) * s > FLIP_PX
    return float(jump[trust].mean())


def score(seq, truth, mask):
    e = [d - t for d, t in zip(seq, truth)]
    gs, gt = [G.grad(d) for d in seq], [G.grad(t) for t in truth]
    k = len(seq) - 1
    E = float(np.mean([np.abs(e[j])[mask].mean() for j in range(1, k)]))
    flick = float(np.mean([np.abs(e[j + 1] - e[j])[mask].mean() for j in range(k)]))
    eflick = float(np.mean([np.maximum(np.abs(gs[j + 1] - gs[j]) - np.abs(gt[j + 1] - gt[j]), 0)[mask].mean()
                            for j in range(k)]))
    lost = float(np.mean([np.maximum(gt[j] - gs[j], 0)[mask].mean() for j in range(1, k)]))
    false = float(np.mean([np.maximum(gs[j] - gt[j], 0)[mask].mean() for j in range(1, k)]))
    return E, flick, eflick, lost, false


def interval(f, i, k, arms, domain, methods):
    truth = [f[i + j] for j in range(k + 1)]
    mask = changed(truth)
    if mask.sum() < 500:
        return None
    rows = {}
    for method in methods:
        for arm in arms:
            seq, flip = shown(method, arm, f, i, k, domain)
            rows[(method, arm)] = score(seq, truth, mask) + (flip,)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="+", help="capture names (NAME-WxH.gray), captured faster than the source")
    ap.add_argument("--size", required=True, help="WxH")
    ap.add_argument("--captures", type=Path, default=Path.home() / ".cache/gamescope-framegen-captures")
    ap.add_argument("--k", default="2,3", help="captured frames per simulated real frame")
    ap.add_argument("--stride", type=int, default=8, help="real intervals between scored ones")
    ap.add_argument("--arms", default="hold,checked/8,checked/4")
    ap.add_argument("--domain", choices=("coded", "linear"), default="coded")
    ap.add_argument("--methods", default="predictive,bidir")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    arms, methods = args.arms.split(","), args.methods.split(",")
    rng = np.random.default_rng(1)
    print(f"{args.size} {args.domain}: E | flick [paired change vs {arms[0]}, 95% CI] | eflick | lost | false | flip")
    for name in args.names:
        f = np.fromfile(args.captures / f"{name}-{args.size}.gray", np.uint8).reshape(-1, h, w)
        dup = np.mean([not np.any(f[i] != f[i - 1]) for i in range(1, len(f))])
        if dup > 0.02:
            print(f"{name}: {dup:.1%} repeated frames, the client was slower than the capture; truth is not evenly spaced")
        f = f.astype(np.float32) / 255
        for k in (int(v) for v in args.k.split(",")):
            rows = [r for r in (interval(f, i, k, arms, args.domain, methods)
                                for i in range(2 * k, len(f) - k, args.stride * k)) if r]
            if not rows:
                print(f"{name} k={k}: no interval with motion")
                continue
            for method in methods:
                ref = np.array([r[(method, arms[0])][1] for r in rows])
                line = f"{name:24s} k={k} {method:10s} n={len(rows):2d}"
                for arm in arms:
                    v = np.array([r[(method, arm)][:5] for r in rows])
                    d = v[:, 1] / ref - 1
                    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
                    lo, hi = np.percentile(boot, [2.5, 97.5]) * 100
                    flips = [r[(method, arm)][5] for r in rows if r[(method, arm)][5] is not None]
                    flip = f" flip {np.mean(flips):.3f}" if flips else ""
                    m = v.mean(0)
                    line += (f" | {arm}: E {m[0]:.4f} flick {m[1]:.4f} {d.mean() * 100:+.1f}% [{lo:+.1f},{hi:+.1f}]"
                             f" eflick {m[2]:.4f} lost {m[3]:.4f} false {m[4]:.4f}{flip}")
                print(line, flush=True)


if __name__ == "__main__":
    main()
