#!/usr/bin/env python3
"""Offline perceptual/temporal evaluator for gamescope's motion-field refiner.

This is the Stage-C companion to framegen-net-train.py and implements Gap E1 of
doc/framegen-proposals/07-frames-only-sota-alignment.md. The frame-generation
research (doc/research-framegen.md, section 6) is explicit that L1/PSNR under-
measure exactly the artifacts framegen produces — ghosting, edge smear, flicker
— and that structural (SSIM) and perceptual/temporal (LPIPS, DISTS, FvVDP)
metrics are the field standard. Our live grading (the Stage-B4 probe) and the
trainer both optimise a scalar photometric residual; this tool adds the
structural and temporal view the literature asks for, on data we already have.

SCOPE (important, and a deliberate ground-truth limitation):
  GAMESCOPE_FRAMEGEN_RECORD captures are FIELD-RESOLUTION LUMA + motion fields
  (the exact tensors the net trains on), NOT the full-resolution COLOR output
  frame. So this tool grades the FIELD's photometric prediction (warp the source
  luma along the field, compare to the destination luma) with structural +
  temporal metrics. True LPIPS/DISTS/FvVDP on the final colour frame need
  full-res generated+ground-truth colour pairs, which the capture does not hold
  — that is Gap E2 (a capture extension), documented in proposal 07, not this
  script. What is measured here is real and useful; it is not a colour-domain
  perceptual score, and it does not pretend to be.

It reuses framegen-net-train.py's verified GSFD parser and net so the two never
drift. With --net it also runs the trained refiner and reports the neutral
(Stage B) vs refined deltas, so a checkpoint can be judged on structure/stability
rather than residual alone before it is shipped.

numpy only, CPU only — same rationale as the trainer.

Usage:
  framegen-net-eval.py --data /path/to/capture/dir
  framegen-net-eval.py --data /path/to/capture/dir --net weights.bin
"""

import argparse
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

# Load the trainer as a module despite its hyphenated filename, so the GSFD
# parser, bilinear sampler and net forward pass are shared verbatim (they MUST
# stay in lockstep with cs_framegen_motion_net.comp; duplicating them here would
# invite exactly the drift the trainer warns against).
_TRAIN_PATH = Path(__file__).with_name("framegen-net-train.py")
_spec = importlib.util.spec_from_file_location("framegen_net_train", _TRAIN_PATH)
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


# --------------------------------------------------------------------------
# Weights blob (reverse of the trainer's export())
# --------------------------------------------------------------------------

def load_blob(path):
    raw = Path(path).read_bytes()
    magic, version, nlayers = struct.unpack_from("<3I", raw, 0)
    if magic != T.MAGIC_BLOB:
        sys.exit(f"{path}: not a GSFR weights blob")
    if nlayers != len(T.LAYERS):
        sys.exit(f"{path}: layer count {nlayers} != expected {len(T.LAYERS)}")
    # The trainer writes the whole shape table first, then all weight/bias data.
    off = 12
    shapes = []
    for i, (exp_ci, exp_co) in enumerate(T.LAYERS, 1):
        ci, co, ksize = struct.unpack_from("<3I", raw, off)
        off += 12
        if (ci, co, ksize) != (exp_ci, exp_co, T.KSIZE):
            sys.exit(f"{path}: layer {i} shape mismatch")
        shapes.append((ci, co, ksize))
    params = {}
    for i, (ci, co, ksize) in enumerate(shapes, 1):
        n = co * ksize * ksize * ci
        w = np.frombuffer(raw, "<f4", count=n, offset=off).reshape(co, ksize, ksize, ci)
        off += n * 4
        params[f"W{i}"] = np.ascontiguousarray(w.transpose(0, 3, 1, 2))  # -> (co,ci,ky,kx)
        params[f"b{i}"] = np.array(
            np.frombuffer(raw, "<f4", count=co, offset=off), np.float32)
        off += co * 4
    if version != T.BLOB_VERSION:
        print(f"note: blob version {version} != trainer {T.BLOB_VERSION}; "
              f"shapes match, evaluating anyway", file=sys.stderr)
    return params


# --------------------------------------------------------------------------
# Warp + refine over a full frame (mirrors the trainer's loss construction and
# cs_framegen_motion_net.comp's refined-field formula, but frame-wide)
# --------------------------------------------------------------------------

def warp(luma_src, flow):
    """Warp luma_src along a (H,W,2) field: dst(p) = src(p - flow)."""
    h, w = luma_src.shape
    py, px = np.mgrid[0:h, 0:w].astype(np.float32)
    val, _, _ = T.bilinear(luma_src, py + 0.5 - flow[..., 1], px + 0.5 - flow[..., 0])
    return val


def full_features(X, Y, luma_src, luma_dst):
    """The 12 channels of cs_framegen_motion_net.comp / trainer.features(),
    built over the whole frame (mode channel is 0, as the trainer uses it)."""
    h, w = luma_dst.shape
    lw = warp(luma_src, X[..., 0:2])
    resid = np.abs(luma_dst - lw) / np.maximum(1.0, np.maximum(np.abs(luma_dst), np.abs(lw)))
    f = np.empty((12, h, w), np.float32)
    f[0] = X[..., 0] * 0.125
    f[1] = X[..., 1] * 0.125
    f[2] = X[..., 2]
    f[3] = np.minimum(X[..., 3] * 0.5, 1.0)
    f[4] = Y[..., 0] * 0.125
    f[5] = Y[..., 1] * 0.125
    f[6] = Y[..., 2]
    f[7] = np.minimum(Y[..., 3] * 0.5, 1.0)
    f[8] = luma_dst
    f[9] = lw
    f[10] = resid
    f[11] = 0.0
    return f


def refine(params, X, Y, luma_src, luma_dst):
    """Return (refined_flow (H,W,2), refined_conf (H,W)) from the net, matching
    the inference-time evidence gate. A zero-head blob returns the raw field."""
    feat = full_features(X, Y, luma_src, luma_dst)[None]                # (1,12,H,W)
    pad = (T.KSIZE - 1) // 2 * len(T.LAYERS)                            # 3 valid 3x3 convs
    feat = np.pad(feat, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="edge")
    o, _ = T.net_fwd(params, feat)                                     # (1,4,H,W)
    o = o[0]
    dF = 2.0 * np.tanh(o[0:2]).transpose(1, 2, 0)                      # (H,W,2)
    Fh = X[..., 0:2].astype(np.float32) + dF
    warped = warp(luma_src, Fh)
    rr = np.abs(warped - luma_dst) / np.maximum(1.0, np.maximum(np.abs(warped), np.abs(luma_dst)))
    t = np.clip((rr - 0.03) / 0.07, 0.0, 1.0)
    gate = 1.0 - t * t * (3.0 - 2.0 * t)                              # complement smoothstep
    dc_raw = o[2]
    dc = dc_raw * np.where(dc_raw < 0.0, 1.0, gate)
    ch = np.clip(X[..., 2].astype(np.float32) + dc, 0.0, 1.0)
    return Fh, ch


# --------------------------------------------------------------------------
# Metrics (structural + temporal; all scale-robust for SDR and HDR luma)
# --------------------------------------------------------------------------

def _boxmean(a, r):
    ap = np.pad(a, r, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(ap, (2 * r + 1, 2 * r + 1))
    return win.mean(axis=(-1, -2))


def ssim(a, b, r=3):
    """Single-scale SSIM over a (2r+1)^2 window; dynamic range from the target."""
    L = max(float(b.max() - b.min()), 1e-3)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_a, mu_b = _boxmean(a, r), _boxmean(b, r)
    va = _boxmean(a * a, r) - mu_a * mu_a
    vb = _boxmean(b * b, r) - mu_b * mu_b
    vab = _boxmean(a * b, r) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / \
        ((mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2))
    return float(np.clip(s, -1.0, 1.0).mean())


def edge_err(a, b):
    """Mean |grad(a)| vs |grad(b)| difference, scale-normalised: catches the
    boundary smear/ghosting that a flat residual average hides."""
    ga = np.hypot(*np.gradient(a))
    gb = np.hypot(*np.gradient(b))
    return float(np.mean(np.abs(ga - gb)) / max(float(gb.mean()), 1e-3))


def resid_map(warped, dst):
    return np.abs(warped - dst) / np.maximum(1.0, np.maximum(np.abs(warped), np.abs(dst)))


def grade_pair(flow, conf, luma_src, luma_dst):
    """All metrics for one warped field against its destination luma."""
    warped = warp(luma_src, flow)
    rm = resid_map(warped, luma_dst)
    cw = float(conf.sum()) or 1.0
    return {
        "resid": float(rm.mean()),
        "resid_conf": float((rm * conf).sum() / cw),   # weighted where it actually ships
        "bad%": float((rm > 0.10).mean() * 100.0),
        "ssim": ssim(warped, luma_dst),
        "edge": edge_err(warped, luma_dst),
    }, rm


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def evaluate(samples, params):
    """Grade every sample in both field directions (fwd: prev->cur, rev:
    cur->prev), for the neutral field and, if params given, the refined field."""
    acc = {"neutral": [], "refined": []}
    seq = {"neutral": [], "refined": []}   # per-frame mean resid, for temporal σ
    for lp, lc, ff, fr in samples:
        for X, Y, ls, ld in ((ff, fr, lp, lc), (fr, ff, lc, lp)):
            neu, rm_n = grade_pair(X[..., 0:2].astype(np.float32),
                                   X[..., 2].astype(np.float32), ls, ld)
            acc["neutral"].append(neu)
            seq["neutral"].append(neu["resid"])
            if params is not None:
                Fh, ch = refine(params, X, Y, ls, ld)
                ref, rm_r = grade_pair(Fh, ch, ls, ld)
                acc["refined"].append(ref)
                seq["refined"].append(ref["resid"])
    out = {}
    for key, rows in acc.items():
        if not rows:
            continue
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        s = np.asarray(seq[key])
        # Temporal stability: how much the per-frame residual lurches. Lower is
        # steadier (the flash/particle-chaos class the adaptation probe fights).
        agg["temporal_sigma"] = float(s.std())
        out[key] = agg
    return out


def fmt(name, m):
    return (f"  {name:8s}  resid={m['resid']:.5f}  conf-resid={m['resid_conf']:.5f}  "
            f"bad={m['bad%']:.2f}%  ssim={m['ssim']:.4f}  edge={m['edge']:.4f}  "
            f"temporal_sigma={m['temporal_sigma']:.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="capture dir (GAMESCOPE_FRAMEGEN_RECORD)")
    ap.add_argument("--net", help="GSFR weights blob to also evaluate (refined field)")
    args = ap.parse_args()

    samples = T.load_dataset(args.data)
    params = load_blob(args.net) if args.net else None
    res = evaluate(samples, params)

    print("\nfield-resolution photometric quality "
          "(structural + temporal; higher ssim / lower everything else is better):")
    print(fmt("neutral", res["neutral"]))
    if "refined" in res:
        print(fmt("refined", res["refined"]))
        n, r = res["neutral"], res["refined"]
        d_resid = (n["resid"] - r["resid"]) / max(n["resid"], 1e-9) * 100.0
        d_bad = n["bad%"] - r["bad%"]
        d_ssim = r["ssim"] - n["ssim"]
        d_edge = (n["edge"] - r["edge"]) / max(n["edge"], 1e-9) * 100.0
        print(f"\n  net effect: resid {d_resid:+.1f}%  bad {d_bad:+.2f}pp  "
              f"ssim {d_ssim:+.4f}  edge {d_edge:+.1f}%  "
              f"(positive resid/edge %, +ssim, -bad = improvement)")


if __name__ == "__main__":
    main()
