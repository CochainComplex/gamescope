#!/usr/bin/env python3
"""Offline gate for motion-field changes, on captured consecutive frames.

A numpy port of the production chain decides, per content class and before
any runtime code, whether a field arm helps each generation method:

  field   luma pair (on the composite's code values, like production's
          srgbView(), which is the UNORM view; 4 bilinear taps per block),
          2x pyramid to /32, R=4 search at /32, vector-median seeded +/-1
          refine, zero anchor, confidence and sub-texel fit, forward-backward
          check (cs_framegen_motion_{luma_pair,pyramid,match,match_refine,
          fbcheck}.comp)
  output  predictive: warp.comp (second-source agreement; sharp fallback:
          displacement scaled by confidence, as production's default);
          bidir: bidir.comp (phase weights, mixedness x agreement,
          crossfade fallback)

Ground truth from triplets A, B, C of distinct consecutive frames:
predictive predicts C from B along A->B, bidir rebuilds B from A and C.
Per class and method, against the first arm, paired over triplets:

  E_chg   abs error where the frame changed (|truth - fallback| > 4/255)
  sta     abs error on static pixels (|truth - fallback| <= 1/255): drag
  lost    edges in the truth the output lacks (smear)
  false   edges the output invents (tearing, block shear)

Captures come from framegen-capture.sh (NAME-WxH.gray). Arms:
  checked/8, checked/4       production field at /8 or /4 (tolerances in px)
  ...+vm                     3x3 vector median on the final field
  ...+nc                     confidence kept only where a vector agrees with
                             its 3x3 vector median within the FB tolerance
  learned/8                  checked/8 refined by a GSFR net (--net)
  ...@xf                     predictive output with the crossfade fallback
                             (GAMESCOPE_FRAMEGEN_SHARP_FALLBACK=0)

  framegen-field-gate.py --size 1280x720 [--stride 6] [--arms a,b,...] [--domain coded|linear] NAME...

The port's numbers count only while it reproduces the GPU. --parity DIR runs
the checked/8 chain on the lumas of GAMESCOPE_FRAMEGEN_RECORD samples (GSFD)
and compares with both fields the GPU recorded; it fails below 99.9% identical
vectors, above a mean confidence error of 0.01 in any field, or on non-finite
values.

  framegen-field-gate.py --parity DIR
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("framegen_net_train", "framegen-net-train.py")
NE = _load("framegen_net_eval", "framegen-net-eval.py")

REG = 0.02                             # match regulariser per texel
FB_TOL, FB_SLOPE = 0.75, 0.05          # FB tolerance at /8, slope per texel
AGREE_LO, AGREE_HI = 0.12, 0.45        # k_flFramegenAgreeLo/Hi
SUPPRESS_LO, SUPPRESS_HI = 0.08, 0.40  # k_flFramegenSuppressLo/Hi
COARSEST = 32


def bilinear(img, y, x):
    return T.bilinear(img, y, x)[0]


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def linear(coded):
    """sRGB decode, for the --domain linear what-if: production binds srgbView(),
    which despite its name is the UNORM view, so 8/10-bit content is matched on
    code values (float content would be linear already)."""
    return np.where(coded <= 0.04045, coded / 12.92, ((coded + 0.055) / 1.055) ** 2.4).astype(np.float32)


# ---------------------------------------------------------------- matcher

def base_luma(full, s):
    h, w = full.shape
    lh, lw = -(-h // s), -(-w // s)
    cy, cx = np.mgrid[0:lh, 0:lw].astype(np.float32)
    yc, xc = (cy + 0.5) / lh * h, (cx + 0.5) / lw * w
    oy, ox = 0.25 / lh * h, 0.25 / lw * w
    return (sum(bilinear(full, yc + a * oy, xc + b * ox) for a in (-1, 1) for b in (-1, 1)) * 0.25).astype(np.float32)


def half(img):
    h, w = img.shape
    lh, lw = max(1, -(-h // 2)), max(1, -(-w // 2))
    cy, cx = np.mgrid[0:lh, 0:lw].astype(np.float32)
    return bilinear(img, (cy + 0.5) / lh * h, (cx + 0.5) / lw * w).astype(np.float32)


def _fetch(img, y, x):
    h, w = img.shape
    return img[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]


def _cur9(cur, yy, xx):
    return [_fetch(cur, yy + dy, xx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def _sad(cur9, prev, yy, xx, mvy, mvx):
    s = np.zeros(yy.shape, np.float32)
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s += np.abs(cur9[k] - _fetch(prev, yy + dy - mvy, xx + dx - mvx))
            k += 1
    return s


def _conf_subpixel(cur9, prev, yy, xx, bmy, bmx, best, zero):
    conf = np.clip(1 - best / 9 * 6, 0, 1) * np.clip((zero - best) * 4, 0, 1)
    sub = [bmx.astype(np.float32), bmy.astype(np.float32)]
    sc = _sad(cur9, prev, yy, xx, bmy, bmx)
    for axis, (m, p) in enumerate((((0, -1), (0, 1)), ((-1, 0), (1, 0)))):
        sm = _sad(cur9, prev, yy, xx, bmy + m[0], bmx + m[1])
        sp = _sad(cur9, prev, yy, xx, bmy + p[0], bmx + p[1])
        den = sm - 2 * sc + sp
        ok = den > 1e-5
        sub[axis] = np.where(ok, sub[axis] + np.clip(0.5 * (sm - sp) / np.where(ok, den, 1), -0.5, 0.5), sub[axis])
    return np.stack([sub[0], sub[1], conf], -1)


def _match(prev, cur, R=4):
    yy, xx = np.mgrid[0:cur.shape[0], 0:cur.shape[1]]
    cur9 = _cur9(cur, yy, xx)
    zero = _sad(cur9, prev, yy, xx, 0, 0)
    best, bmy, bmx = zero.copy(), np.zeros_like(yy), np.zeros_like(xx)
    for my in range(-R, R + 1):
        for mx in range(-R, R + 1):
            if mx or my:
                s = _sad(cur9, prev, yy, xx, my, mx) + REG * (abs(mx) + abs(my))
                b = s < best
                best, bmy, bmx = np.where(b, s, best), np.where(b, my, bmy), np.where(b, mx, bmx)
    return _conf_subpixel(cur9, prev, yy, xx, bmy, bmx, _sad(cur9, prev, yy, xx, bmy, bmx), zero)


def _neighbours(field, shape, coarse):
    """The 3x3 vectors around each texel: of the coarser level (coarse=True,
    doubled grid) or of the field itself."""
    h, w = field.shape[:2]
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = (yy >> 1, xx >> 1) if coarse else (yy, xx)
    return np.stack([field[np.clip(cy + dy, 0, h - 1), np.clip(cx + dx, 0, w - 1), :2]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)


def vector_median(nb):
    """Minimum L1-distance-sum vector of the 9 (first wins, as the shader)."""
    dist = np.abs(nb[:, None] - nb[None, :]).sum(-1).sum(1)
    return np.take_along_axis(nb, np.argmin(dist, 0)[None, ..., None], 0)[0]


def _refine(prev, cur, coarse, final):
    seedF = vector_median(_neighbours(coarse, cur.shape, True)) * 2.0
    yy, xx = np.mgrid[0:cur.shape[0], 0:cur.shape[1]]
    sx, sy = np.rint(seedF[..., 0]).astype(np.int64), np.rint(seedF[..., 1]).astype(np.int64)
    cur9 = _cur9(cur, yy, xx)
    bestCost = np.full(cur.shape, 1e30, np.float32)
    bestSad, bmy, bmx = bestCost.copy(), sy.copy(), sx.copy()
    for my in (-1, 0, 1):
        for mx in (-1, 0, 1):
            vy, vx = sy + my, sx + mx
            s = _sad(cur9, prev, yy, xx, vy, vx)
            cost = s + REG * (np.abs(vx - seedF[..., 0]) + np.abs(vy - seedF[..., 1]))
            b = cost < bestCost
            bestCost, bestSad = np.where(b, cost, bestCost), np.where(b, s, bestSad)
            bmy, bmx = np.where(b, vy, bmy), np.where(b, vx, bmx)
    zero = _sad(cur9, prev, yy, xx, 0, 0)
    z = zero + REG * (np.abs(seedF[..., 0]) + np.abs(seedF[..., 1])) < bestCost
    bestSad, bmy, bmx = np.where(z, zero, bestSad), np.where(z, 0, bmy), np.where(z, 0, bmx)
    if not final:
        return np.stack([bmx.astype(np.float32), bmy.astype(np.float32), np.zeros(cur.shape, np.float32)], -1)
    return _conf_subpixel(cur9, prev, yy, xx, bmy, bmx, bestSad, zero)


def field_from_base(prev_b, cur_b, s):
    levels = [(prev_b, cur_b)]
    while s * 2 ** (len(levels) - 1) < COARSEST:
        levels.append((half(levels[-1][0]), half(levels[-1][1])))
    f = _match(*levels[-1])
    for i in range(len(levels) - 2, -1, -1):
        f = _refine(levels[i][0], levels[i][1], f, i == 0)
    return f


def fbcheck(F, R, tol):
    h, w = F.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    def factor(py, px):
        py, px = np.clip(py, 0, h - 1), np.clip(px, 0, w - 1)
        f = F[py, px, :2]
        by = np.floor(py + 0.5 - f[..., 1] - 0.5).astype(np.int64)
        bx = np.floor(px + 0.5 - f[..., 0] - 0.5).astype(np.int64)
        fac, err = np.zeros((h, w), np.float32), np.full((h, w), 1e30, np.float32)
        for dy in (0, 1):
            for dx in (0, 1):
                r = R[np.clip(by + dy, 0, h - 1), np.clip(bx + dx, 0, w - 1), :2]
                e = np.abs(f + r).sum(-1)
                t = tol + FB_SLOPE * (np.abs(f).sum(-1) + np.abs(r).sum(-1))
                fac, err = np.maximum(fac, np.clip((2 * t - e) / t, 0, 1)), np.minimum(err, e)
        return fac, err

    tent, acc, wsum, errc = (1.0, 2.0, 1.0), 0.0, 0.0, None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            fac, err = factor(yy + dy, xx + dx)
            acc, wsum = acc + fac * tent[dx + 1] * tent[dy + 1], wsum + tent[dx + 1] * tent[dy + 1]
            errc = err if dx == 0 and dy == 0 else errc
    out = F.copy()
    out[..., 2] = F[..., 2] * (acc / wsum)
    return np.concatenate([out[..., :3], errc[..., None]], -1)


# ---------------------------------------------------------------- arms

def arm_fields(prev_l, cur_l, arm, net):
    """Checked forward (cur-anchored) and reverse fields, and the scale."""
    s = 4 if "/4" in arm else 8
    pb, cb = base_luma(prev_l, s), base_luma(cur_l, s)
    Fw, Rv = field_from_base(pb, cb, s), field_from_base(cb, pb, s)
    tol = FB_TOL * 8 / s
    X, Y = fbcheck(Fw, Rv, tol), fbcheck(Rv, Fw, tol)
    for suffix in arm.split("+")[1:]:
        X, Y = (post(X, suffix, tol), post(Y, suffix, tol))
    if arm.startswith("learned"):
        Fh, ch = NE.refine(net, X, Y, pb, cb)
        X = np.concatenate([Fh, ch[..., None], X[..., 3:]], -1).astype(np.float32)
    return X, Y, s


def post(F, how, tol):
    vm = vector_median(_neighbours(F, F.shape[:2], False))
    out = F.copy()
    if how == "vm":
        out[..., :2] = vm
    elif how == "nc":
        dev = np.abs(F[..., :2] - vm).sum(-1)
        out[..., 2] = F[..., 2] * (1 - smoothstep(tol, 2 * tol, dev))
    else:
        raise ValueError(how)
    return out


# ---------------------------------------------------------------- output

def sample_field(F, s, h, w):
    py, px = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    return [bilinear(F[..., k], py / s, px / s) for k in range(3)], py, px


def predict(A, B, F, s, strength=1.0, sharp=True):
    """warp.comp. Sharp (the default): where confidence is low, move B only as
    far as the confidence allows; else crossfade with the bounded extrapolation."""
    h, w = B.shape
    (vx, vy, c), py, px = sample_field(F, s, h, w)
    motion = bilinear(B, py - s * vy * strength, px - s * vx * strength)
    second = bilinear(A, py - s * vy * (strength + 1), px - s * vx * (strength + 1))
    c = c * (1 - smoothstep(AGREE_LO, AGREE_HI, np.abs(motion - second) / np.maximum(1.0, np.abs(motion))))
    if sharp:
        return bilinear(B, py - s * vy * strength * c, px - s * vx * strength * c)
    delta = B - A
    extra = B + delta * strength * (1 - smoothstep(SUPPRESS_LO, SUPPRESS_HI, np.abs(delta)))
    pad = np.pad(B, 1, mode="edge")
    cross = [pad[1:-1, 2:], pad[1:-1, :-2], pad[2:, 1:-1], pad[:-2, 1:-1], B]
    extra = np.clip(extra, np.minimum.reduce(cross), np.maximum.reduce(cross))
    return c * motion + (1 - c) * extra


def interpolate(A, C, F, R, s, t=0.5):
    h, w = A.shape
    (fx, fy, cf), py, px = sample_field(F, s, h, w)
    (rx, ry, cr), _, _ = sample_field(R, s, h, w)
    from_cur = bilinear(C, py + s * fy * (1 - t), px + s * fx * (1 - t))
    from_prev = bilinear(A, py + s * ry * t, px + s * rx * t)
    wf, wr = cf * t, cr * (1 - t)
    wsum = wf + wr
    warped = (wf * from_cur + wr * from_prev) / np.maximum(wsum, 1e-6)
    mixed = 4 * wf * wr / np.maximum(wsum * wsum, 1e-9)
    mx = np.maximum(np.abs(from_cur), np.abs(from_prev))
    validity = np.clip(wsum, 0, 1) * (1 - mixed * smoothstep(AGREE_LO, AGREE_HI, np.abs(from_cur - from_prev) / np.maximum(1.0, mx)))
    return validity * warped + (1 - validity) * ((1 - t) * A + t * C)


# ---------------------------------------------------------------- metrics

def grad(img):
    gy, gx = np.gradient(img)
    return np.hypot(gx, gy)


def score(pred, truth, fallback):
    d = np.abs(truth - fallback)
    chg, sta = d > 4 / 255, d <= 1 / 255
    if chg.sum() < 500:
        return None
    e = np.abs(pred - truth)
    gp, gt = grad(pred), grad(truth)
    return (float(e[chg].mean()), float(e[sta].mean()) if sta.any() else 0.0,
            float(np.maximum(gt - gp, 0)[chg].mean()), float(np.maximum(gp - gt, 0)[chg].mean()))


def triplet(A, B, C, arms, net, domain):
    Al, Bl, Cl = (linear(A), linear(B), linear(C)) if domain == "linear" else (A, B, C)
    rows = {}
    for arm in arms:
        field, _, out = arm.partition("@")
        X, _, s = arm_fields(Al, Bl, field, net)
        rp = score(predict(A, B, X, s, sharp=out != "xf"), C, B)
        X, Y, s = arm_fields(Al, Cl, field, net)
        rb = score(interpolate(A, C, X, Y, s), B, 0.5 * (A + C))
        if rp is None or rb is None:
            return None
        rows[("predictive", arm)], rows[("bidir", arm)] = rp, rb
    return rows


def distinct_frames(path, w, h):
    f = np.fromfile(path, np.uint8).reshape(-1, h, w)
    keep = [0] + [i for i in range(1, len(f)) if np.any(f[i] != f[i - 1])]
    return f[keep].astype(np.float32) / 255


PARITY_VECTORS, PARITY_CONF_MAE = 0.999, 0.01  # share of identical vectors, mean |conf error|


def parity(rec_dir, every=7):
    """Does the port reproduce the GPU's checked /8 fields (vectors and confidence,
    both directions) from the lumas of GSFD samples?"""
    files = sorted(Path(rec_dir).glob("fg_*.bin"))[::every]
    if not files:
        sys.exit(f"no fg_*.bin samples under {rec_dir}")
    match, mae = [], []
    for f in files:
        lp, lc, gpu_fwd, gpu_rev = T.load_sample(f)
        Fw, Rv = field_from_base(lp, lc, 8), field_from_base(lc, lp, 8)
        for port, gpu in ((fbcheck(Fw, Rv, FB_TOL), gpu_fwd), (fbcheck(Rv, Fw, FB_TOL), gpu_rev)):
            gpu = gpu.astype(np.float32)[..., :3]
            if not (np.isfinite(port[..., :3]).all() and np.isfinite(gpu).all()):
                print(f"parity: non-finite field in {f.name}")
                return 1
            match.append(float((np.abs(port[..., :2] - gpu[..., :2]).max(-1) < 0.02).mean()))
            mae.append(float(np.abs(port[..., 2] - gpu[..., 2]).mean()))
    m, e = float(np.mean(match)), float(np.max(mae))
    print(f"parity over {len(files)} GSFD pairs, both directions: vectors {m * 100:.3f}% identical, "
          f"worst confidence MAE {e:.5f} (limits {PARITY_VECTORS * 100:.1f}%, {PARITY_CONF_MAE})")
    return 0 if m >= PARITY_VECTORS and e <= PARITY_CONF_MAE else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help="capture names (NAME-WxH.gray)")
    ap.add_argument("--size", help="WxH")
    ap.add_argument("--parity", type=Path, metavar="DIR", help="check the port against GSFD samples in DIR")
    ap.add_argument("--captures", type=Path, default=Path.home() / ".cache/gamescope-framegen-captures")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--arms", default="checked/8,checked/4")
    ap.add_argument("--net", type=Path, help="GSFR weights for learned arms")
    ap.add_argument("--domain", choices=("coded", "linear"), default="coded",
                    help="matching domain; coded is production for 8/10-bit content")
    args = ap.parse_args()
    if args.parity:
        return parity(args.parity)
    if not args.names or not args.size:
        ap.error("NAME... and --size are required without --parity")
    w, h = (int(v) for v in args.size.split("x"))
    arms = args.arms.split(",")
    net = NE.load_blob(args.net) if args.net else None
    rng = np.random.default_rng(1)
    print(f"{args.size} {args.domain}: paired change vs {arms[0]} [95% CI] | sta (drag) | lost edges (smear) | false edges (tearing)")
    for name in args.names:
        f = distinct_frames(args.captures / f"{name}-{args.size}.gray", w, h)
        rows = [r for r in (triplet(f[i - 1], f[i], f[i + 1], arms, net, args.domain)
                            for i in range(1, len(f) - 1, args.stride)) if r]
        if not rows:
            print(f"{name}: no triplet with motion")
            continue
        for method in ("predictive", "bidir"):
            ref = np.array([r[(method, arms[0])] for r in rows])
            line = f"{name:22s} {method:10s} n={len(rows):2d}"
            for arm in arms:
                v = np.array([r[(method, arm)] for r in rows])
                d = v[:, 0] / ref[:, 0] - 1
                boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
                lo, hi = np.percentile(boot, [2.5, 97.5]) * 100
                m = v.mean(0)
                line += (f" | {arm}: {d.mean() * 100:+.1f}% [{lo:+.1f},{hi:+.1f}] sta {m[1]:.4f}"
                         f" lost {m[2]:.4f} false {m[3]:.4f}")
            print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
