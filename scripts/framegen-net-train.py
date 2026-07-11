#!/usr/bin/env python3
"""Offline trainer for gamescope's learned motion-field refiner (Stage C).

Consumes the field-resolution tensors captured by GAMESCOPE_FRAMEGEN_RECORD
(one 'GSFD' file per real frame: both low-res lumas + both checked motion
fields, raw — pre-refinement, pre-trust) and trains the tiny convolutional
net cs_framegen_motion_net.comp runs, exporting a 'GSFR' weights blob for
GAMESCOPE_FRAMEGEN_NET.

The objective is the same self-supervision Stage B4 grades with: the field
maps the source frame onto the destination frame, so the destination IS the
ground truth. The offline objective predicts a bounded flow residual (2*tanh,
+-2 field texels) and an additive confidence recalibration; output four stays
at its zero-neutral causal-shading default because GSFD v1 has no third-frame
tensor. The loss warps the source
luma along the refined flow and charges each texel either the warp error
(weighted by the refined confidence) or the crossfade-fallback error
(weighted by the complement) — teaching both heads jointly: fix the flow
where fixable, lower the confidence where not, raise it where the warp is
demonstrably right. Both field directions train the same weights (the
reverse field is the same problem with the lumas swapped), exactly matching
the two binding-swapped dispatches at inference.

Feature construction here MUST stay in lockstep with the shader —
cs_framegen_motion_net.comp documents the 12 channels.

numpy only, CPU only, by design: the net is ~4.6k parameters and trains in
minutes; the GPUs stay free for the game (and the deployment target's
framegen card is deliberately the weaker one).

Usage:
  framegen-net-train.py --data /path/to/capture/dir --out weights.bin
  framegen-net-train.py --init --out neutral.bin   # untrained blob = Stage B
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC_SAMPLE = 0x44465347  # 'GSFD'
MAGIC_BLOB = 0x52465347    # 'GSFR'
BLOB_VERSION = 3           # + zero-neutral causal shading-focus output
LAYERS = ((12, 16), (16, 16), (16, 4))
KSIZE = 3
CHARB_EPS = 1e-3
FLOW_AUX = 0.05     # keep a flow-learning signal where confidence was killed
LAMBDA_DF = 0.01    # pull the flow residual toward Stage B
LAMBDA_TV = 0.005   # smoothness of the refined flow (flow-boundary jaggies)
LAMBDA_DC = 0.005   # keep the confidence correction gentle


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def load_sample(path):
    raw = path.read_bytes()
    hdr = struct.unpack_from("<8I", raw, 0)
    if hdr[0] != MAGIC_SAMPLE or hdr[1] != 1:
        raise ValueError(f"{path}: not a GSFD v1 sample")
    w, h, luma_bpp, field_bpp, flags = hdr[2], hdr[3], hdr[4], hdr[5], hdr[6]
    if field_bpp != 8 or not (flags & 1):
        raise ValueError(f"{path}: unsupported layout")
    off = 8 * 4 + 8  # header + seqno

    def luma_plane():
        nonlocal off
        n = w * h * luma_bpp
        plane = np.frombuffer(raw, dtype=np.float16, count=n // 2, offset=off)
        off += n
        if luma_bpp == 8:  # rgba16f luma texture: luma sits in .r
            plane = plane.reshape(h, w, 4)[..., 0]
        return np.ascontiguousarray(plane.reshape(h, w), dtype=np.float32)

    def field_plane():
        nonlocal off
        n = w * h * 8
        plane = np.frombuffer(raw, dtype=np.float16, count=n // 2, offset=off)
        off += n
        return np.ascontiguousarray(plane.reshape(h, w, 4))

    luma_prev, luma_cur = luma_plane(), luma_plane()
    field_fwd, field_rev = field_plane(), field_plane()
    return luma_prev, luma_cur, field_fwd, field_rev


def load_dataset(data_dir):
    files = sorted(Path(data_dir).glob("fg_*.bin"))
    if not files:
        sys.exit(f"no fg_*.bin samples under {data_dir}")
    samples = []
    for f in files:
        try:
            samples.append(load_sample(f))
        except ValueError as e:
            print(f"skipping {e}", file=sys.stderr)
    print(f"loaded {len(samples)} samples "
          f"({samples[0][0].shape[1]}x{samples[0][0].shape[0]} field res)")
    return samples


# --------------------------------------------------------------------------
# Bilinear sampling (GL semantics: clamp-to-edge, half-texel centers)
# --------------------------------------------------------------------------

def bilinear(img, cy, cx):
    """Sample img at continuous texel coords (cy, cx); returns value and the
    analytic gradient (d/dcy, d/dcx). Matches texture() with a normalized
    clamp-to-edge bilinear sampler."""
    h, w = img.shape
    fy = cy - 0.5
    fx = cx - 0.5
    y0 = np.floor(fy).astype(np.int64)
    x0 = np.floor(fx).astype(np.int64)
    wy = (fy - y0).astype(np.float32)
    wx = (fx - x0).astype(np.float32)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    v00 = img[y0c, x0c]
    v10 = img[y0c, x1c]
    v01 = img[y1c, x0c]
    v11 = img[y1c, x1c]
    top = v00 + wx * (v10 - v00)
    bot = v01 + wx * (v11 - v01)
    val = top + wy * (bot - top)
    dvdx = (1.0 - wy) * (v10 - v00) + wy * (v11 - v01)
    dvdy = bot - top
    return val, dvdy, dvdx


# --------------------------------------------------------------------------
# Features (channel-for-channel the shader's phase A)
# --------------------------------------------------------------------------

def features(X, Y, luma_src, luma_dst, y0, x0, size, mode=0.0):
    """12-channel feature crop of `size` starting at (y0, x0)."""
    ys, xs = slice(y0, y0 + size), slice(x0, x0 + size)
    Xc = X[ys, xs].astype(np.float32)
    Yc = Y[ys, xs].astype(np.float32)
    ld = luma_dst[ys, xs].astype(np.float32)
    py, px = np.mgrid[y0:y0 + size, x0:x0 + size].astype(np.float32)
    lw, _, _ = bilinear(luma_src, py + 0.5 - Xc[..., 1], px + 0.5 - Xc[..., 0])
    resid = np.abs(ld - lw) / np.maximum(1.0, np.maximum(np.abs(ld), np.abs(lw)))
    f = np.empty((12, size, size), np.float32)
    f[0] = Xc[..., 0] * 0.125
    f[1] = Xc[..., 1] * 0.125
    f[2] = Xc[..., 2]
    f[3] = np.minimum(Xc[..., 3] * 0.5, 1.0)
    f[4] = Yc[..., 0] * 0.125
    f[5] = Yc[..., 1] * 0.125
    f[6] = Yc[..., 2]
    f[7] = np.minimum(Yc[..., 3] * 0.5, 1.0)
    f[8] = ld
    f[9] = lw
    f[10] = resid
    f[11] = mode
    return f


# --------------------------------------------------------------------------
# The net: three valid 3x3 convs (matches the shader's fused kernel over a
# padded tile; training crops sit in the interior so no padding is needed)
# --------------------------------------------------------------------------

def conv_fwd(x, W, b):
    win = np.lib.stride_tricks.sliding_window_view(x, (KSIZE, KSIZE), axis=(2, 3))
    return np.einsum("bihwkl,oikl->bohw", win, W, optimize=True) + b[None, :, None, None]


def conv_bwd(x, W, dout):
    win = np.lib.stride_tricks.sliding_window_view(x, (KSIZE, KSIZE), axis=(2, 3))
    dW = np.einsum("bihwkl,bohw->oikl", win, dout, optimize=True)
    db = dout.sum(axis=(0, 2, 3))
    dpad = np.pad(dout, ((0, 0), (0, 0), (2, 2), (2, 2)))
    dwin = np.lib.stride_tricks.sliding_window_view(dpad, (KSIZE, KSIZE), axis=(2, 3))
    dx = np.einsum("bohwkl,oikl->bihw", dwin, W[:, :, ::-1, ::-1], optimize=True)
    return dx, dW, db


def net_fwd(params, x):
    a1 = np.maximum(conv_fwd(x, params["W1"], params["b1"]), 0.0)
    a2 = np.maximum(conv_fwd(a1, params["W2"], params["b2"]), 0.0)
    o = conv_fwd(a2, params["W3"], params["b3"])
    return o, (x, a1, a2)


def net_bwd(params, cache, do):
    x, a1, a2 = cache
    da2, dW3, db3 = conv_bwd(a2, params["W3"], do)
    da2 *= a2 > 0
    da1, dW2, db2 = conv_bwd(a1, params["W2"], da2)
    da1 *= a1 > 0
    _, dW1, db1 = conv_bwd(x, params["W1"], da1)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def init_params(rng):
    p = {}
    for i, (ci, co) in enumerate(LAYERS, 1):
        if i == len(LAYERS):
            # Zero head: the untrained net is exactly Stage B.
            p[f"W{i}"] = np.zeros((co, ci, KSIZE, KSIZE), np.float32)
        else:
            std = np.sqrt(2.0 / (ci * KSIZE * KSIZE))
            p[f"W{i}"] = rng.normal(0.0, std, (co, ci, KSIZE, KSIZE)).astype(np.float32)
        p[f"b{i}"] = np.zeros(co, np.float32)
    return p


# --------------------------------------------------------------------------
# Loss on one batch of views
# --------------------------------------------------------------------------

def charb(x):
    return np.sqrt(x * x + CHARB_EPS * CHARB_EPS)


def batch_loss(params, views, size, train=True):
    """views: list of (X, Y, luma_src, luma_dst, y0, x0) where (y0, x0) is the
    top-left of the (size+6)^2 feature crop; the size^2 loss region sits at
    +3. Returns (metrics, grads or None)."""
    feat = np.stack([features(X, Y, ls, ld, y0, x0, size + 6)
                     for X, Y, ls, ld, y0, x0 in views])
    o, cache = net_fwd(params, feat)

    B = len(views)
    F = np.stack([v[0][v[4] + 3:v[4] + 3 + size, v[5] + 3:v[5] + 3 + size, 0:2]
                  for v in views]).astype(np.float32)          # (B,size,size,2)
    conf = np.stack([v[0][v[4] + 3:v[4] + 3 + size, v[5] + 3:v[5] + 3 + size, 2]
                     for v in views]).astype(np.float32)
    ld = np.stack([v[3][v[4] + 3:v[4] + 3 + size, v[5] + 3:v[5] + 3 + size]
                   for v in views]).astype(np.float32)
    ls_pt = np.stack([v[2][v[4] + 3:v[4] + 3 + size, v[5] + 3:v[5] + 3 + size]
                      for v in views]).astype(np.float32)

    th = np.tanh(o[:, 0:2].transpose(0, 2, 3, 1))              # (B,size,size,2)
    dF = 2.0 * th
    Fh = F + dF
    # Warp the source luma along the refined flow (per view: source differs).
    warped = np.empty_like(ld)
    gy = np.empty_like(ld)
    gx = np.empty_like(ld)
    py, px = np.mgrid[3:3 + size, 3:3 + size].astype(np.float32)
    for i, (X, Y, luma_src, _, y0, x0) in enumerate(views):
        cy = (py + y0) + 0.5 - Fh[i, ..., 1]
        cx = (px + x0) + 0.5 - Fh[i, ..., 0]
        warped[i], gy[i], gx[i] = bilinear(luma_src, cy, cx)

    # Evidence gate on confidence raises (mirrors inference): a bounded
    # correction may recover confidence only when its FINAL vector predicts
    # this frame pair. The gate is intentionally stop-gradient; it is a safety
    # verdict, while FLOW_AUX below supplies the rejected flow's learning path.
    refined_resid = np.abs(warped - ld) / np.maximum(
        1.0, np.maximum(np.abs(warped), np.abs(ld)))
    t = np.clip((refined_resid - 0.03) / 0.07, 0.0, 1.0)
    gate = 1.0 - t * t * (3.0 - 2.0 * t)
    dc_raw = o[:, 2]
    dc_chain = np.where(dc_raw < 0.0, 1.0, gate)
    dc = dc_raw * dc_chain
    ch = np.clip(conf + dc, 0.0, 1.0)

    ew = warped - ld
    rw = charb(ew)
    rfb = charb(ls_pt - ld)
    n = float(B * size * size)

    data = float(np.sum(ch * rw + 0.5 * (1.0 - ch) * rfb
                        + FLOW_AUX * rw) / n)
    reg = (LAMBDA_DF * float(np.sum(dF * dF)) / n
           + LAMBDA_DC * float(np.sum(dc * dc)) / n)
    tvy = Fh[:, 1:, :, :] - Fh[:, :-1, :, :]
    tvx = Fh[:, :, 1:, :] - Fh[:, :, :-1, :]
    reg += LAMBDA_TV * (float(np.sum(tvy * tvy)) + float(np.sum(tvx * tvx))) / n

    metrics = {
        "loss": data + reg, "data": data,
        "neutral": float(np.sum((conf + FLOW_AUX)
                                * charb(bl_neutral(views, size, py, px)[0] - ld)
                                + 0.5 * (1.0 - conf) * rfb) / n),
        "mean_dF": float(np.mean(np.abs(dF))),
        "mean_dc": float(np.mean(dc)),
    }
    if not train:
        return metrics, None

    # Backprop.
    do = np.zeros_like(o)
    dL_dch = (rw - 0.5 * rfb) / n
    live = ((conf + dc) > 0.0) & ((conf + dc) < 1.0)
    do[:, 2] = (dL_dch * live * dc_chain
                + (2.0 * LAMBDA_DC / n) * dc * dc_chain)

    dL_drw = (ch + FLOW_AUX) / n
    dL_dwarp = dL_drw * (ew / rw)
    dL_dFh = np.empty_like(Fh)
    dL_dFh[..., 0] = -dL_dwarp * gx
    dL_dFh[..., 1] = -dL_dwarp * gy
    # TV term.
    tv_g = np.zeros_like(Fh)
    tv_g[:, 1:, :, :] += 2.0 * LAMBDA_TV / n * tvy
    tv_g[:, :-1, :, :] -= 2.0 * LAMBDA_TV / n * tvy
    tv_g[:, :, 1:, :] += 2.0 * LAMBDA_TV / n * tvx
    tv_g[:, :, :-1, :] -= 2.0 * LAMBDA_TV / n * tvx
    dL_dFh += tv_g
    dL_ddF = dL_dFh + (2.0 * LAMBDA_DF / n) * dF
    dL_dth = dL_ddF * 2.0
    do[:, 0:2] = (dL_dth * (1.0 - th * th)).transpose(0, 3, 1, 2)

    return metrics, net_bwd(params, cache, do)


def bl_neutral(views, size, py, px):
    """Warp along the RAW field — the Stage-B baseline the net must beat."""
    out = np.empty((len(views), size, size), np.float32)
    for i, (X, Y, luma_src, _, y0, x0) in enumerate(views):
        Fc = X[y0 + 3:y0 + 3 + size, x0 + 3:x0 + 3 + size, 0:2].astype(np.float32)
        out[i], _, _ = bilinear(luma_src, (py + y0) + 0.5 - Fc[..., 1],
                                (px + x0) + 0.5 - Fc[..., 0])
    return out, None


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def make_views(samples, idx, rng, crop, count):
    views = []
    for _ in range(count):
        lp, lc, ff, fr = samples[idx[rng.integers(len(idx))]]
        h, w = lp.shape
        if h < crop + 6 or w < crop + 6:
            sys.exit("samples smaller than the training crop; lower --crop")
        y0 = int(rng.integers(0, h - crop - 6 + 1))
        x0 = int(rng.integers(0, w - crop - 6 + 1))
        # Forward view: refine F (prev -> cur); reverse view: refine R.
        views.append((ff, fr, lp, lc, y0, x0))
        views.append((fr, ff, lc, lp, y0, x0))
    return views


def export(params, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<3I", MAGIC_BLOB, BLOB_VERSION, len(LAYERS)))
        for ci, co in LAYERS:
            f.write(struct.pack("<3I", ci, co, KSIZE))
        for i in range(1, len(LAYERS) + 1):
            # Shader layout: [c_out][ky][kx][c_in] — input channels contiguous
            # so the kernel reads whole vec4 groups.
            params[f"W{i}"].transpose(0, 2, 3, 1).astype("<f4").tofile(f)
            params[f"b{i}"].astype("<f4").tofile(f)
    total = sum(p.size for p in params.values())
    print(f"exported {total} floats -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="capture directory (GAMESCOPE_FRAMEGEN_RECORD)")
    ap.add_argument("--out", required=True, help="output weights blob")
    ap.add_argument("--init", action="store_true",
                    help="export an untrained (neutral, zero-head) blob and exit")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8, help="crops per step (x2 directions)")
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    params = init_params(rng)
    if args.init:
        export(params, args.out)
        return
    if not args.data:
        sys.exit("--data is required unless --init")

    samples = load_dataset(args.data)
    n_val = max(1, len(samples) // 10)
    val_idx = list(range(len(samples) - n_val, len(samples)))
    train_idx = list(range(len(samples) - n_val))

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    b1, b2, adam_eps = 0.9, 0.999, 1e-8

    def evaluate():
        ev_rng = np.random.default_rng(1234)
        views = make_views(samples, val_idx, ev_rng, args.crop, 4 * args.batch)
        met, _ = batch_loss(params, views, args.crop, train=False)
        return met

    base = evaluate()
    print(f"val before: data={base['data']:.5f} neutral={base['neutral']:.5f}")

    for step in range(1, args.steps + 1):
        views = make_views(samples, train_idx, rng, args.crop, args.batch)
        met, grads = batch_loss(params, views, args.crop)
        lr = args.lr * (0.5 * (1.0 + np.cos(np.pi * step / args.steps)))
        # Global-norm clip: cheap insurance against a bad crop.
        gn = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = min(1.0, 1.0 / (gn + 1e-12))
        for k in params:
            g = grads[k] * scale
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            mh = m[k] / (1 - b1 ** step)
            vh = v[k] / (1 - b2 ** step)
            params[k] -= lr * mh / (np.sqrt(vh) + adam_eps)
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d}  loss={met['loss']:.5f} data={met['data']:.5f} "
                  f"neutral={met['neutral']:.5f} |dF|={met['mean_dF']:.3f} "
                  f"dc={met['mean_dc']:+.3f} lr={lr:.2e}")

    fin = evaluate()
    gain = (base["neutral"] - fin["data"]) / max(base["neutral"], 1e-9) * 100.0
    print(f"val after:  data={fin['data']:.5f} neutral={fin['neutral']:.5f} "
          f"(-{gain:.1f}% vs Stage B)  |dF|={fin['mean_dF']:.3f} dc={fin['mean_dc']:+.3f}")
    export(params, args.out)


if __name__ == "__main__":
    main()
