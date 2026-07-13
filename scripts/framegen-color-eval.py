#!/usr/bin/env python3
"""Evaluate full-resolution held-out Gamescope frame-generation captures.

GAMESCOPE_FRAMEGEN_RECORD_COLOR records real triplets A/B/C while presenting
all three normally. The framegen GPU predicts B from A and C at B's measured
phase, never presents those predictions, and writes paired occlusion strengths
0/0.5/1 beside the exact real B in one GSCF file. This provides genuine
full-colour ground truth without relying on a repeatable second run or
pretending an endpoint is an intermediate frame.

Higher PSNR/SSIM and lower MAE/bad/edge/residual-change are better. Core metrics
grade the captured output code values; EOTF metadata is reported but no guessed
HDR display transform is applied. Optional LPIPS is used only when explicitly
requested and importable; the core evaluator is numpy-only. Capture file writes
perturb presentation cadence, so these are image-quality measurements, not a
live pacing benchmark.

Usage:
  framegen-color-eval.py --data /path/to/gscf-directory
  framegen-color-eval.py --data /path/to/gscf-directory --per-frame
  framegen-color-eval.py --data /path/to/gscf-directory --lpips
"""

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np


MAGIC = 0x46435347  # 'GSCF'
VERSION = 2
HEADER = struct.Struct("<12I8Q4f")


def fourcc(text):
    return sum(ord(ch) << (8 * i) for i, ch in enumerate(text))


ARGB8888 = fourcc("AR24")
XRGB8888 = fourcc("XR24")
ABGR8888 = fourcc("AB24")
XBGR8888 = fourcc("XB24")
ARGB2101010 = fourcc("AR30")
XRGB2101010 = fourcc("XR30")
ABGR2101010 = fourcc("AB30")
XBGR2101010 = fourcc("XB30")
ABGR16161616 = fourcc("AB48")
XBGR16161616 = fourcc("XB48")
ABGR16161616F = fourcc("AB4H")
XBGR16161616F = fourcc("XB4H")
ABGR32323232F = fourcc("AB8F")

FORMAT_BPP = {
    ARGB8888: 4,
    XRGB8888: 4,
    ABGR8888: 4,
    XBGR8888: 4,
    ARGB2101010: 4,
    XRGB2101010: 4,
    ABGR2101010: 4,
    XBGR2101010: 4,
    ABGR16161616: 8,
    XBGR16161616: 8,
    ABGR16161616F: 8,
    XBGR16161616F: 8,
    ABGR32323232F: 16,
}


def fourcc_name(value):
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def decode_plane(raw, width, height, drm_format, bpp):
    expected = width * height * bpp
    if len(raw) != expected:
        raise ValueError(f"plane is {len(raw)} bytes, expected {expected}")

    if drm_format in (ABGR8888, XBGR8888):
        rgba = np.frombuffer(raw, np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(rgba[..., :3], dtype=np.float32) / 255.0
    if drm_format in (ARGB8888, XRGB8888):
        bgra = np.frombuffer(raw, np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(bgra[..., 2::-1], dtype=np.float32) / 255.0

    if drm_format in (ABGR2101010, XBGR2101010, ARGB2101010, XRGB2101010):
        word = np.frombuffer(raw, "<u4").reshape(height, width)
        lo = (word & 0x3FF).astype(np.float32) / 1023.0
        mid = ((word >> 10) & 0x3FF).astype(np.float32) / 1023.0
        hi = ((word >> 20) & 0x3FF).astype(np.float32) / 1023.0
        if drm_format in (ABGR2101010, XBGR2101010):
            return np.stack((lo, mid, hi), axis=-1)
        return np.stack((hi, mid, lo), axis=-1)

    if drm_format in (ABGR16161616, XBGR16161616):
        rgba = np.frombuffer(raw, "<u2").reshape(height, width, 4)
        return np.ascontiguousarray(rgba[..., :3], dtype=np.float32) / 65535.0
    if drm_format in (ABGR16161616F, XBGR16161616F):
        rgba = np.frombuffer(raw, "<f2").reshape(height, width, 4)
        return np.ascontiguousarray(rgba[..., :3], dtype=np.float32)
    if drm_format == ABGR32323232F:
        rgba = np.frombuffer(raw, "<f4").reshape(height, width, 4)
        return np.ascontiguousarray(rgba[..., :3], dtype=np.float32)

    raise ValueError(f"unsupported DRM format {fourcc_name(drm_format)!r} (0x{drm_format:08x})")


def load_sample(path):
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("truncated GSCF header")
    values = HEADER.unpack_from(raw)
    header = values[:12]
    meta = values[12:20]
    phase = values[20]
    magic, version, width, height, drm_format, bpp, eotf, flags, planes = header[:9]
    if magic != MAGIC or version not in (1, VERSION):
        raise ValueError(f"not a supported GSCF v1/v{VERSION} capture")
    expected_planes = 2 if version == 1 else 4
    expected_flags = 1 if version == 1 else 3
    if (flags & expected_flags) != expected_flags or planes != expected_planes:
        raise ValueError("capture has no supported exact held-out reference layout")
    if width == 0 or height == 0 or bpp != FORMAT_BPP.get(drm_format):
        raise ValueError("capture has invalid dimensions, format, or bytes-per-pixel")
    if not math.isfinite(phase) or not 0.0 < phase < 1.0:
        raise ValueError("capture has an invalid held-out phase")
    if not (meta[1] < meta[2] < meta[3] and meta[4] < meta[5] < meta[6]):
        raise ValueError("capture has non-monotonic frame IDs or timestamps")
    timestamp_phase = (meta[5] - meta[4]) / (meta[6] - meta[4])
    if not math.isclose(phase, timestamp_phase, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("capture phase disagrees with its timestamps")
    if version == VERSION:
        strengths = values[21:24]
        if (not all(math.isfinite(value) and 0.0 <= value <= 1.0
                    for value in strengths)
                or not strengths[0] < strengths[1] < strengths[2]):
            raise ValueError("capture has invalid paired candidate strengths")
    plane_bytes = width * height * bpp
    expected = HEADER.size + planes * plane_bytes
    if len(raw) != expected:
        raise ValueError(f"file is {len(raw)} bytes, expected {expected}")
    decoded = []
    for i in range(planes):
        begin = HEADER.size + i * plane_bytes
        decoded.append(decode_plane(raw[begin:begin + plane_bytes],
                                    width, height, drm_format, bpp))
    if version == 1:
        candidates = [(None, decoded[0])]
    else:
        candidates = list(zip(values[21:24], decoded[:3]))
    reference = decoded[-1]
    return {
        "path": path,
        "width": width,
        "height": height,
        "format": drm_format,
        "eotf": eotf,
        "phase": phase,
        "sequence": meta[0],
        "anchor_id": meta[1],
        "reference_id": meta[2],
        "endpoint_id": meta[3],
        "anchor_ns": meta[4],
        "reference_ns": meta[5],
        "endpoint_ns": meta[6],
        "candidates": candidates,
        "reference": reference,
    }


def box_mean(image, radius):
    """Edge-padded box mean using an integral image, with bounded memory."""
    padded = np.pad(image, radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64)
    size = 2 * radius + 1
    total = (integral[size:, size:] - integral[:-size, size:]
             - integral[size:, :-size] + integral[:-size, :-size])
    return (total / float(size * size)).astype(np.float32)


def ssim(a, b, radius=3):
    dynamic_range = max(float(np.percentile(b, 99.9) - np.percentile(b, 0.1)), 1e-3)
    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    ma, mb = box_mean(a, radius), box_mean(b, radius)
    va = np.maximum(box_mean(a * a, radius) - ma * ma, 0.0)
    vb = np.maximum(box_mean(b * b, radius) - mb * mb, 0.0)
    vab = box_mean(a * b, radius) - ma * mb
    score = ((2.0 * ma * mb + c1) * (2.0 * vab + c2)) / (
        (ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(np.clip(score, -1.0, 1.0).mean())


def luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def grade(sample, generated):
    reference = sample["reference"]
    finite = np.isfinite(generated).all(axis=-1) & np.isfinite(reference).all(axis=-1)
    if not finite.all():
        generated = np.where(np.isfinite(generated), generated, 0.0)
        reference = np.where(np.isfinite(reference), reference, 0.0)

    # scRGB can exceed one or go slightly negative. A robust reference scale
    # keeps thresholds meaningful without clipping HDR highlights before grading.
    scale = max(float(np.percentile(np.abs(reference), 99.9)), 1.0)
    delta = np.abs(generated - reference) / scale
    pixel_error = delta.max(axis=-1)
    mse = float(np.mean(((generated - reference) / scale) ** 2))
    psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(mse)

    gen_luma = luma(generated) / scale
    ref_luma = luma(reference) / scale
    grad_g = np.hypot(*np.gradient(gen_luma))
    grad_r = np.hypot(*np.gradient(ref_luma))
    edge = float(np.mean(np.abs(grad_g - grad_r)) / max(float(grad_r.mean()), 1e-4))
    return {
        "mae": float(delta.mean()),
        "bad%": float((pixel_error > 0.05).mean() * 100.0),
        "p95": float(np.percentile(pixel_error, 95.0)),
        "psnr": psnr,
        "ssim": ssim(gen_luma, ref_luma),
        "edge": edge,
        "nonfinite%": float((~finite).mean() * 100.0),
    }


def temporal_residual_change(samples, candidate_index, block_rows=128):
    """Mean screen-space change in prediction error over contiguous probes."""
    total = 0.0
    values = 0
    transitions = 0
    for previous, current in zip(samples, samples[1:]):
        # A successful continuous sequence closes A/B/C, then reuses C as the
        # next A. Do not call a skipped/discontinuous capture temporal data.
        if (current["anchor_id"] != previous["endpoint_id"]
                or current["reference_ns"] <= previous["reference_ns"]):
            continue
        previous_reference = previous["reference"]
        current_reference = current["reference"]
        scale = max(
            float(np.percentile(np.abs(previous_reference), 99.9)),
            float(np.percentile(np.abs(current_reference), 99.9)),
            1.0,
        )
        previous_generated = previous["candidates"][candidate_index][1]
        current_generated = current["candidates"][candidate_index][1]
        for first_row in range(0, current["height"], block_rows):
            rows = slice(first_row, min(first_row + block_rows, current["height"]))
            previous_error = (previous_generated[rows] - previous_reference[rows]) / scale
            current_error = (current_generated[rows] - current_reference[rows]) / scale
            delta = np.abs(current_error - previous_error)
            total += float(delta.sum(dtype=np.float64))
            values += delta.size
        transitions += 1
    return (total / values if values else None), transitions


def lpips_scores(samples, candidate_index):
    try:
        import lpips
        import torch
    except ImportError as exc:
        print(f"LPIPS unavailable ({exc}); install torch + lpips or omit --lpips",
              file=sys.stderr)
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex").to(device).eval()
    scores = []
    with torch.no_grad():
        for sample in samples:
            scale = max(float(np.percentile(np.abs(sample["reference"]), 99.9)), 1.0)
            tensors = []
            for rgb in (sample["candidates"][candidate_index][1], sample["reference"]):
                rgb = np.clip(rgb / scale, 0.0, 1.0)
                tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)[None]).float().to(device) * 2.0 - 1.0)
            scores.append(float(model(tensors[0], tensors[1]).item()))
    return scores


def format_row(name, metrics):
    return (f"{name}: mae={metrics['mae']:.6f} bad={metrics['bad%']:.3f}% "
            f"p95={metrics['p95']:.5f} psnr={metrics['psnr']:.3f}dB "
            f"ssim={metrics['ssim']:.5f} edge={metrics['edge']:.5f} "
            f"nonfinite={metrics['nonfinite%']:.4f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="directory containing color_*.gscf")
    parser.add_argument("--per-frame", action="store_true", help="print every sample")
    parser.add_argument("--lpips", action="store_true", help="also run optional torch/lpips")
    args = parser.parse_args()

    files = sorted(Path(args.data).glob("color_*.gscf"))
    if not files:
        sys.exit(f"no color_*.gscf captures under {args.data}")
    samples = []
    for path in files:
        try:
            samples.append(load_sample(path))
        except ValueError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
    if not samples:
        sys.exit("no valid GSCF samples")

    shape_keys = {(s["width"], s["height"], s["format"], s["eotf"]) for s in samples}
    if len(shape_keys) != 1:
        sys.exit("capture mixes dimensions/formats/EOTFs; evaluate each homogeneous set separately")

    candidate_strengths = [candidate[0] for candidate in samples[0]["candidates"]]
    if any([candidate[0] for candidate in sample["candidates"]] != candidate_strengths
           for sample in samples):
        sys.exit("capture mixes candidate layouts")
    rows_by_candidate = []
    for candidate_index in range(len(candidate_strengths)):
        rows_by_candidate.append([
            grade(sample, sample["candidates"][candidate_index][1]) for sample in samples
        ])
    if args.per_frame:
        for sample_index, sample in enumerate(samples):
            for candidate_index, strength in enumerate(candidate_strengths):
                label = "configured" if strength is None else f"occlusion={strength:g}"
                row = rows_by_candidate[candidate_index][sample_index]
                print(f"{sample['path'].name} phase={sample['phase']:.4f} {label} "
                      + format_row("", row).lstrip(": "))

    width, height, drm_format, eotf = next(iter(shape_keys))
    phases = np.asarray([sample["phase"] for sample in samples], np.float64)
    print(f"loaded {len(samples)} exact held-out pairs at {width}x{height}, "
          f"format={fourcc_name(drm_format)!r}, EOTF={eotf}, "
          f"phase mean/range={phases.mean():.4f}/{phases.min():.4f}-{phases.max():.4f}")
    for candidate_index, strength in enumerate(candidate_strengths):
        rows = rows_by_candidate[candidate_index]
        summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        summary["sample_mae_sigma"] = float(np.std([row["mae"] for row in rows]))
        label = "configured" if strength is None else f"occlusion={strength:g}"
        print(format_row(label, summary))
        residual_change, transitions = temporal_residual_change(samples, candidate_index)
        residual_text = "n/a" if residual_change is None else f"{residual_change:.6f}"
        print(f"  sample MAE sigma={summary['sample_mae_sigma']:.6f}; "
              f"temporal residual-change MAE={residual_text} ({transitions} transitions)")
        if candidate_index > 0:
            deltas = np.asarray([
                row["mae"] - baseline["mae"]
                for row, baseline in zip(rows, rows_by_candidate[0])
            ])
            print(f"  paired MAE delta vs first candidate={deltas.mean():+.6f}; "
                  f"wins/ties/losses={(deltas < 0).sum()}/{(deltas == 0).sum()}/{(deltas > 0).sum()}")

    if args.lpips:
        for candidate_index, strength in enumerate(candidate_strengths):
            scores = lpips_scores(samples, candidate_index)
            if scores is not None:
                label = "configured" if strength is None else f"occlusion={strength:g}"
                print(f"{label}: lpips={np.mean(scores):.6f} "
                      f"sample_sigma={np.std(scores):.6f}")


if __name__ == "__main__":
    main()
