"""Dummy-tensor inference benchmark and PCA/entropy effective-rank helper."""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
from torchvision.ops.misc import Conv2dNormActivation

# Tuple format: (in_c, kernel, exp_c, out_c, use_se, act, stride)
ERM_TRUNK_SPECS = {
    "Ch48": (48, [
        (16, 3, 48, 24, False, "RE", 2),
        (24, 5, 72, 32, True, "HS", 1),
        (32, 5, 96, 48, True, "HS", 2),
        (48, 5, 96, 48, True, "HS", 1),
    ]),
    "Ch32": (32, [
        (16, 3, 48, 24, False, "RE", 2),
        (24, 5, 48, 24, True, "HS", 1),
        (24, 5, 72, 32, True, "HS", 2),
        (32, 5, 64, 32, True, "HS", 1),
    ]),
    "Ch24": (24, [
        (16, 3, 32, 16, False, "RE", 2),
        (16, 5, 32, 24, True, "HS", 1),
        (24, 5, 48, 24, True, "HS", 2),
        (24, 5, 48, 24, True, "HS", 1),
    ]),
}


# Multi-distance training duplicates the stem (Stages 0-4 for Ref. 576, 0-1 for ERM) per distance and shares the trunk/head.
def build_ref576() -> nn.Module:
    """Single-distance deployment Ref. 576 (Modified MobileNetV3-Small, 13 blocks)."""
    base = mobilenet_v3_small(weights=None)
    c0 = base.features[0][0]
    base.features[0][0] = nn.Conv2d(
        1, c0.out_channels, kernel_size=c0.kernel_size, stride=c0.stride, padding=c0.padding, bias=False
    )
    return nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(576, 6))


def build_erm(arch: str) -> nn.Module:
    """Single-distance deployment ERM network (Ch24 / Ch32 / Ch48, 6 blocks)."""
    last_ch, trunk_specs = ERM_TRUNK_SPECS[arch]
    stem = [
        Conv2dNormActivation(1, 16, kernel_size=3, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish),
        InvertedResidual(InvertedResidualConfig(16, 3, 24, 16, False, "RE", 2, 1, 1.0), nn.BatchNorm2d),
    ]
    trunk = [
        InvertedResidual(InvertedResidualConfig(ic, k, ec, oc, se, act, s, 1, 1.0), nn.BatchNorm2d)
        for ic, k, ec, oc, se, act, s in trunk_specs
    ]
    return nn.Sequential(*stem, *trunk, nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(last_ch, 6))


def effective_rank(features: np.ndarray, active_thresh: float = 1e-3) -> dict:
    """Compute PCA effective rank (Roy & Vetterli) and RankMe on (N, D) features."""
    N, D = features.shape
    X_c = features - features.mean(axis=0, keepdims=True)
    C = (X_c.T @ X_c) / (N - 1)
    eig = np.maximum(np.linalg.eigvalsh(C)[::-1], 0.0)
    p = eig / (eig.sum() + 1e-15)
    p_nz = p[p > 1e-15]

    H = float(-np.sum(p_nz * np.log(p_nz)))
    r_eff = float(np.exp(H))
    eta_info_pct = float(H / np.log(D) * 100.0)
    pc_90 = int(np.searchsorted(np.cumsum(p), 0.90) + 1)
    active_pct = float((np.abs(features) > active_thresh).mean() * 100.0)

    G = (features.T @ features) / N
    eig_raw = np.maximum(np.linalg.eigvalsh(G)[::-1], 0.0)
    p_raw = eig_raw / (eig_raw.sum() + 1e-15)
    p_raw_nz = p_raw[p_raw > 1e-15]
    rankme = float(np.exp(-np.sum(p_raw_nz * np.log(p_raw_nz))))

    return {
        "r_eff": r_eff,
        "eta_info_pct": eta_info_pct,
        "pc_90": pc_90,
        "active_pct": active_pct,
        "rankme": rankme,
    }


def benchmark_model(model: nn.Module, x: torch.Tensor, warmup: int, runs: int, device: str):
    """Measure single-frame inference latency in PyTorch eager mode."""
    model = model.eval().to(device)
    is_cuda = device.startswith("cuda")
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    mean_ms = float(arr.mean())
    median_ms = float(np.median(arr))
    hz = 1000.0 / mean_ms
    return mean_ms, median_ms, hz


def main():
    parser = argparse.ArgumentParser(description="CBC phase retrieval dummy-tensor benchmark")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rank-demo", action="store_true")
    args = parser.parse_args()

    if args.device == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)

    x = torch.randn(1, 1, 256, 256, device=args.device)
    models = [
        ("Ref. 576", build_ref576()),
        ("Ch48", build_erm("Ch48")),
        ("Ch32", build_erm("Ch32")),
        ("Ch24", build_erm("Ch24")),
    ]

    print(f"Device: {args.device} | Input: {tuple(x.shape)} | Warmup: {args.warmup} | Runs: {args.runs}")
    print(f"{'Model':<10} {'Params':>10} {'Mean (ms)':>11} {'Median (ms)':>13} {'Freq (Hz)':>11}")
    print("-" * 59)
    for name, m in models:
        params = sum(p.numel() for p in m.parameters())
        mean_ms, med_ms, hz = benchmark_model(m, x, args.warmup, args.runs, args.device)
        print(f"{name:<10} {params:>10,} {mean_ms:>11.3f} {med_ms:>13.3f} {hz:>11.1f}")

    if args.rank_demo:
        rng = np.random.default_rng(42)
        latent = rng.normal(size=(2000, 7))
        proj = rng.normal(size=(7, 576))
        synth = latent @ proj + 0.03 * rng.normal(size=(2000, 576))
        stats = effective_rank(synth)
        print(
            f"\nSynthetic rank-7 (2000, 576) -> "
            f"r_eff={stats['r_eff']:.2f}, eta_info={stats['eta_info_pct']:.1f}%, "
            f"pc_90={stats['pc_90']}, rankme={stats['rankme']:.2f}, "
            f"active={stats['active_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
