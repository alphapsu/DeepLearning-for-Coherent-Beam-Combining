"""
7-Beam Coherent Beam Combining (CBC) Phase Retrieval:
Standalone Model Structure, Theoretical Analysis, and CPU Benchmark Demo.
"""

import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
from torchvision.ops.misc import Conv2dNormActivation


def cyclical_phase_loss(pred_rad: torch.Tensor, target_rad: torch.Tensor) -> torch.Tensor:
    """2*pi 주기를 반영한 주기 코사인 손실: L = mean(2 - 2 cos(Delta phi))."""
    return torch.mean(2.0 - 2.0 * torch.cos(target_rad - pred_rad))


def compute_phase_mae_deg_and_strehl(pred_rad: torch.Tensor, target_rad: torch.Tensor):
    """래핑된 위상 MAE(도) 및 7-빔 광학 스트렐 비 계산 (기준 빔 b=0 오차=0)."""
    diff = (target_rad - pred_rad + math.pi) % (2.0 * math.pi) - math.pi
    mae_deg = torch.mean(torch.abs(diff)).item() * (180.0 / math.pi)
    zeros = torch.zeros((pred_rad.shape[0], 1), device=pred_rad.device, dtype=pred_rad.dtype)
    err = torch.cat([zeros, target_rad - pred_rad], dim=1)
    strehl = (torch.cos(err).sum(dim=1) ** 2 + torch.sin(err).sum(dim=1) ** 2) / 49.0
    return mae_deg, strehl.mean().item()


class SharedTrunkReferenceNet(nn.Module):
    """원본 기준 모델(Ref. 576): 초기 5단 거리별 입력 + 후반 8단 공유."""

    def __init__(self, distance_keys=("d5", "d25", "d50", "d75", "d100"), split_k: int = 5):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        first_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            1, first_conv.out_channels,
            kernel_size=first_conv.kernel_size, stride=first_conv.stride,
            padding=first_conv.padding, bias=False,
        )
        self.stems = nn.ModuleDict({
            k: copy.deepcopy(nn.Sequential(*list(base.features[:split_k])))
            for k in distance_keys
        })
        self.shared_trunk = nn.Sequential(*list(base.features[split_k:]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(576, 6)

    def forward(self, x: torch.Tensor, dist_key: str):
        gap = torch.flatten(self.avgpool(self.shared_trunk(self.stems[dist_key](x))), 1)
        return self.head(gap), gap


ERM_CONFIGS = {
    "Ch24": (24, [(16, 3, 32, 16, False, "RE", 2), (16, 5, 32, 24, True, "HS", 1),
                    (24, 5, 48, 24, True, "HS", 2), (24, 5, 48, 24, True, "HS", 1)]),
    "Ch32": (32, [(16, 3, 48, 24, False, "RE", 2), (24, 5, 48, 24, True, "HS", 1),
                    (24, 5, 72, 32, True, "HS", 2), (32, 5, 64, 32, True, "HS", 1)]),
    "Ch48": (48, [(16, 3, 48, 24, False, "RE", 2), (24, 5, 72, 32, True, "HS", 1),
                    (32, 5, 96, 48, True, "HS", 2), (48, 5, 96, 48, True, "HS", 1)]),
}


class EffectiveRankNet(nn.Module):
    """Compact 6-block 1/16 (16x16) phase retrieval network (Ch24 / Ch32 / Ch48)."""

    def __init__(self, distance_keys=("d5", "d25", "d50", "d75", "d100"), arch="Ch32", teacher_dim=576):
        super().__init__()
        last_ch, trunk_specs = ERM_CONFIGS[arch]
        self.stems = nn.ModuleDict({
            k: nn.Sequential(
                Conv2dNormActivation(1, 16, kernel_size=3, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish),
                InvertedResidual(InvertedResidualConfig(16, 3, 24, 16, False, "RE", 2, 1, 1.0), nn.BatchNorm2d),
            ) for k in distance_keys
        })
        self.shared_trunk = nn.Sequential(*[
            InvertedResidual(InvertedResidualConfig(ic, k, ec, oc, se, act, s, 1, 1.0), nn.BatchNorm2d)
            for ic, k, ec, oc, se, act, s in trunk_specs
        ])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(last_ch, 6)
        self.feat_proj = nn.Linear(last_ch, teacher_dim)

    def forward(self, x: torch.Tensor, dist_key: str, return_features: bool = False):
        gap = torch.flatten(self.avgpool(self.shared_trunk(self.stems[dist_key](x))), 1)
        pred = self.head(gap)
        if return_features:
            return pred, gap, self.feat_proj(gap)
        return pred


def analyze_effective_rank_and_pca(features: np.ndarray, active_thresh: float = 1e-3):
    """Compute Shannon-entropy effective rank (r_eff), RankMe, info efficiency (%), and dead filters."""
    N, D = features.shape
    X_c = features - features.mean(axis=0, keepdims=True)
    eigvals = np.maximum(np.linalg.eigvalsh((X_c.T @ X_c) / (N - 1))[::-1], 0.0)
    p = eigvals / (eigvals.sum() + 1e-15)
    p_nz = p[p > 1e-15]
    h_ent = float(-np.sum(p_nz * np.log(p_nz)))

    gram = (features.T @ features) / N
    eig_raw = np.maximum(np.linalg.eigvalsh(gram)[::-1], 0.0)
    p_raw = eig_raw / (eig_raw.sum() + 1e-15)
    p_raw_nz = p_raw[p_raw > 1e-15]

    active_frac = (np.abs(features) > active_thresh).mean(axis=0)
    return {
        "r_eff": float(np.exp(h_ent)),
        "info_efficiency_pct": float(h_ent / np.log(D) * 100.0),
        "rankme": float(np.exp(-np.sum(p_raw_nz * np.log(p_raw_nz)))),
        "pc_for_90pct": int(np.searchsorted(np.cumsum(p) * 100.0, 90.0) + 1),
        "mean_active_pct": float(active_frac.mean() * 100.0),
    }


def count_single_deploy_params(model: nn.Module, dist_key: str = "d50") -> int:
    """단일 거리 배포 기준 파라미터 수 (1개 stem + shared_trunk + head)."""
    stem_p = sum(p.numel() for p in model.stems[dist_key].parameters())
    trunk_p = sum(p.numel() for p in model.shared_trunk.parameters())
    head_p = sum(p.numel() for p in model.head.parameters())
    return stem_p + trunk_p + head_p


def main():
    print("=== 1. Parameter Comparison (Single-Distance Deployment) ===")
    ref = SharedTrunkReferenceNet().eval()
    print(f"Ref. 576 (13-block): {count_single_deploy_params(ref):,} params")
    for arch in ("Ch48", "Ch32", "Ch24"):
        m = EffectiveRankNet(arch=arch).eval()
        print(f"{arch} (6-block): {count_single_deploy_params(m):,} params")

    print("\n=== 2. Effective Rank (r_eff) Demo (Synthetic) ===")
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(2000, 7))
    proj = rng.normal(size=(7, 576))
    synth_gap = latent @ proj + 0.03 * rng.normal(size=(2000, 576))
    print("Low-rank 576-D feature diagnostics:", analyze_effective_rank_and_pca(synth_gap))

    print("\n=== 3. Single-Frame PyTorch CPU Latency ===")
    ch32 = EffectiveRankNet(arch="Ch32").eval()
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        for _ in range(15):
            _ = ch32(x, "d50")
        t0 = time.perf_counter()
        for _ in range(60):
            _ = ch32(x, "d50")
        ms = (time.perf_counter() - t0) / 60 * 1000.0
    print(f"Ch32 PyTorch CPU avg: {ms:.3f} ms ({1000.0 / ms:.1f} Hz)")


if __name__ == "__main__":
    main()
