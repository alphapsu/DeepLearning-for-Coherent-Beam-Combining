# 결맞음 빔 결합(CBC) 비초점면 위상 복원 모델 구조 및 유효 자유도 기반 경량화 기술 가이드

이 문서는 **7-빔 결맞음 빔 결합(Coherent Beam Combining, CBC)** 간섭 무늬 영상으로부터 단일 스텝으로 위상을 복원하는 **원본 기준 모델(13단계)**과 **유효 자유도 기반 경량 모델(ERM, 6단계)**의 레이어 구조를 상세히 설명하고, 모델 분석·압축에 사용된 핵심 이론과 직접 실행해 볼 수 있는 독립 예제 코드를 정리한 한국어 기술 보충 자료입니다.

---

## 1. 문제 정의 및 주기 위상 손실 · 광학 평가 지표

### 1.1 물리적 배경
7-빔 타일드 어퍼처(tiled-aperture) CBC 시스템에서 기준 빔($b=0, \phi_0 = 0$)과 나머지 6개 빔의 상대 위상 $\boldsymbol{\phi} = (\phi_1, \dots, \phi_6) \in (-\pi, \pi]^6$에 따른 축방향 거리 $z$에서의 간섭 세기 분포는 다음과 같습니다:

$$I(\mathbf{r}; \boldsymbol{\phi}, z) = \left| E_0(\mathbf{r}; z) + \sum_{m=1}^{6} E_m(\mathbf{r}; z)\,e^{i\phi_m} \right|^2$$

- **초점면($z = 0\text{ mm}$)의 켤레 위상 모호성:** 이상적 초점면에서는 중심 반전 대칭 $I(\mathbf{r}; -\boldsymbol{\phi}) = I(-\mathbf{r}; +\boldsymbol{\phi})$으로 인해 위상 부호($\pm\boldsymbol{\phi}$)를 구분하는 반대칭 성분이 소실됩니다.
- **비초점면($z > 0$)의 대칭성 깨짐:** 관측 평면을 이동하면 프레넬 회절 이차 위상항 $\exp(ikr^2/2z)$이 반전 대칭을 깨뜨려 단일 영상만으로도 부호 모호성 없는 고정밀 위상 복원이 가능해집니다.

### 1.2 주기 위상 손실($\mathcal{L}_{\text{cos}}$) 및 스트렐 비(Strehl Ratio)
위상 $\phi \in (-\pi, \pi]$는 $2\pi$ 주기를 갖는 원형 변수이므로 일반 MSE 대신 코사인 거리 기반 손실을 사용하며, 결합 효율은 광학 스트렐 비로 평가합니다:

$$\mathcal{L}_{\text{cos}}(\hat{\boldsymbol{\phi}}, \boldsymbol{\phi}) = \frac{1}{6}\sum_{m=1}^{6}\big(2 - 2\cos(\phi_m - \hat{\phi}_m)\big), \qquad \text{Strehl} = \left|\frac{1}{7}\sum_{b=0}^{6} e^{i(\phi_b - \hat{\phi}_b)}\right|^2 \quad (\phi_0 = \hat{\phi}_0 = 0)$$

```python
import math
import torch

def cyclical_phase_loss(pred_rad: torch.Tensor, target_rad: torch.Tensor) -> torch.Tensor:
    """2*pi 위상 주기를 반영한 주기 코사인 손실 (범위: 0 ~ 4)."""
    return torch.mean(2.0 - 2.0 * torch.cos(target_rad - pred_rad))

def compute_phase_mae_deg_and_strehl(pred_rad: torch.Tensor, target_rad: torch.Tensor):
    """위상 오차 MAE(도) 및 7-빔 광학 스트렐 비 계산."""
    diff = (target_rad - pred_rad + math.pi) % (2.0 * math.pi) - math.pi
    mae_deg = torch.mean(torch.abs(diff)).item() * (180.0 / math.pi)

    zeros = torch.zeros((pred_rad.shape[0], 1), device=pred_rad.device, dtype=pred_rad.dtype)
    err = torch.cat([zeros, target_rad - pred_rad], dim=1)  # 기준 빔 b=0 오차는 0
    strehl = (torch.cos(err).sum(dim=1) ** 2 + torch.sin(err).sum(dim=1) ** 2) / 49.0
    return mae_deg, strehl.mean().item()

# 실행 예시 (더미 텐서)
pred = torch.randn(32, 6) * 0.1
target = torch.zeros(32, 6)
print("Loss:", cyclical_phase_loss(pred, target).item())
print("MAE(deg), Strehl:", compute_phase_mae_deg_and_strehl(pred, target))
```

---

## 2. 원본 기준 모델 구조 (`Ref. 576`, 13단계)

원본 기준 모델은 `MobileNetV3-Small`을 1채널 흑백 간섭 무늬 입력(`256×256`)과 6개 상대 위상 직접 회귀(`nn.Linear(576, 6)`)에 맞게 변형한 구조입니다. 다중 거리 학습 시에는 **초기 5단계(`Blocks 0..4`, 전체 파라미터의 약 2.0%)만 거리별로 분리**하고, **후반 8단계(`Blocks 5..12` + GAP + 출력층, 약 98.0%)의 가중치를 모든 거리가 공유**합니다.

### 2.1 레이어별 상세 구조 (`1 × 256 × 256` 입력 기준)

| 단계 (Stage) | 역할 구분 | 연산 블록 유형 | 출력 텐서 크기 `(C, H, W)` | 공간 축소율 | 파라미터 수 | 비고 |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **Stage 0** | 거리별 입력 | `Conv2d(k=3, s=2) + BN + Hardswish` | `(16, 128, 128)` | `1/2` | `176` | 초기 공간 에지/무늬 포착 |
| **Stage 1** | 거리별 입력 | `InvertedResidual(16→16, s=2, SE)` | `(16, 64, 64)` | `1/4` | `744` | 거리별 입력부 내 SE 포함 |
| **Stage 2** | 거리별 입력 | `InvertedResidual(16→24, s=2)` | `(24, 32, 32)` | `1/8` | `3,864` | - |
| **Stage 3** | 거리별 입력 | `InvertedResidual(24→24, s=1)` | `(24, 32, 32)` | `1/8` | `5,416` | - |
| **Stage 4** | 거리별 입력 | `InvertedResidual(24→40, s=2, SE)` | `(40, 16, 16)` | `1/16` | `13,736` | `5`단계 누적 `23,936` (SE 공유 시 고유 `18,928`) |
| **Stage 5** | 공통 추출부 | `InvertedResidual(40→40, s=1, SE)` | `(40, 16, 16)` | `1/16` | `57,264` | `16×16` 동일 해상도 반복 시작 |
| **Stage 6** | 공통 추출부 | `InvertedResidual(40→40, s=1, SE)` | `(40, 16, 16)` | `1/16` | `57,264` | - |
| **Stage 7** | 공통 추출부 | `InvertedResidual(40→48, s=1, SE)` | `(48, 16, 16)` | `1/16` | `21,968` | - |
| **Stage 8** | 공통 추출부 | `InvertedResidual(48→48, s=1, SE)` | `(48, 16, 16)` | `1/16` | `29,800` | - |
| **Stage 9** | 공통 추출부 | `InvertedResidual(48→96, s=2, SE)` | `(96, 8, 8)` | `1/32` | `91,848` | 공간 해상도 `8×8`로 축소 |
| **Stage 10** | 공통 추출부 | `InvertedResidual(96→96, s=1, SE)` | `(96, 8, 8)` | `1/32` | `294,096` | **전체 파라미터의 ~29% 차지** |
| **Stage 11** | 공통 추출부 | `InvertedResidual(96→96, s=1, SE)` | `(96, 8, 8)` | `1/32` | `294,096` | **전체 파라미터의 ~29% 차지** |
| **Stage 12** | 공통 추출부 | `Conv2d(1×1, 96→576) + BN + Hardswish` | `(576, 8, 8)` | `1/32` | `56,448` | `576`개 고차원 특징 맵 확장 |
| **GAP + Head** | 위상 회귀 | `AdaptiveAvgPool2d(1) + Linear(576, 6)` | `(6,)` | 전역 | `3,462` | 6개 상대 위상(rad) 직접 출력 |

> **관찰 포인트:** 기준 모델의 **73% 이상 파라미터가 `Stage 9~12` (`8×8` 해상도, `96→576` 특징 맵 확장 구간)에 집중**되어 있으며, 정작 576차원 요약 벡터의 유효 자유도($r_{\text{eff}}$)는 `7.56`에 불과해 극심한 꼬리부 과잉 설계가 존재합니다.

### 2.2 다중 거리 공유 기준 모델 구현 예시 (PyTorch)

```python
import copy
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class SharedTrunkReferenceNet(nn.Module):
    """원본 기준 모델(Ref. 576): 초기 5단계(Blocks 0..4) 거리별 분리 + 후반 8단계 공유."""
    def __init__(self, distance_keys=("d5", "d25", "d50", "d75", "d100"), split_k: int = 5):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        # 1채널 흑백 간섭 무늬 입력으로 첫 번째 Conv 수정
        first_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            1, first_conv.out_channels,
            kernel_size=first_conv.kernel_size, stride=first_conv.stride,
            padding=first_conv.padding, bias=False,
        )

        # 초기 split_k(=5)개 블록을 각 거리별 입력 어댑터(Stem)로 복제
        self.stems = nn.ModuleDict({
            k: copy.deepcopy(nn.Sequential(*list(base.features[:split_k])))
            for k in distance_keys
        })
        # 후반 8단계(Blocks 5..12) + GAP + 6-위상 회귀 헤드는 전 거리 공유
        self.shared_trunk = nn.Sequential(*list(base.features[split_k:]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(576, 6)

    def forward(self, x: torch.Tensor, dist_key: str):
        h = self.stems[dist_key](x)
        feat = self.shared_trunk(h)
        gap = torch.flatten(self.avgpool(feat), 1)  # (B, 576)
        return self.head(gap), gap
```

---

## 3. 압축 경량 모델 구조 (`ERM-24 / 32 / 48`, 6단계)

유효 자유도 분석 결과($r_{\text{eff}} = 7.56$)를 바탕으로 불필요한 후반부와 과잉 특징 맵을 제거한 **Effective-Rank Matched (`ERM`)** 구조는 다음 3가지 원칙으로 설계되었습니다:

1. **연산 단 대폭 축소 (`13단 → 6단`):** 거리별 초기 입력 `2단` + 공통 위상 추출 `4단`으로 단순화해 동일 해상도 반복 연산 제거.
2. **공간 해상도 보존 (`1/32 → 1/16`):** `8×8` 다운샘플링을 없애고 **`16×16` 공간 격자를 유지**하여 비초점면 빔 사이의 미세 간섭 줄무늬 해상도를 보존.
3. **특징 맵 개수 압축 (`576 → 24 / 32 / 48개`):** 유효 자유도(`7.56`)의 약 `3~6배` 수준으로 특징 차원을 직접 매칭해 비활성 필터 없이 고효율 연산 달성.

### 3.1 `ERM-32` 레이어별 상세 구조 (`1 × 256 × 256` 입력 기준)

| 단계 (Stage) | 역할 구분 | 연산 구성 `(in_c → exp_c → out_c, k, s)` | 출력 크기 `(C, H, W)` | 공간 축소율 | 파라미터 수 (`ERM-32`) |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **Stage 0** | 거리별 입력 | `Conv2d(1→16, k=3, s=2) + BN + Hardswish` | `(16, 128, 128)` | `1/2` | `176` |
| **Stage 1** | 거리별 입력 | `InvertedResidual(16→24→16, k=3, s=2, ReLU)` | `(16, 64, 64)` | `1/4` | `1,112` (거리별 합계 `1,288`) |
| **Stage 2** | 공통 추출부 | `InvertedResidual(16→48→24, k=3, s=2, ReLU)` | `(24, 32, 32)` | `1/8` | `2,592` |
| **Stage 3** | 공통 추출부 | `InvertedResidual(24→48→24, k=5, s=1, SE, HS)` | `(24, 32, 32)` | `1/8` | `5,344` |
| **Stage 4** | 공통 추출부 | `InvertedResidual(24→72→32, k=5, s=2, SE, HS)` | `(32, 16, 16)` | `1/16` | `9,736` |
| **Stage 5** | 공통 추출부 | `InvertedResidual(32→64→32, k=5, s=1, SE, HS)` | `(32, 16, 16)` | `1/16` | `8,144` (`16×16` 해상도 보존) |
| **GAP + Head** | 위상 회귀 | `AdaptiveAvgPool2d(1) + Linear(32, 6)` | `(6,)` | 전역 | `198` (공유 합계 `26,014`) |

- **단일 거리 배포 파라미터:** `1,288 + 26,014 = 27,302` (`27.3K`, 기준 단일 모델 `930.2K` 대비 **-97.1%**)
- **5-거리 통합 파라미터:** `5 × 1,288 + 26,014 = 32,454` (`32.5K`, 기준 통합 모델 `1,005.9K` 대비 **-96.8%**)
- `ERM-24` (`16→24→24` 최종 `24`개) 및 `ERM-48` (`24→32→48` 최종 `48`개)은 `Stage 2~5` 채널 폭만 조정한 동일 6단계 구조입니다.

### 3.2 `ERM` 경량 모델 구현 예시 (PyTorch)

```python
import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
from torchvision.ops.misc import Conv2dNormActivation

ERM_CONFIGS = {
    # (in_c, kernel, exp_c, out_c, use_se, act, stride)
    "ERM-24": (24, [(16, 3, 32, 16, False, "RE", 2), (16, 5, 32, 24, True, "HS", 1),
                    (24, 5, 48, 24, True, "HS", 2),  (24, 5, 48, 24, True, "HS", 1)]),
    "ERM-32": (32, [(16, 3, 48, 24, False, "RE", 2), (24, 5, 48, 24, True, "HS", 1),
                    (24, 5, 72, 32, True, "HS", 2),  (32, 5, 64, 32, True, "HS", 1)]),
    "ERM-48": (48, [(16, 3, 48, 24, False, "RE", 2), (24, 5, 72, 32, True, "HS", 1),
                    (32, 5, 96, 48, True, "HS", 2),  (48, 5, 96, 48, True, "HS", 1)]),
}

class EffectiveRankNet(nn.Module):
    """6단계 1/16 스케일 경량 위상 복원 네트워크 (ERM-24 / 32 / 48)."""
    def __init__(self, distance_keys=("d5", "d25", "d50", "d75", "d100"), arch="ERM-32", teacher_dim=576):
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
        self.feat_proj = nn.Linear(last_ch, teacher_dim)  # 학습(지식 증류) 전용, 배포 시 제거

    def forward(self, x: torch.Tensor, dist_key: str, return_features: bool = False):
        h = self.stems[dist_key](x)
        gap = torch.flatten(self.avgpool(self.shared_trunk(h)), 1)
        pred = self.head(gap)
        if return_features:
            return pred, gap, self.feat_proj(gap)
        return pred
```

---

## 4. 핵심 분석 및 압축 방법론 (이론 + 실행 가능한 예제 코드)

### 4.1 특징 공간 PCA · 폰 노이만 엔트로피 · 유효 자유도 ($r_{\text{eff}}$) 및 정보 효율 ($\eta_{\text{info}}$)

네트워크 마지막 풀링(GAP) 특징 행렬 $\mathbf{X} \in \mathbb{R}^{N \times D}$ ($N$개 샘플, $D$개 특징 맵)의 평균을 제거한 $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{X}}$ 로부터 표본 공분산 행렬을 구합니다:

$$\mathbf{C} = \frac{1}{N - 1}\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} \in \mathbb{R}^{D \times D}, \qquad p_i = \frac{\lambda_i}{\sum_{j=1}^{D}\lambda_j} \quad (\lambda_i: \mathbf{C}\text{의 고유값})$$

- **폰 노이만(스펙트럼) 엔트로피 & 유효 자유도 (Effective Rank, Roy & Vetterli 2007 [4]):**
  $$S_{\text{vN}} = -\sum_{p_i > 0} p_i \ln p_i, \qquad r_{\text{eff}} = \exp(S_{\text{vN}})$$
  특징 차원이 `D=576`이더라도 대부분의 에너지가 소수 고유모드에 몰려 있으면 $r_{\text{eff}} \ll D$가 되며, 반대로 비중심 그람 행렬 $\mathbf{G} = \frac{1}{N}\mathbf{X}^\top\mathbf{X}$를 쓰면 표현 붕괴 진단 지표인 **RankMe (Garrido et al. 2023 [5])**가 됩니다.
- **정보 활용 효율 (Normalized Entropy):** $\eta_{\text{info}} = \frac{S_{\text{vN}}}{\ln D} \times 100\%$ 로, 전체 채널 용량 중 실질적 정보 표현에 쓰이는 비율을 뜻합니다 (`576`차원 `31.8%` → `ERM-32` `57.8%`).

```python
import numpy as np

def analyze_effective_rank_and_pca(features: np.ndarray, active_thresh: float = 1e-3):
    """
    features: shape (N, D) — N개 테스트 샘플에서 추출한 요약(GAP) 특징 행렬
    반환값: 유효 자유도(r_eff), 정보 효율(%), RankMe, PCA 누적 분산, 비활성 필터 수
    """
    N, D = features.shape
    X_centered = features - features.mean(axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / (N - 1)

    eigvals = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0.0)
    p = eigvals / (eigvals.sum() + 1e-15)
    p_nz = p[p > 1e-15]

    s_vn = float(-np.sum(p_nz * np.log(p_nz)))
    r_eff = float(np.exp(s_vn))
    info_efficiency_pct = float(s_vn / np.log(D) * 100.0)
    participation_ratio = float(1.0 / np.sum(p_nz ** 2))
    cum_pca_pct = np.cumsum(p) * 100.0

    # 비중심 RankMe (Garrido et al., ICML 2023)
    gram = (features.T @ features) / N
    eig_raw = np.maximum(np.linalg.eigvalsh(gram)[::-1], 0.0)
    p_raw = eig_raw / (eig_raw.sum() + 1e-15)
    p_raw_nz = p_raw[p_raw > 1e-15]
    rankme = float(np.exp(-np.sum(p_raw_nz * np.log(p_raw_nz))))

    # 채널 활성률 및 비활성(Dead) 필터 진단
    active_frac = (np.abs(features) > active_thresh).mean(axis=0)
    dead_filters = int((active_frac < 0.20).sum())

    return {
        "r_eff": r_eff,
        "info_efficiency_pct": info_efficiency_pct,
        "participation_ratio": participation_ratio,
        "rankme": rankme,
        "pc_for_90pct": int(np.searchsorted(cum_pca_pct, 90.0) + 1),
        "mean_active_pct": float(active_frac.mean() * 100.0),
        "dead_filters": dead_filters,
    }

# 저랭크 + 미약한 잡음을 가진 576차원 특징 공간 시뮬레이션
rng = np.random.default_rng(42)
latent = rng.normal(size=(2000, 7))                      # 실제 내재 자유도 = 7
proj = rng.normal(size=(7, 576))
synthetic_gap = latent @ proj + 0.03 * rng.normal(size=(2000, 576))
print(analyze_effective_rank_and_pca(synthetic_gap))
```

### 4.2 다중 거리 표현 정렬 검증: 선형 CKA (Centered Kernel Alignment)
서로 다른 비초점 거리(예: `+5 mm` vs `+100 mm`)에서 추출한 중간층 특징 $\mathbf{A} \in \mathbb{R}^{N \times d_1}, \mathbf{B} \in \mathbb{R}^{N \times d_2}$ 가 회전·스케일 변환을 제외하고 동일한 위상 다양체를 학습했는지 검증할 때 사용합니다:

$$\text{CKA}(\mathbf{A}, \mathbf{B}) = \frac{\|\tilde{\mathbf{A}}^\top \tilde{\mathbf{B}}\|_F^2}{\|\tilde{\mathbf{A}}^\top \tilde{\mathbf{A}}\|_F \, \|\tilde{\mathbf{B}}^\top \tilde{\mathbf{B}}\|_F} \in [0, 1]$$

```python
def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """서로 다른 거리/모델의 중간 특징 공간 간 회전 불변 표현 유사도 (0~1)."""
    A_c = A - A.mean(axis=0, keepdims=True)
    B_c = B - B.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(A_c.T @ B_c, ord="fro") ** 2
    norm_a = np.linalg.norm(A_c.T @ A_c, ord="fro")
    norm_b = np.linalg.norm(B_c.T @ B_c, ord="fro")
    return float(cross / (norm_a * norm_b + 1e-12))

# 직교 회전(R)만 다른 동일 위상 표현이라면 CKA는 ~1.0
Q, _ = np.linalg.qr(rng.normal(size=(64, 64)))
feat_dist1 = rng.normal(size=(500, 64))
feat_dist2 = feat_dist1 @ Q
print("Rotated CKA:", round(linear_cka(feat_dist1, feat_dist2), 4))
```

### 4.3 특징·예측 전수 지식 증류(KD) 및 CPU 추론 속도 측정
소형 학생 모델(`ERM`)은 정답 라벨 학습 손실 $\mathcal{L}_{\text{task}}$ 외에, 대형 기준 모델의 위상 예측값 $\hat{\boldsymbol{\phi}}_T$ 와 요약 특징 벡터 $\mathbf{z}_T$ 를 동시에 추종하여 경량화에 따른 정확도 손실을 억제합니다:

$$\mathcal{L} = \mathcal{L}_{\text{cos}}(\hat{\boldsymbol{\phi}}_S, \boldsymbol{\phi}) + 0.5\,\mathcal{L}_{\text{cos}}(\hat{\boldsymbol{\phi}}_S, \hat{\boldsymbol{\phi}}_T) + 0.1\Big(1 - \cos(\mathbf{P}\mathbf{z}_S, \mathbf{z}_T)\Big)$$

```python
import time
import torch.nn.functional as F

def distillation_step_loss(s_pred, y_true, t_pred, s_proj, t_gap):
    """실제 과제 손실 + 교사 예측 전수 + 교사 요약 특징 코사인 정렬 손실."""
    l_task = cyclical_phase_loss(s_pred, y_true)
    l_kd = cyclical_phase_loss(s_pred, t_pred)
    l_feat = 1.0 - F.cosine_similarity(s_proj, t_gap, dim=1).mean()
    return l_task + 0.5 * l_kd + 0.1 * l_feat

# CPU 단일 프레임 추론 지연시간 측정 예시 (PyTorch 기준; 배포 시 ONNX Runtime 권장)
erm32 = EffectiveRankNet(arch="ERM-32").eval()
dummy_in = torch.randn(1, 1, 256, 256)
with torch.no_grad():
    for _ in range(20):
        _ = erm32(dummy_in, "d50")
    t0 = time.perf_counter()
    for _ in range(100):
        _ = erm32(dummy_in, "d50")
    avg_ms = (time.perf_counter() - t0) / 100 * 1000.0
print(f"PyTorch CPU avg latency: {avg_ms:.3f} ms ({1000.0/avg_ms:.1f} Hz)")
```

---

## 5. 모델별 성능 요약 (`AMD Ryzen 7 9800X3D`, ONNX Runtime FP32 4스레드)

| 모델 | 5-거리 총 파라미터 | 단일 배포 파라미터 | 평균 MAE (°) | 평균 스트렐 비 | $r_{\text{eff}}$ | $\eta_{\text{info}}$ (%) | 활성 필터 비율 | CPU 지연시간 (처리 주파수) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ref. (`576`, 13단)** | `1,005.9K` | `930.2K` | `8.17°` | `0.9721` | `7.56` | `31.8%` | - | `1.180 ms` (`847 Hz`) |
| **Mid (`64`)** | `286.0K` | `266.8K` | `8.07°` | - | `10.32` | `56.1%` | - | `0.733 ms` (`1,364 Hz`) |
| **ERM-48 (6단)** | `51.2K` | `46.0K` | `9.33°` | `0.9580` | `7.31` | `51.4%` | `99.85%` | `0.392 ms` (`2,550 Hz`) |
| **ERM-32 (6단, 권장)** | **`32.5K`** | **`27.3K`** | **`10.48°`** | **`0.9454`** | **`7.40`** | **`57.8%`** | **`99.87%`** | **`0.313 ms` (`3,196 Hz`)** |
| **ERM-24 (6단)** | `21.6K` | `16.4K` | `12.31°` | `0.9281` | `7.25` | `62.3%` | `99.88%` | `0.271 ms` (`3,693 Hz`) |

---

## 참고문헌

- [1] P. Zhou et al., "Review of Coherent Laser Beam Combining Research Progress in the Past Decade," *Chin. J. Lasers* 48(4), 0401003 (2021).
- [2] H. Jia et al., "A phase-error prediction method for coherent beam combining via convolutional neural network," *Optik* 246, 167827 (2021).
- [3] Y. Xie et al., "Single-step phase identification and phase locking for coherent beam combination using deep learning," *Sci. Rep.* 14, 7501 (2024).
- [4] O. Roy and M. Vetterli, "The effective rank: A measure of effective dimensionality," *Proc. EUSIPCO*, pp. 606–610 (2007).
- [5] Q. Garrido, R. Balestriero, L. Najman, and Y. LeCun, "RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank," *Proc. ICML*, PMLR 202, pp. 10929–10974 (2023).
