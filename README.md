# Deep Learning for CBC Phase Retrieval

English | [한국어](README_KO.md)

Poster supplement — Sang-Eon Park, Sungmin Park, Dong-Il Yeom (Ajou University).

Covers model architectures, PCA/entropy effective-rank analysis, and a dummy-tensor inference latency benchmark. Datasets and trained weights are omitted.

---

## 1. Model Architectures

Input is a `(1, 256, 256)` grayscale interference image; output is 6 relative phases (rad, beam 0 as reference).
During multi-distance training, early stem stages are duplicated per defocus distance while the trunk and head are shared. Single-distance deployment uses 1 stem + trunk + linear head.

| Model | Blocks | Final Grid | Params |
| :--- | :---: | :---: | :---: |
| `Ref. 576` | `5 + 8` | `576×8×8` | `930,182` |
| `Ch48` | `2 + 4` | `48×16×16` | `46,022` |
| `Ch32` | `2 + 4` | `32×16×16` | `27,302` |
| `Ch24` | `2 + 4` | `24×16×16` | `16,406` |

![Model architectures](assets/fig5_erm_architecture.png)

<details>
<summary>Ref. 576 (Modified MobileNetV3-Small, 13 blocks)</summary>

| Stage | Op | Out `(C, H, W)` | Params |
| :--- | :--- | :---: | :---: |
| `0 (Stem)` | `Conv 1→16, k3 s2, HS` | `16, 128, 128` | `176` |
| `1 (Stem)` | `InvRes 16→16→16, k3 s2, SE, RE` | `16, 64, 64` | `744` |
| `2 (Stem)` | `InvRes 16→72→24, k3 s2, RE` | `24, 32, 32` | `3,864` |
| `3 (Stem)` | `InvRes 24→88→24, k3 s1, RE` | `24, 32, 32` | `5,416` |
| `4 (Stem)` | `InvRes 24→96→40, k5 s2, SE, HS` | `40, 16, 16` | `13,736` |
| `5 (Trunk)` | `InvRes 40→240→40, k5 s1, SE, HS` | `40, 16, 16` | `57,264` |
| `6 (Trunk)` | `InvRes 40→240→40, k5 s1, SE, HS` | `40, 16, 16` | `57,264` |
| `7 (Trunk)` | `InvRes 40→120→48, k5 s1, SE, HS` | `48, 16, 16` | `21,968` |
| `8 (Trunk)` | `InvRes 48→144→48, k5 s1, SE, HS` | `48, 16, 16` | `29,800` |
| `9 (Trunk)` | `InvRes 48→288→96, k5 s2, SE, HS` | `96, 8, 8` | `91,848` |
| `10 (Trunk)` | `InvRes 96→576→96, k5 s1, SE, HS` | `96, 8, 8` | `294,096` |
| `11 (Trunk)` | `InvRes 96→576→96, k5 s1, SE, HS` | `96, 8, 8` | `294,096` |
| `12 (Trunk)` | `Conv 96→576, k1 s1, HS` | `576, 8, 8` | `56,448` |
| `GAP + FC` | `AvgPool(1) + Linear(576→6)` | `6` | `3,462` |

Stages 9–12 (`8×8` tail): `736,488` params (`79.2%` of total).

</details>

<details>
<summary>Ch32 (6 blocks, 27,302 params)</summary>

| Stage | Op | Out `(C, H, W)` | Params |
| :--- | :--- | :---: | :---: |
| `0 (Stem)` | `Conv 1→16, k3 s2, HS` | `16, 128, 128` | `176` |
| `1 (Stem)` | `InvRes 16→24→16, k3 s2, RE` | `16, 64, 64` | `1,112` |
| `2 (Trunk)` | `InvRes 16→48→24, k3 s2, RE` | `24, 32, 32` | `2,592` |
| `3 (Trunk)` | `InvRes 24→48→24, k5 s1, SE, HS` | `24, 32, 32` | `5,344` |
| `4 (Trunk)` | `InvRes 24→72→32, k5 s2, SE, HS` | `32, 16, 16` | `9,736` |
| `5 (Trunk)` | `InvRes 32→64→32, k5 s1, SE, HS` | `32, 16, 16` | `8,144` |
| `GAP + FC` | `AvgPool(1) + Linear(32→6)` | `6` | `198` |

</details>

<details>
<summary>Ch48 (46,022) & Ch24 (16,406)</summary>

Both use the same `Stage 0–1` stem as `Ch32` (`1,288` params).

#### `Ch48` Trunk + Head
| Stage | Op | Out `(C, H, W)` | Params |
| :--- | :--- | :---: | :---: |
| `2 (Trunk)` | `InvRes 16→48→24, k3 s2, RE` | `24, 32, 32` | `2,592` |
| `3 (Trunk)` | `InvRes 24→72→32, k5 s1, SE, HS` | `32, 32, 32` | `9,736` |
| `4 (Trunk)` | `InvRes 32→96→48, k5 s2, SE, HS` | `48, 16, 16` | `15,288` |
| `5 (Trunk)` | `InvRes 48→96→48, k5 s1, SE, HS` | `48, 16, 16` | `16,824` |
| `GAP + FC` | `AvgPool(1) + Linear(48→6)` | `6` | `294` |

#### `Ch24` Trunk + Head
| Stage | Op | Out `(C, H, W)` | Params |
| :--- | :--- | :---: | :---: |
| `2 (Trunk)` | `InvRes 16→32→16, k3 s2, RE` | `16, 32, 32` | `1,472` |
| `3 (Trunk)` | `InvRes 16→32→24, k5 s1, SE, HS` | `24, 32, 32` | `2,808` |
| `4 (Trunk)` | `InvRes 24→48→24, k5 s2, SE, HS` | `24, 16, 16` | `5,344` |
| `5 (Trunk)` | `InvRes 24→48→24, k5 s1, SE, HS` | `24, 16, 16` | `5,344` |
| `GAP + FC` | `AvgPool(1) + Linear(24→6)` | `6` | `150` |

</details>

`InvRes(in→exp→out)`: MobileNetV3 inverted residual block (`SE`: squeeze-and-excitation, `RE`: ReLU, `HS`: Hardswish).

---

## 2. Effective Rank Analysis (PCA & Shannon Entropy)

From the GAP feature matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ ($N = 10{,}000$ validation samples across 5 distances), let $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{X}}$ be the mean-centered matrix:

$$\mathbf{C} = \frac{1}{N - 1}\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}, \qquad p_i = \frac{\lambda_i}{\sum_{j=1}^{D}\lambda_j}$$

$$H = -\sum_{p_i > 0} p_i \ln p_i, \qquad r_{\mathrm{eff}} = e^{H}, \qquad \eta_{\mathrm{info}} = \frac{H}{\ln D}$$

$r_{\mathrm{eff}} \in [1, D]$ counts effective independent feature modes (`1` if a single mode dominates, `D` if variance is uniform) [1]. Replacing $\mathbf{C}$ with the uncentered Gram matrix gives RankMe [2].

Active rate is the mean fraction of GAP entries with `|x| > 1e-3`.

| Model (`D`) | $r_{\mathrm{eff}}$ | $\eta_{\mathrm{info}}$ | Active |
| :--- | :---: | :---: | :---: |
| `Ref. 576` | `7.56` | `31.8%` | `31.1%` |
| `Ch48` | `7.31` | `51.4%` | `99.85%` |
| `Ch32` | `7.40` | `57.8%` | `99.87%` |
| `Ch24` | `7.25` | `62.3%` | `99.88%` |

On `Ref. 576`, the top 7 PCs capture ≥ 90% of variance (top 9 ≥ 95%); on `Ch32`, the top 6 PCs capture ≥ 94%.

![PCA and effective rank](assets/fig4_pca_effective_rank.png)

<details>
<summary>NumPy snippet: effective_rank(X)</summary>

```python
import numpy as np

def effective_rank(X: np.ndarray, active_thresh: float = 1e-3):
    N, D = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)
    eig = np.maximum(np.linalg.eigvalsh((X_c.T @ X_c) / (N - 1))[::-1], 0.0)
    p = eig / (eig.sum() + 1e-15)
    p_nz = p[p > 1e-15]
    H = -np.sum(p_nz * np.log(p_nz))
    return {
        "r_eff": float(np.exp(H)),
        "eta_info_pct": float(H / np.log(D) * 100.0),
        "pc_90": int(np.searchsorted(np.cumsum(p), 0.90) + 1),
        "active_pct": float((np.abs(X) > active_thresh).mean() * 100.0),
    }
```

</details>

---

## 3. Inference Latency Benchmark

```bash
pip install -r requirements.txt
python benchmark.py
```

Optional flags:

```bash
python benchmark.py --runs 200 --warmup 30 --threads 4 --device cpu --rank-demo
```

`benchmark.py` runs single-frame dummy `(1, 1, 256, 256)` tensors in PyTorch eager mode (`torch`, `torchvision`, `numpy` only). Poster timings (`1.180 / 0.392 / 0.313 / 0.271 ms` for `Ref. 576 / Ch48 / Ch32 / Ch24`) were measured in ONNX Runtime FP32 (4 threads, Ryzen 7 9800X3D).

---

## References

- `[1]` O. Roy & M. Vetterli, "The effective rank: A measure of effective dimensionality," *Proc. EUSIPCO*, pp. 606–610 (2007).
- `[2]` Q. Garrido et al., "RankMe: Assessing the downstream performance of pretrained self-supervised representations by their rank," *Proc. ICML*, PMLR 202, pp. 10929–10974 (2023).
