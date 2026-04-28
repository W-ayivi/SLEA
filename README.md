# SLEA: Stochastic Saccadic Lightweight Efficient Attention

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Physica%20Scripta-red)](https://doi.org/10.1088/1402-4896)

Official PyTorch implementation of the paper:

> **SLEA: A Stochastic Saccadic Lightweight Efficient Attention Framework with MobileNetV2 for Robust and Explainable Medical Image Classification**
> Williams Ayivi, Xiaoling Zhang, Amil Aligayev
> *Physica Scripta*, Manuscript PHYSSCR-149591

---

## Overview

SLEA is a lightweight framework for 2D medical image classification that draws structural inspiration from human saccadic vision. Rather than applying convolution uniformly across the entire image, SLEA samples a small set of localized crops at each training step and fuses their representations through token-level attention. This design is intended to concentrate representational capacity on spatially informative subregions while keeping the parameter budget compact.

<p align="center">
  <img src="assets/architecture.png" width="90%" alt="SLEA Architecture"/>
</p>

The framework operates in four stages:

| Stage | Component | Description |
|---|---|---|
| A | Stochastic Saccadic Sampler | Draws K=8 random 64×64 crops from a 256×256 input |
| B | Glimpse Encoder | Shared frozen MobileNetV2 → GAP → Linear(128) |
| C | Attentive Fusion | 4-head self-attention over K glimpse tokens → mean pool |
| D | MLP Head | 2-layer classifier for single-label or multi-label output |

---

## Key Properties

- **2.45M parameters** — smaller than DenseNet-121, ResNet-50, and Swin-Tiny
- **0.29 GFLOPs** per single forward pass (K=8, N=1)
- **Reproducible inference** via multi-view voting with fixed random seeds (N=5 passes)
- **Built-in explainability** via glimpse-level and full-image Grad-CAM
- **Spatial regularization** via stochastic sampling, which changes glimpse locations at every epoch and reduces reliance on fixed positional cues

---

## Results

All experiments were run on a single NVIDIA GeForce RTX 4060 GPU (8 GB VRAM) with PyTorch 2.1.

### Main Results (mean ± 95% CI over 3 independent runs)

| Dataset | Task | Accuracy | F1-Macro | ROC-AUC | MCC |
|---|---|---|---|---|---|
| ChestX-ray14 | Multi-label | 98.2 ± 0.18 | 98.0 ± 0.20 | 98.5 ± 0.15 | 97.8 ± 0.22 |
| ISIC 2019 | Multi-class | 99.1 ± 0.12 | 99.1 ± 0.12 | 99.4 ± 0.09 | 99.0 ± 0.13 |
| APTOS 2019 | Multi-class | 99.3 ± 0.14 | 99.2 ± 0.15 | 99.6 ± 0.10 | 99.1 ± 0.16 |
| COVID-19 | Binary | 100.0 ± 0.00 | 100.0 ± 0.00 | ---† | ---† |

† ROC-AUC is omitted for COVID-19 because all three runs reached perfect binary separation (zero score variance), making it uninformative. The 100% result reflects dataset separability rather than model superiority and should not be interpreted as a clinical claim.

> **Note on ChestX-ray14:** The reported AUC of 98.5% is notably higher than the established benchmark range (82–90%) and should be interpreted with caution. We used the official patient-level split and standard one-vs-rest evaluation, but differences in preprocessing, inference protocol, and implementation details can influence absolute performance on this benchmark. We regard independent replication under closely matched conditions as an important next step.

### Efficiency

| Configuration | Params (M) | FLOPs single-pass (G) | FLOPs full inference (G) |
|---|---|---|---|
| SLEA | 2.45 | 0.29 | 1.45 |
| MobileNetV2 (reference) | 2.2 | 0.39 | 0.39 |
| DenseNet-121 (reference) | 7.9 | 3.71 | 3.71 |
| ResNet-50 (reference) | 25.6 | 5.37 | 5.37 |
| Swin-Tiny (reference) | 29.0 | 5.90 | 5.90 |

Full-inference FLOPs for SLEA reflect K×N = 8×5 = 40 backbone evaluations under multi-view voting. Reference baseline FLOPs reflect a single forward pass and are not directly comparable.

---

## Installation

```bash
git clone https://github.com/your-username/slea.git
cd slea
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (recommended)

---

## Project Structure

```
slea/
├── models/
│   ├── slea.py          # Full model: Stages A–D
│   └── gradcam.py       # Grad-CAM explainability
├── datasets/
│   └── datasets.py      # Dataset classes for all four benchmarks
├── scripts/
│   ├── train.py         # Training loop
│   └── evaluate.py      # Evaluation with multi-view voting
├── utils/
│   ├── metrics.py       # Full metric suite (AUC, F1, MCC, specificity)
│   └── logger.py        # Console + file logger
├── configs/
│   ├── chestxray14.yaml
│   ├── isic2019.yaml
│   ├── aptos2019.yaml
│   └── covid19.yaml
└── requirements.txt
```

---

## Data Preparation

All four datasets are publicly available. Download and organize as follows:

### ChestX-ray14
Download from [Kaggle](https://www.kaggle.com/datasets/mohammadzunaed/nih-chest-x-ray14) or the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC).

```
chestxray14/
    images/           *.png
    train_official.csv
    val_official.csv
    test_official.csv
```

Use the **official patient-level splits** provided by Wang et al. (2017) to prevent data leakage. The CSV files must contain an `Image Index` column and one column per pathology label (0 or 1).

### ISIC 2019
Download from [Kaggle](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification).

```
isic2019/
    images/           *.jpg
    train.csv         # columns: image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC
    val.csv
    test.csv
```

### APTOS 2019
Download from [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection).

```
aptos2019/
    train_images/     *.png
    train.csv         # columns: id_code, diagnosis (0–4)
    val.csv
    test.csv
```

### COVID-19 Radiography Database
Download from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

```
covid19/
    train/
        COVID/    *.png
        Normal/   *.png
    val/
        COVID/    *.png
        Normal/   *.png
    test/
        COVID/    *.png
        Normal/   *.png
```

Update the paths in the corresponding YAML config file before running.

---

## Training

```bash
# ChestX-ray14
python scripts/train.py --config configs/chestxray14.yaml

# ISIC 2019
python scripts/train.py --config configs/isic2019.yaml

# APTOS 2019  (uses FP32; use_amp: false in config)
python scripts/train.py --config configs/aptos2019.yaml

# COVID-19
python scripts/train.py --config configs/covid19.yaml
```

Checkpoints are saved after every epoch to `outputs/<dataset>/`. The best checkpoint by validation metric is saved as `best_model.pt`.

---

## Evaluation

```bash
# Single-pass evaluation
python scripts/evaluate.py \
    --config configs/chestxray14.yaml \
    --checkpoint outputs/chestxray14/best_model.pt

# Multi-view voting (N=5, reproducible)
python scripts/evaluate.py \
    --config configs/chestxray14.yaml \
    --checkpoint outputs/chestxray14/best_model.pt \
    --multi_view
```

---

## Quick API Usage

```python
import torch
from models import SLEA, GradCAM

# Build model
model = SLEA(
    n_classes       = 14,     # ChestX-ray14
    image_size      = 256,
    glimpse_size    = 64,
    n_glimpses      = 8,
    token_dim       = 128,
    n_heads         = 4,
    freeze_backbone = True,
    pretrained      = True,
    n_votes         = 5,
)

# Single stochastic forward pass (training)
x      = torch.randn(4, 3, 256, 256)
logits = model(x, multi_view=False)   # (4, 14)

# Reproducible multi-view voting (inference)
probs  = model(x, multi_view=True)    # (4, 14)

# Expected spatial coverage at K=8
print(f"Expected coverage: {model.expected_coverage:.1%}")  # ~40.3%

# Grad-CAM
gradcam   = GradCAM(model)
heatmaps  = gradcam.generate(x[:1], target_class=3)  # K glimpse heatmaps
global_cam = gradcam.generate_global(x[:1])            # full-image overlay
gradcam.remove_hooks()
```

---

## Hyperparameters

All experiments in the paper use the following settings:

| Hyperparameter | Value |
|---|---|
| Backbone | MobileNetV2 (ImageNet pretrained, frozen) |
| Input resolution S | 256 × 256 |
| Glimpse size G | 64 × 64 |
| Number of glimpses K | 8 |
| Token dimension D | 128 |
| Attention heads | 4 |
| MLP hidden units | 128 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁵ |
| Batch size | 8 |
| Epochs | 50 |
| Gradient clip | 1.0 |
| Inference voting passes N | 5 |
| Precision (CUDA) | AMP (FP32 for APTOS2019) |

---

## Ablation Studies

The paper reports five ablation studies. To replicate, modify the relevant parameter in the config file:

| Study | Parameter | Values tested |
|---|---|---|
| Glimpse count | `n_glimpses` | 4, **8**, 12, 16 |
| Crop size | `glimpse_size` | 32, 48, **64**, 96 |
| Token fusion | (code, not config) | Mean Pool, Concat+Linear, **MHA** |
| Sampling strategy | (code, not config) | Uniform Grid, Sliding Window, Gaussian, Foveated, **Stochastic** |
| Backbone strategy | `freeze_backbone` | From scratch, Fine-tuned, **Frozen pretrained** |

Bold denotes the default configuration used in all reported experiments.

---

## Explainability

SLEA supports two modes of Grad-CAM visualization:

**Glimpse-level heatmaps** — one Grad-CAM map per sampled crop, showing which local features anchored the prediction.

**Full-image overlay** — the full resized image is passed as a single K=1 glimpse to produce a global saliency map.

```python
gradcam = GradCAM(model, target_layer='encoder.features.18')

# Glimpse-level heatmaps for all K=8 crops
heatmaps = gradcam.generate(image_tensor)

# Full-image overlay
cam = gradcam.generate_global(image_tensor)

# Overlay on original image
import numpy as np
image_np = (image_tensor.squeeze().permute(1,2,0).numpy() * 255).astype(np.uint8)
overlay  = GradCAM.overlay(image_np, cam, alpha=0.4)
```

---

## Limitations

- SLEA is developed and evaluated for **2D medical image classification only**. Extension to 3D volumetric modalities such as CT and MRI would require substantial architectural changes.
- Stochastic inference does not guarantee exhaustive spatial coverage. Very small focal findings may be missed in a given set of test-time glimpses even under multi-view voting.
- The frozen MobileNetV2 backbone cannot adapt lower-level feature representations to medical image statistics that differ substantially from ImageNet.
- All evaluations are conducted on publicly available benchmark datasets under controlled conditions. Performance on noisier, multi-institutional, or prospectively collected clinical data may differ.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ayivi2025slea,
  title   = {{SLEA}: A Stochastic Saccadic Lightweight Efficient Attention
             Framework with {MobileNetV2} for Robust and Explainable
             Medical Image Classification},
  author  = {Ayivi, Williams and Zhang, Xiaoling and Aligayev, Amil},
  journal = {Physica Scripta},
  year    = {2025},
  note    = {Manuscript PHYSSCR-149591}
}
```

---

## Acknowledgements

This work was supported by the National Natural Science Foundation of China under Grants 62471113 and the Sichuan Science and Technology Program 2024NS-FSC0479. Additional support from the European Union Horizon 2020 research and innovation programme under grant agreement no. 857470, and from the European Regional Development Fund via the Foundation for Polish Science International Research Agenda PLUS programme, grant No. MAB PLUS/2018/8.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
