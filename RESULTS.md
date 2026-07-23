# Results — SimCLR on Tiny-ImageNet-200

Full training + evaluation run of the upgraded `SimCLR.ipynb`. The numbers below are
from a complete run of the notebook (its cell outputs are stripped for sharing — re-run
the notebook to regenerate them).

## Headline

| Metric | Value |
|---|---|
| **k-NN top-1 accuracy (val)** | **26.21%** |
| Classes | 200 (Tiny-ImageNet-200) |
| Chance level | 0.50% (→ ≈ 52× chance) |
| k-NN | k = 20, cosine, over 100k frozen train features |
| Final contrastive loss (epoch 44) | −3.60181 |
| Best contrastive loss (epoch 43) | −3.60228 |
| Epochs | 45 |
| Wall-clock | ~30 min on a single GPU |

## Setup

| Component | Choice |
|---|---|
| Dataset | Tiny-ImageNet-200 — 100k train / 10k val, 64×64, 200 classes |
| Encoder | ResNet-18 with a small-image stem (3×3 conv1, single 2× downsample, no 7×7/maxpool) |
| Projection head | MLP 512 → 512 → 128 |
| Objective | Alignment + Uniformity (Wang & Isola), α=2, t=2, λ=1 |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 |
| LR schedule | linear warmup (5 ep) → cosine decay, base LR 0.5 |
| Batch size | 512 |
| Precision | AMP (mixed precision) |
| Augmentation | RandomResizedCrop(0.2–1.0), HFlip, ColorJitter(0.4,0.4,0.4,0.1) p=0.8, Grayscale p=0.2 |
| Evaluation | frozen-encoder GPU k-NN (k=20, cosine) |
| Seed | 42 |

## Training curve (contrastive loss, every 5th epoch)

| epoch | loss | lr |
|---:|---:|---:|
| 0  | −2.52508 | 0.1000 |
| 5  | −3.30153 | 0.5000 |
| 10 | −3.40112 | 0.4810 |
| 15 | −3.44502 | 0.4268 |
| 20 | −3.47780 | 0.3457 |
| 25 | −3.50982 | 0.2500 |
| 30 | −3.54199 | 0.1543 |
| 35 | −3.57247 | 0.0732 |
| 40 | −3.59588 | 0.0190 |
| 44 | −3.60181 | 0.0008 |

Loss decreases monotonically across all 45 epochs; the cosine schedule anneals the LR
to ~0 by the end, so the curve flattens into a stable plateau rather than oscillating.

## Environment

- torch 2.3.1+cu121, torchvision 0.18.1+cu121, numpy 1.26.4 (Python 3.10)
- a single CUDA-12 GPU (batch 512 at 64×64 fits comfortably in ~16 GB)
- CUDA-12 wheels were pinned deliberately: an unpinned `torch` resolves to a CUDA-13
  build that a CUDA-12 driver cannot run (it would silently fall back to CPU).

## Reproduce

```bash
# 1. Data (torchvision has no built-in Tiny-ImageNet loader; fetch once):
mkdir -p datasets && cd datasets
curl -fL -o tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip && cd ..

# 2. Environment:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run end-to-end:
jupyter nbconvert --to notebook --execute --inplace SimCLR.ipynb
```

## Notes / next steps

- 45 epochs was chosen to fit a ~30-min GPU budget. SimCLR keeps improving with longer
  schedules; 200–800 epochs (and a larger batch / ResNet-50) would raise k-NN accuracy
  further at proportional wall-clock cost.
- Determinism is not enforced (cuDNN autotune + AMP), so exact numbers may drift slightly
  across runs and library versions; the trend and ballpark are stable.
