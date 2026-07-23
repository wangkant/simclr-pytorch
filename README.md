# SimCLR on Tiny-ImageNet-200 (PyTorch)

A PyTorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709)-style
self-supervised contrastive learning on **Tiny-ImageNet-200** (200 classes, 64×64).
The model learns visual representations without any labels, then a frozen-feature k-NN
classifier measures how useful those representations are downstream.

> Upgraded from the original CIFAR-10 version of this notebook — bigger dataset, stronger
> training recipe. The CIFAR-10 implementation is preserved in the git history.

## Result

**k-NN top-1 accuracy: 26.21%** on the Tiny-ImageNet-200 validation set (200-way,
chance = 0.5%), after 45 epochs of from-scratch self-supervised pre-training (~30 min on
a single GPU). Full numbers and the training curve are in [`RESULTS.md`](RESULTS.md).

## Approach

| Component | Choice |
|---|---|
| **Encoder** | ResNet-18 with a small-image stem (3×3 conv1, single 2× downsample) — the stock 7×7-stride-2 + maxpool stem discards too much of a 64×64 image |
| **Projection head** | MLP into the contrastive embedding space (512 → 512 → 128) |
| **Augmentations** | RandomResizedCrop, ColorJitter, RandomGrayscale, RandomHorizontalFlip (Gaussian blur intentionally omitted for these small images, per the SimCLR paper) |
| **Objective** | Alignment + uniformity (Wang & Isola formulation), closely related to InfoNCE |
| **Optimizer** | SGD + momentum 0.9 + weight decay 1e-4, cosine LR schedule with linear warmup |
| **Precision** | AMP (mixed precision) |
| **Evaluation** | GPU k-NN (k=20, cosine) over frozen encoder features |

The split between alignment (pulling positive pairs together) and uniformity (spreading
the feature distribution over the unit sphere) is more numerically stable for small-batch
contrastive training than the temperature-scaled softmax variant of NT-Xent, and easier
to debug — each component has its own loss curve.

## Data setup

torchvision has no built-in Tiny-ImageNet loader, so fetch the dataset once into
`datasets/` (the notebook reads `datasets/tiny-imagenet-200/`):

```bash
mkdir -p datasets && cd datasets
curl -fL -o tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip && cd ..
```

This yields `datasets/tiny-imagenet-200/{train,val,...}` (100k train images across 200
classes, 10k labelled val images). `datasets/` is git-ignored.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook SimCLR.ipynb      # or run headless (below)
```

GPU strongly recommended. The pinned CUDA-12 wheels in `requirements.txt` are deliberate:
an unpinned `torch` currently resolves to a CUDA-13 build, which fails to initialise on
CUDA-12 drivers and silently falls back to CPU.

Run end-to-end, headless, writing outputs back into the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace SimCLR.ipynb
```

## Notebook layout

`SimCLR.ipynb` is self-contained and runs end-to-end:

1. **Data loading & augmentation** — two augmented views per image form positive pairs.
2. **Model** — ResNet-18 encoder (small-image stem, FC dropped) + projection head.
3. **Training** — alignment + uniformity contrastive loss, SGD + cosine schedule, AMP.
   Per-epoch loss is also streamed to `run_logs/train_live.log` so long headless runs are
   monitorable.
4. **Evaluation** — freeze the encoder, embed every train and val image, run cosine k-NN
   on the val set (done on-GPU: 100k × 10k similarities, chunked).

Hyperparameters live in a single **Configuration** cell near the top (dataset path, image
size, batch, epochs, base LR, warmup, k).

## Reproducibility

A seed cell near the top sets `SEED = 42` for Python's `random`, NumPy, and PyTorch (CPU
and all CUDA devices). Caveat: seeding alone does **not** guarantee bit-exact repeats on
GPU — cuDNN autotuned convolutions and AMP are nondeterministic, and `cudnn.benchmark` is
left on for speed. Full determinism would additionally require
`torch.use_deterministic_algorithms(True)` and `cudnn.deterministic = True /
benchmark = False`, at a performance cost, so it is not enforced here. The trend and
ballpark are stable; exact digits may drift across runs and torch/CUDA versions.

## Files

- `SimCLR.ipynb` — end-to-end notebook (training + evaluation).
- `RESULTS.md` — full results: headline accuracy, setup, and per-epoch training curve.
- `requirements.txt` — dependencies with bounded/pinned, driver-compatible versions.

## References

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*
  (SimCLR), [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and
  Uniformity on the Hypersphere*, [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)
- Le & Yang, *Tiny ImageNet Visual Recognition Challenge* (Stanford CS231N)
