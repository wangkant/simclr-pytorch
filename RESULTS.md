# Results

One full run of `SimCLR.ipynb`, 45 epochs, about 30 min on one GPU. Cell outputs are stripped from the committed notebook.

| | |
|---|---|
| k-NN top-1 (val, k=20, cosine) | **26.21%** |
| Chance | 0.5% |
| Final loss | -3.60181 |

## Setup

- Tiny-ImageNet-200: 100k train / 10k val, 64x64
- ResNet-18 with small-image stem, projection head 512 -> 512 -> 128
- Alignment + uniformity loss, alpha=2, t=2, lambda=1
- SGD, momentum 0.9, weight decay 1e-4, batch 512
- Base LR 0.5, 5 warmup epochs, then cosine decay
- RandomResizedCrop (scale 0.2-1.0), horizontal flip, ColorJitter(0.4, 0.4, 0.4, 0.1) p=0.8, grayscale p=0.2
- AMP, seed 42
- torch 2.3.1+cu121, torchvision 0.18.1, Python 3.10

## Training loss

| epoch | loss | lr |
|---:|---:|---:|
| 0  | -2.52508 | 0.1000 |
| 5  | -3.30153 | 0.5000 |
| 10 | -3.40112 | 0.4810 |
| 15 | -3.44502 | 0.4268 |
| 20 | -3.47780 | 0.3457 |
| 25 | -3.50982 | 0.2500 |
| 30 | -3.54199 | 0.1543 |
| 35 | -3.57247 | 0.0732 |
| 40 | -3.59588 | 0.0190 |
| 44 | -3.60181 | 0.0008 |

The loss keeps going down for all 45 epochs and flattens at the end as the LR goes to 0. The exact numbers can move a little between runs since cuDNN autotuning and AMP are not deterministic.
