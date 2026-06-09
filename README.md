# SimCLR on CIFAR-10 (PyTorch)

A PyTorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709)-style self-supervised contrastive learning on the CIFAR-10 dataset. The model learns visual representations without any labels, then a frozen-feature k-NN classifier measures how useful those representations are downstream.

## Approach

| Component | Choice |
|---|---|
| **Encoder** | ResNet-18 backbone |
| **Projection head** | MLP into the contrastive embedding space |
| **Augmentations** | RandomResizedCrop, ColorJitter, RandomGrayscale, RandomHorizontalFlip (Gaussian blur is intentionally omitted for 32x32 CIFAR images, per the SimCLR paper) |
| **Objective** | Alignment + uniformity (Wang & Isola formulation), closely related to InfoNCE — Wang & Isola show InfoNCE asymptotically optimizes alignment and uniformity |
| **Evaluation** | k-NN over frozen encoder features on CIFAR-10 test set |

The split between alignment (pulling positive pairs together) and uniformity (spreading the feature distribution over the unit sphere) is more numerically stable for small-batch contrastive training than the temperature-scaled softmax variant of NT-Xent, and easier to debug — each component has its own loss curve.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook SimCLR.ipynb      # or open in Colab
```

GPU is not required but strongly recommended — training a useful encoder takes hundreds of epochs on CIFAR-10 even with a small batch.

## Notebook layout

`SimCLR.ipynb` is self-contained and runs end-to-end:

1. **Data loading & augmentation** — Two augmented views per image form positive pairs.
2. **Model** — Encoder (ResNet-18, last FC dropped) + projection head.
3. **Training** — Contrastive loss (alignment + uniformity) for *N* epochs.
4. **Evaluation** — Freeze the encoder, build an embedding for each train and test image, run k-NN classification on the test set.

Hyperparameters live next to the code they configure: batch size in the data-loading cell, the uniformity scale `t` in the loss cell, learning rate in the optimizer cell, epoch count in the training cell, and k in the k-NN evaluation cell.

## Files

- `SimCLR.ipynb` — End-to-end notebook (training + evaluation).
- `requirements.txt` — Pinned-version-free dependency list (`torch`, `torchvision`, `scikit-learn`, `notebook`).

## References

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)
