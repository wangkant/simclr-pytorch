# SimCLR on CIFAR-10 (PyTorch)

A PyTorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709)-style self-supervised contrastive learning on CIFAR-10. The model learns image representations without any labels, and then a k-NN classifier on the frozen features is used to check how good those representations are.

## Approach

| Component | Choice |
|---|---|
| **Encoder** | ResNet-18 backbone |
| **Projection head** | MLP into the contrastive embedding space |
| **Augmentations** | RandomResizedCrop, ColorJitter, RandomGrayscale, RandomHorizontalFlip (no Gaussian blur, the SimCLR paper skips it for small 32x32 images) |
| **Objective** | Alignment + uniformity (Wang & Isola formulation) |
| **Evaluation** | k-NN over frozen encoder features on the CIFAR-10 test set |

I used the alignment + uniformity loss instead of the usual NT-Xent softmax version because it seems to train more stably with a small batch size, and it is easier to debug since each of the two parts has its own loss curve.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook SimCLR.ipynb      # or open in Colab
```

A GPU is not required but is strongly recommended, since training a useful encoder takes a lot of epochs on CIFAR-10 even with a small batch.

## Notebook layout

`SimCLR.ipynb` runs end to end:

1. **Data loading & augmentation** - two augmented views of each image form a positive pair.
2. **Model** - encoder (ResNet-18, last FC dropped) + projection head.
3. **Training** - contrastive loss (alignment + uniformity) for *N* epochs.
4. **Evaluation** - freeze the encoder, get an embedding for every train and test image, then run k-NN on the test set.

The hyperparameters sit next to the code that uses them: batch size in the data-loading cell, the uniformity scale `t` in the loss cell, learning rate in the optimizer cell, epoch count in the training cell, and k in the k-NN cell.

## Reproducibility

There is a seed cell near the top that sets `SEED = 42` for `random`, NumPy and PyTorch (CPU and CUDA), so the shuffling and augmentation sampling repeat.

This does not make GPU runs bit-exact though, since some CUDA kernels are nondeterministic. Getting full determinism would also need `torch.use_deterministic_algorithms(True)` and `cudnn.deterministic = True` / `benchmark = False`, which costs speed, so I left it off. Numbers can also shift a bit between torch/CUDA versions.

## Current status

The outputs saved in the notebook are from a partial run: training stopped after 2 of the 10 epochs and I never ran the k-NN cell, so there is no accuracy number committed yet. Still waiting on GPU time for a full run.

## Files

- `SimCLR.ipynb` - the notebook (training + evaluation).
- `requirements.txt` - dependencies with version bounds (`torch`, `torchvision`, `numpy`, `scikit-learn`, `notebook`).

## References

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)
