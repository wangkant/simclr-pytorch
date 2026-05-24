# SimCLR on CIFAR-10 (PyTorch)

A PyTorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709)-style self-supervised contrastive learning on CIFAR-10. The model learns image representations without any labels, and then a k-NN classifier on the frozen features is used to check how good those representations are.

## Approach

| Component | Choice |
|---|---|
| **Encoder** | ResNet-18 backbone |
| **Projection head** | MLP into the contrastive embedding space |
| **Augmentations** | RandomResizedCrop, ColorJitter, RandomGrayscale, GaussianBlur, RandomHorizontalFlip |
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

All hyperparameters are in a single config cell at the top of the notebook (batch size, learning rate, temperature, epoch count, k for k-NN).

## Files

- `SimCLR.ipynb` - the notebook (training + evaluation).
- `requirements.txt` - dependency list (`torch`, `torchvision`, `scikit-learn`).

## References

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)
