# SimCLR on CIFAR-10 (PyTorch)

A PyTorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709)-style self-supervised learning on the CIFAR-10 dataset. The model learns visual representations without labels, which are then evaluated with k-NN classification.

## Approach

- **Encoder**: ResNet-18 backbone
- **Projection Head**: MLP that maps representations to the contrastive space
- **Augmentations**: Random crop, color jitter, grayscale, Gaussian blur
- **Loss**: Alignment + uniformity objectives for contrastive learning
- **Evaluation**: k-NN classifier on frozen representations

## Tech Stack

- PyTorch
- torchvision
- Jupyter Notebook

## Usage

Open `SimCLR.ipynb` in Jupyter or Google Colab and run all cells. The notebook includes:
1. Data loading and augmentation pipeline
2. SimCLR model definition (encoder + projection head)
3. Training loop with contrastive loss
4. k-NN evaluation on learned embeddings

## Results

The learned representations achieve competitive k-NN accuracy on CIFAR-10 without any supervised training labels.

## File Structure

- `SimCLR.ipynb` — Full implementation and training notebook
