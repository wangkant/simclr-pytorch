# SimCLR on Tiny-ImageNet-200

PyTorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709)-style self-supervised learning on Tiny-ImageNet-200 (200 classes, 64x64). The encoder is trained without labels and then evaluated with k-NN on the frozen features.

k-NN top-1 accuracy is **26.21%** on the val set (chance is 0.5%) after 45 epochs from scratch. Details are in [RESULTS.md](RESULTS.md).

## Method

- ResNet-18 with a 3x3 conv1 and only one downsample, since the default stem loses too much of a 64x64 image
- MLP projection head (512 -> 512 -> 128)
- Alignment + uniformity loss (Wang & Isola) instead of NT-Xent
- SGD with momentum, 5 warmup epochs then cosine LR, mixed precision
- k-NN (k=20, cosine) on GPU for evaluation

## Running

Download Tiny-ImageNet into `datasets/`:

```bash
mkdir -p datasets && cd datasets
curl -fL -o tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q tiny-imagenet-200.zip && cd ..
```

Install and open the notebook:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" notebook
jupyter notebook SimCLR.ipynb
```

Use the CUDA 12 wheels. An unpinned torch pulls a CUDA 13 build, which falls back to CPU on a CUDA 12 driver.

## References

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, [arXiv:2005.10242](https://arxiv.org/abs/2005.10242)
