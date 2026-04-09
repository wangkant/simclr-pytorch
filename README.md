# SimCLR-Style Self-Supervised Learning on CIFAR-10

This project is a PyTorch implementation of a SimCLR-style self-supervised learning pipeline on CIFAR-10. It uses a ResNet-18 encoder with a projection head, applies image augmentations to construct positive pairs, and trains the model with alignment and uniformity loss. The learned representations are evaluated with k-NN classification.
