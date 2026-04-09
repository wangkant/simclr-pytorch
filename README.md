# SimCLR-Style Representation Learning on CIFAR-10

This project implements a SimCLR-style self-supervised learning pipeline in PyTorch.

## What it does
- Uses a ResNet-18 encoder with a projection head
- Applies image augmentations to create positive pairs
- Trains with a contrastive learning objective
- Evaluates learned representations on CIFAR-10 using k-NN

## Tech Stack
PyTorch, torchvision, Python, Jupyter Notebook

## Notes
This is a personal implementation for learning self-supervised and contrastive representation learning.
