# ResNet18 on CIFAR10 with PyTorch

This project implements ResNet18 from scratch using PyTorch and trains it on the CIFAR10 image classification dataset.

## Project Structure

```text
ResNet18_CIFAR10/
├── model.py
├── train.py
├── .gitignore
└── README.md

Model
  The model is a CIFAR10-style ResNet18.
  Compared with the original ImageNet ResNet18, this implementation uses:
  3x3 convolution as the first layer
  stride 1 in the first convolution
  no initial max pooling layer
  This is because CIFAR10 images are only 32x32, and aggressive early downsampling may lose too much spatial information.
Dataset
  CIFAR10 contains 10 classes of 32x32 RGB images.
  The training pipeline uses:
  RandomCrop with padding
  RandomHorizontalFlip
  ToTensor
  Normalize
Training
  Default settings:
  Batch size: 128
  Optimizer: SGD
  Learning rate: 0.1
  Momentum: 0.9
  Weight decay: 5e-4
  Epochs: 10
Run
  python train.py
The dataset will be downloaded automatically by torchvision.
Current Result
  After 1 epoch, the model reached about 43% test accuracy in the initial experiment.
  Further training with more epochs and a learning rate scheduler can improve performance.
