# ShiftGuard AI

ShiftGuard AI is a machine learning research project investigating when neural-network predictions become unreliable under distribution shift.

## Research Question

Can a machine learning system detect when an input differs enough from its training distribution that its prediction should no longer be trusted?

## Initial Goal

The first version will train an image classifier and investigate how its predictions and confidence change when test images are systematically corrupted or shifted.

## Technologies

- Python
- PyTorch
- NumPy
- scikit-learn
- Matplotlib

## Development Environment

- Python 3.14
- PyTorch 2.13 with ROCm
- AMD Radeon RX 9060 XT
- Ubuntu 26.04
- CIFAR-10 dataset

GPU acceleration through ROCm has been successfully configured and verified.

## Current Status

- Development environment configured
- CIFAR-10 dataset loaded
- Dataset structure inspected
- Training samples visualized
- AMD GPU acceleration enabled
