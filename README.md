# ShiftGuard AI

ShiftGuard AI is a machine learning research project investigating when neural-network predictions become unreliable under distribution shift.

## Research Question

Can a machine learning system detect when an input differs enough from its training distribution that its prediction should no longer be trusted?

## Approach

The project first trains a baseline image classifier on CIFAR-10. The model will then be tested on systematically modified images, such as blurred, noisy or brightness-shifted inputs, to analyze how accuracy and confidence change under distribution shift.

The long-term goal is to develop a reliability score that can identify potentially untrustworthy predictions.

## Technologies

- Python
- PyTorch
- NumPy
- scikit-learn
- Matplotlib
- ROCm

## Development Environment

- Ubuntu 26.04
- Python 3.14
- PyTorch 2.13 with ROCm
- AMD Radeon RX 9060 XT
- CIFAR-10

## Current Progress

- Development and GPU environment configured
- CIFAR-10 loaded, inspected and visualized
- Baseline CNN implemented and successfully tested
- Next: train and evaluate the baseline CNN

## Research Log

Detailed development notes and experimental results are documented in [`RESEARCH_LOG.md`](RESEARCH_LOG.md).