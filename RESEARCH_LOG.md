# ShiftGuard AI – Research Log

## 2026-09-16

### Goal

Set up the ShiftGuard AI development environment, prepare CIFAR-10, configure GPU acceleration, and implement the first baseline CNN.

### Research Question

How does the reliability of neural-network confidence change under increasing distribution shift?

The long-term goal is to detect when a neural network is making predictions on unfamiliar or strongly altered data and should therefore not be trusted.

---

### Development Environment

Completed:

- Created the public GitHub repository
- Set up a Python virtual environment
- Installed the required ML libraries
- Configured PyTorch
- Created the initial project structure
- Connected Hackatime and Stardance

The project is now ready for reproducible machine-learning experiments.

---

### CIFAR-10 Dataset

Prepared and explored the CIFAR-10 dataset:

- 50,000 training images
- 10,000 test images
- 10 classes
- 32 × 32 RGB images
- Pixel values converted to 0.0–1.0
- Data loaded in batches using PyTorch DataLoader

I also created a visualization of random training images.

### Observation

The images are very low-resolution and some classes are already difficult to distinguish. This may become important when later applying blur, noise, brightness changes and other distribution shifts.

---

### AMD GPU and ROCm Setup

The initial PyTorch installation used NVIDIA CUDA and could not use my AMD Radeon RX 9060 XT.

I installed ROCm 10 and switched to a ROCm-compatible PyTorch build.

Verified:

- GPU: AMD Radeon RX 9060 XT
- PyTorch: 2.13.0+rocm10.0.0
- ROCm/HIP detected
- GPU acceleration available

### What I Learned

CUDA is designed for NVIDIA GPUs, while AMD GPUs use ROCm/HIP. PyTorch still exposes many ROCm operations through the `torch.cuda` interface, allowing similar training code to work across both platforms.

---

### Baseline CNN

Implemented and successfully tested the first CNN for CIFAR-10.

Architecture:

- Input: 3 × 32 × 32 RGB image
- Conv2D: 3 → 32 feature maps
- ReLU activation
- MaxPooling
- Conv2D: 32 → 64 feature maps
- ReLU activation
- MaxPooling
- Flatten: 64 × 8 × 8 = 4096 features
- Fully connected layer: 4096 → 128
- ReLU activation
- Output layer: 128 → 10 logits

The output contains one logit for each CIFAR-10 class.

To verify the architecture, I passed a dummy batch of four artificial 32 × 32 RGB images through the network.

Test result:

- Input shape: `[4, 3, 32, 32]`
- Output shape: `[4, 10]`
- Architecture test completed successfully

This confirmed that all convolutional, pooling and linear layers are dimensionally compatible.

---

### Current State

- Development environment configured
- CIFAR-10 loaded and visualized
- AMD GPU acceleration working
- Baseline CNN implemented and tested
- Project ready for model training

### Next Steps

- Implement the training pipeline
- Move model and data to the GPU
- Define loss function and optimizer
- Train the baseline CNN
- Measure loss and accuracy
- Save and evaluate the trained model
- Begin controlled distribution-shift experiments



## 2026-09-17

### Baseline CNN Training

Trained the baseline CNN on the 50,000 CIFAR-10 training images for five epochs using the AMD Radeon RX 9060 XT.

Results:

- Epoch 1: Loss 1.3057 | Accuracy 53.47%
- Epoch 2: Loss 0.9289 | Accuracy 67.42%
- Epoch 3: Loss 0.7707 | Accuracy 73.19%
- Epoch 4: Loss 0.6499 | Accuracy 77.30%
- Epoch 5: Loss 0.5520 | Accuracy 80.57%

The steadily decreasing loss and increasing training accuracy indicate that the CNN successfully learned patterns from the training dataset.

The trained model weights were saved as `models/baseline_cnn.pth`.

### Next Step

Evaluate the trained model on the unseen CIFAR-10 test set and establish the clean-data baseline for accuracy, loss and prediction confidence.