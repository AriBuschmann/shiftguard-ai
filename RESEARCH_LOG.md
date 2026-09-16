# ShiftGuard AI – Research Log

## 2026-09-16 3:30 pm

### Goal
Set up the development environment and prepare the first baseline experiment.

### Research question
How does the reliability of neural-network confidence change under increasing distribution shift?

### Work completed
- Created GitHub repository
- Set up Python virtual environment
- Installed ML dependencies
- Verified PyTorch environment


## CIFAR-10 dataset exploration - 3:50 pm

- Downloaded CIFAR-10 successfully.
- Training set: 50,000 images
- Test set: 10,000 images
- Image size: 32 × 32 pixels
- Channels: RGB
- Number of classes: 10
- Data is loaded in batches of 64 images
- Pixel values are converted to the range 0.0–1.0

### Observation

The low image resolution makes some classes visually difficult to distinguish.
This may become relevant when investigating model reliability under image corruptions and distribution shift.


### GPU acceleration setup - 5:30 pm

Configured ROCm 10 for the AMD Radeon RX 9060 XT.

PyTorch initially used a CUDA build intended for NVIDIA GPUs, so I replaced it with a ROCm-compatible version.

Verification results:

- PyTorch: 2.13.0+rocm10.0.0
- GPU detected: AMD Radeon RX 9060 XT
- GPU acceleration available: True
- ROCm/HIP backend successfully detected

### What I learned

CUDA is specific to NVIDIA GPUs, while AMD GPUs use ROCm/HIP.

PyTorch still uses the `torch.cuda` API for many GPU operations when running with ROCm, so the same training code can often work on both NVIDIA and AMD hardware.

### Current project state

CIFAR-10 can now be loaded and visualized successfully. The development environment is ready for GPU-accelerated model training.

### Next step

Design and implement the first baseline CNN for CIFAR-10.