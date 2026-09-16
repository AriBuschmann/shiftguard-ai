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

### Next step
Load and inspect CIFAR-10 and build the first baseline classifier.


## CIFAR-10 dataset exploration 3:50 pm

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