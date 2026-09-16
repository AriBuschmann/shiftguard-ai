# Defining the Convolutional Neural Network (CNN)

"""
Total data flow of the CNN:

RGB-Image
3 × 32 × 32
    ↓
Conv1
32 × 32 × 32
    ↓
ReLU
    ↓
MaxPool
32 × 16 × 16
    ↓
Conv2
64 × 16 × 16
    ↓
ReLU
    ↓
MaxPool
64 × 8 × 8
    ↓
Flatten
4096
    ↓
Linear
128
    ↓
ReLU
    ↓
Linear
10
    ↓
10 Logits
"""

import torch
import torch.nn as nn # contains the modules for neural networks


class BaselineCNN(nn.Module): # inherits from PyTorch's base class for neural networks
    def __init__(self):
        super().__init__()

        # first convolution layer --> simple features
        self.conv1 = nn.Conv2d(
            in_channels=3, # RGB = 3 colours
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # second convolution layer --> learns more complex patterns from the features produced by conv1 (therefore out_channels=64)
        self.conv2 = nn.Conv2d(
            in_channels=32, # feature maps from conv1
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU() # creates non-linearity
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # only keeps the biggest value out of 4

        self.fc1 = nn.Linear(64 * 8 * 8, 128) # receives 4096 inputs and creates 128 outputs
        self.fc2 = nn.Linear(128, 10) # 10 represents the number of classes

    def forward(self, x):
        # input shape: [number of images, colour channels, pixel height, pixel width] = [batch_size, 3, 32, 32]

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # [batch_size, 32, 16, 16]

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # [batch_size, 64, 8, 8]

        x = torch.flatten(x, start_dim=1) # switching from convolutional layers to linear layers --> batch-dimension is not changed
        # [batch_size, 4096]

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        # output shape: [batch_size, 10]

        return x # 10 values (logits, not probabilities) per picture, the largest logit determines the most probable class


if __name__ == "__main__":
    model = BaselineCNN()
    dummy_images = torch.randn(4, 3, 32, 32) # fake batch of 4 CIFAR-10 images for testing
    outputs = model(dummy_images)

    print(model)
    print("Input shape:", dummy_images.shape)
    print("Output shape:", outputs.shape)
    assert outputs.shape == (4, 10)
    print("Model test successful!")
