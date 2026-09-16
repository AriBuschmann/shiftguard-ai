import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

images, labels = next(iter(loader))

# names of the classes
classes = dataset.classes

# area with 16 cells - 4 x 4
fig, axes = plt.subplots(4, 4, figsize=(8,8))

for i, ax in enumerate(axes.flat):
    image = images[i].numpy()

    # PyTorch: [colour channel, height, width]
    # Matplotlib: [height, width, colour channel]
    image = np.transpose(image, (1, 2, 0))

    ax.imshow(image, interpolation="nearest")

    # place the correct class name above each image
    ax.set_title(classes[labels[i].item()])
    ax.axis("off")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/cifar10_samples.png", dpi=150)
