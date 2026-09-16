import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.ToTensor(), # transformation to numerical values (PyTorch tensor)
])

# training dataset of 60.000 pictures bvbvbvbvbv
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform # every image is converted to a tensor
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False, # test images
    download=True,
    transform=transform
)

# creating Mini-Batches
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True # randomizing the order of the images so that the model does not learn the order of the images
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False # order is unimportant
)


print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Classes:", train_dataset.classes)

# iterator to get the batches (with images and labels) one after another
images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
print("Pixel range:", images.min().item(), "-", images.max().item()) 