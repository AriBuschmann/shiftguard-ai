# model training with loss function, backpropagation and optimizer

"""
Complete workflow:

Load CIFAR-10
        ↓
Normalize images
        ↓
64 images per batch
        ↓
Images to RX 9060 XT
        ↓
CNN on RX 9060 XT
        ↓
Forward pass
        ↓
10 logits per image
        ↓
Cross-entropy loss
        ↓
Backpropagation
        ↓
Adam updates weights
        ↓
Next batch
        ↓
After 50,000 images: Loss + Accuracy
        ↓
Repeat 5 times
        ↓
Save trained weights
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import BaselineCNN


BATCH_SIZE = 64
LEARNING_RATE = 0.001 # common starting value for optimizer Adam
NUM_EPOCHS = 5
RANDOM_SEED = 42 # improves reproducibility of random processes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "baseline_cnn.pth"


torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# choosing GPU / CPU for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

data_generator = torch.Generator()
data_generator.manual_seed(RANDOM_SEED)


# dataset preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),

    # normalised value = (pixel value - mean) / standard deviation
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=data_generator
)

# creating CNN model and moving it to GPU / CPU
model = BaselineCNN().to(device)


# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(
    model.parameters(), # all the trainable parameters of the CNN
    lr=LEARNING_RATE
)


# training
def train():
    print("\nStarting training...\n")

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:

            # Move batch to GPU / CPU
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # remove gradients from the previous batch
            outputs = model(images) # forward pass - it does what I have defined in model.py (forward function) and returns the logits for each image
            loss = criterion(outputs, labels) # calculate loss by comparing the predicted logits with the true class labels
            loss.backward() # backpropagation - calculates the gradients of the loss backwards through the network
            optimizer.step() # update model weights (is repeated approximately 782 times per epoch)
            running_loss += loss.item() * images.size(0) # total the loss
            predicted_classes = outputs.argmax(dim=1) # determine predicted classes by finding the index of the largest logit for each image
            correct_predictions += (predicted_classes == labels).sum().item() # count correct predictions
            total_samples += labels.size(0) # count processed images

        # calculate statistics for the complete epoch
        average_loss = running_loss / total_samples

        accuracy = (correct_predictions / total_samples) * 100

        # show results for the epoch
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )


def save_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(), # contains the model's trained parameters
        MODEL_PATH
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
    save_model()