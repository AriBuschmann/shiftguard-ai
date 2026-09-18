# Evaluation of the trained baseline CNN on clean CIFAR-10 test data

"""
Complete workflow:

Load trained CNN weights
        ↓
Load 10,000 CIFAR-10 test images
        ↓
Apply the same normalization as during training
        ↓
Images to GPU / CPU (RX 9060 XT)
        ↓
Model in evaluation mode
        ↓
Forward pass
        ↓
10 logits per image
        ↓
Calculate test loss
        ↓
Softmax probabilities
        ↓
Predicted class + confidence
        ↓
Measure accuracy and confidence
        ↓
Establish clean baseline for ShiftGuard
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import BaselineCNN


BATCH_SIZE = 64


# project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_cnn.pth"


# choosing GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# dataset preprocessing
# must be identical to the normalization used during training (train.py), so the comparison is fair
transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


# CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False, # images were not used during training
    download=False,
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# create CNN architecture
model = BaselineCNN().to(device)


# check whether trained model weights (baseline_cnn.pth) exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Run src/train.py before evaluating the model."
    )


# load trained model weights
state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

# weights are loaded into the newly created BaselineCNN
model.load_state_dict(state_dict)


# loss function
criterion = nn.CrossEntropyLoss()


def evaluate():
    # switch model from training mode to evaluation mode
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    total_confidence = 0.0

    correct_confidence = 0.0
    incorrect_confidence = 0.0

    correct_count = 0
    incorrect_count = 0

    print("\nEvaluating baseline CNN...\n")

    # gradients are not needed during evaluation
    with torch.no_grad():

        for images, labels in test_loader:

            # move batch to GPU / CPU
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images) # forward pass
            loss = criterion(outputs, labels) # calculate loss: compare predicted logits with true class labels
            total_loss += loss.item() * images.size(0) # exactly the same as in train.py
            probabilities = torch.softmax(outputs, dim=1) # convert logits into probabilities
            confidences, predicted_classes = probabilities.max(dim=1) # highest probability = predicted class and confidence
            correct_mask = predicted_classes == labels # determine which predictions are correct
            incorrect_mask = ~correct_mask # values are inverted: correct predictions are False, incorrect predictions are True
            batch_size = labels.size(0) # number of samples in the current batch
            total_samples += batch_size
            correct_predictions += correct_mask.sum().item()

            # accumulate confidence over all predictions
            total_confidence += confidences.sum().item()

            # accumulate confidence separately for correct predictions
            correct_confidence += confidences[correct_mask].sum().item()
            correct_count += correct_mask.sum().item()

            # accumulate confidence separately for incorrect predictions
            incorrect_confidence += confidences[incorrect_mask].sum().item()
            incorrect_count += incorrect_mask.sum().item()


    # calculate final metrics
    average_loss = total_loss / total_samples

    accuracy = (
        correct_predictions / total_samples
    ) * 100

    mean_confidence = (
        total_confidence / total_samples
    ) * 100

    mean_correct_confidence = (
        correct_confidence / correct_count
    ) * 100 if correct_count > 0 else 0.0

    mean_incorrect_confidence = (
        incorrect_confidence / incorrect_count
    ) * 100 if incorrect_count > 0 else 0.0


    # show clean baseline results
    print(f"Test Loss: {average_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%") # accuracy answers "How many predictions were correct?"
    print(f"Mean Confidence: {mean_confidence:.2f}%") # confidence answers "How confident was the model in its predictions?"
    print(f"Mean Confidence (Correct): {mean_correct_confidence:.2f}%")
    print(f"Mean Confidence (Incorrect): {mean_incorrect_confidence:.2f}%")


if __name__ == "__main__":
    evaluate()
