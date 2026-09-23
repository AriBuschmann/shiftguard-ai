# Controlled distribution shifts for ShiftGuard AI

"""
Gaussian noise workflow:

PIL image
    ↓
ToTensor()
    ↓
Pixel values in range [0, 1]
    ↓
Add Gaussian noise
    ↓
Clamp values back to [0, 1]
    ↓
Normalize
    ↓
CNN

The noise is deliberately added before CIFAR-10 normalization so that sigma has a clear interpretation relative to the original pixel range.
"""

import torch


# predefined Gaussian noise severity levels
# sigma describes the standard deviation of the noise on images scaled to [0, 1]
GAUSSIAN_NOISE_SIGMAS = {
    0: 0.00,
    1: 0.05,
    2: 0.10,
    3: 0.15,
    4: 0.20,
    5: 0.25,
}


def get_gaussian_noise_sigma(severity):
    if severity not in GAUSSIAN_NOISE_SIGMAS:
        raise ValueError(
            f"Invalid Gaussian noise severity: {severity}. "
            f"Expected one of {list(GAUSSIAN_NOISE_SIGMAS.keys())}."
        )

    return GAUSSIAN_NOISE_SIGMAS[severity] # return the predefined sigma value for a Gaussian noise severity level


class GaussianNoise:

    # configure transformation
    def __init__(self, severity):
        self.severity = severity
        self.sigma = get_gaussian_noise_sigma(severity)

    # apply Gaussian noise to one image tensor
    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "GaussianNoise expects a PyTorch tensor as input. "
                "Apply transforms.ToTensor() before GaussianNoise."
            )

        if not image.is_floating_point():
            raise TypeError(
                "GaussianNoise expects a floating-point tensor."
            )

        if image.ndim != 3:
            raise ValueError(
                f"Expected image shape [channels, height, width], "
                f"but received shape {tuple(image.shape)}."
            )

        if image.min().item() < 0.0 or image.max().item() > 1.0:
            raise ValueError(
                "GaussianNoise expects pixel values in the range [0, 1]. "
                "Apply it after ToTensor() and before Normalize()."
    )

        if self.sigma == 0.0: # severity 0 represents the unchanged clean image
            return image.clone()

        noise = torch.randn_like(image) * self.sigma # generate random Gaussian noise with mean 0 and the selected sigma
        noisy_image = image + noise # add noise to the original image

        # keep all pixel values inside the valid [0, 1] range
        noisy_image = torch.clamp(
            noisy_image,
            min=0.0,
            max=1.0
        )

        return noisy_image

    def __repr__(self): # control how the object is displayed as text
        return (
            f"{self.__class__.__name__}("
            f"severity={self.severity}, sigma={self.sigma:.2f})"
        )

# apply Gaussian noise to one real CIFAR-10 test image and verify that the output shape and pixel range remain valid
def _run_smoke_test():

    from pathlib import Path
    from torchvision import datasets, transforms

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    # load one original CIFAR-10 test image without normalization
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False
    )

    pil_image, _ = dataset[0]
    image = transforms.ToTensor()(pil_image)
    torch.manual_seed(42) # fixed seed makes this small test reproducible
    shift = GaussianNoise(severity=3)
    noisy_image = shift(image)

    print("Gaussian noise test")
    print(f"Shift: {shift}")
    print(f"Original shape: {image.shape}")
    print(f"Noisy shape: {noisy_image.shape}")
    print(f"Original range: {image.min().item():.4f} - {image.max().item():.4f}")
    print(
        f"Noisy range: "
        f"{noisy_image.min().item():.4f} - "
        f"{noisy_image.max().item():.4f}"
    )

    # verify that the transformation does not change the image dimensions
    assert noisy_image.shape == image.shape

    # verify that clamping keeps all pixels inside the valid range
    assert noisy_image.min().item() >= 0.0
    assert noisy_image.max().item() <= 1.0

    print("Gaussian noise test successful!")


if __name__ == "__main__":
    _run_smoke_test()