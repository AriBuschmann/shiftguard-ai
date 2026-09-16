import torch
import numpy as np
import sklearn

print("ShiftGuard AI environment is ready!")
print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)
print("scikit-learn version:", sklearn.__version__)
print("ROCm/HIP version:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))