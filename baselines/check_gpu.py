# check_gpu.py
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Device 0 name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)
else:
    print("No CUDA device detected — check your driver or docker run command.")

