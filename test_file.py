import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (compiled):", torch.version.cuda)
print("CUDA version (runtime):", torch._C._cuda_getCompiledVersion() if torch.cuda.is_available() else "N/A")
print("GPU detected:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
