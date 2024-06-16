import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('CUDA device name:', torch.cuda.get_device_name(torch.cuda.current_device()))
    print('CUDA device capability:', torch.cuda.get_device_capability(torch.cuda.current_device()))