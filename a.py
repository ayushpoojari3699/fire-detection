import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print("cuDNN Version:", torch.backends.cudnn.version())
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")

check_cuda()
