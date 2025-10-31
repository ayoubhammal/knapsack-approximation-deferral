import random

import numpy as np
import torch 

def get_gpu_memory() -> str:
    output = []
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(device)
            output.append(f"GPU-{device}: {free / 1024**3:.2f} GB free of {total / 1024**3:.2f} GB")
    return "\n".join(output)

def initialize_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
