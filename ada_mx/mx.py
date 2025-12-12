import torch
from typing import Tuple


# converts a fp16 tensor into fp8(e4m3) tensor and fp8(e8m0) scaling factor 
def quantize(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pass



# converts a row major scaling factor tensor into block layout
# https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout 
def to_block(t: torch.Tensor) -> torch.Tensor:
    pass

