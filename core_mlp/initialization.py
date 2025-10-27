import torch
from typing import Tuple

def _calculate_fan_in_and_fan_out_(tensor:torch.Tensor)->Tuple[int,int]:
    if tensor.ndimension() < 2:
        raise ValueError("Fan in and fan out require atleast 2 dimensions")
    
    #tensor = [n_out,n_in] --> means = tensor.size(0) = n_out, tensor.size(1) = n_in
    input_num_fmaps = tensor.size(1)
    output_num_fmaps = tensor.size(0)

    receptive_field_size = 1
    if tensor.ndimension() > 2:
        receptive_field_size = tensor[0][0].numel()
    return input_num_fmaps*receptive_field_size, output_num_fmaps*receptive_field_size

def xavior_uniform(tensor: torch.Tensor,gain: float = 1.0):
    fan_in,fan_out = _calculate_fan_in_and_fan_out_(tensor)
    std = gain * (2.0/(fan_in + fan_out))**0.5
    a = (3.0**0.5 )*std
    with torch.no_grad():
        return tensor.uniform_(-a,a)
    
def he_normal(tensor: torch.Tensor,alpha: float = 0.0,mode: str = "fan_in"):
    fan_in,fan_out = _calculate_fan_in_and_fan_out_(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    std = (2.0/(1 + alpha**2)/fan)**0.5   # alpha is the constant in Leaky ReLU(alpha*x) which avoids making negative inputs as 0(dead)
    with torch.no_grad():
        return tensor.normal_(0,std)
    