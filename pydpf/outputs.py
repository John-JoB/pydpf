import torch
from typing import Callable
def filtering_mean(function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x):
    def _filtering_mean(state: torch.Tensor, norm_weights: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.einsum('ij..., ij -> i... ', function(state), torch.exp(norm_weights))
    return _filtering_mean

def MSE_loss(prediction: torch.Tensor, ground_truth: torch.Tensor):
    return torch.sum(torch.mean((prediction - ground_truth) ** 2, dim=(0,1)))




