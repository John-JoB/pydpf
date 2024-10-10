import torch
from typing import Tuple
from .utils import batched_select

def multinomial(state: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sampled_indices = torch.multinomial(torch.exp(weights), weights.size(1), replacement=True).detach()
    return batched_select(state, sampled_indices).detach(), torch.zeros_like(weights), sampled_indices