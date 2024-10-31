
from .base import Module
import torch
from torch import Tensor
from typing import Tuple, Callable
from .custom_types import Resampler, WeightedSample

class ConditionalResampler(Module):
    def __init__(self, resampler:Resampler,  condition:Callable[[Tensor, Tensor], Tensor]):
        super().__init__()
        self.resampler = resampler
        self.condition = condition

    def forward(self, state: Tensor, weight: Tensor) -> WeightedSample:
        with torch.no_grad():
            resample_mask = self.condition(state, weight)
        masked_state = state[resample_mask]
        masked_weight = weight[resample_mask]
        out_state = state.clone()
        out_weight = weight.clone()
        resampled_state , resampled_weight = self.resampler(masked_state, masked_weight)
        out_state[resample_mask] = resampled_state
        out_weight[resample_mask] = resampled_weight
        return out_state, out_weight

def ESS_Condition(threshold):
    def forward(state, weight):
        return (1/torch.sum(torch.exp(2*weight), dim=1)) < threshold*weight.size(1)
    return forward