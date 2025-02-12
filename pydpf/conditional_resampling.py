from .base import Module
import torch
from torch import Tensor
from typing import Tuple, Callable
from .custom_types import Resampler, WeightedSample

class ConditionalResampler(Module):
    def __init__(self, resampler:Resampler,  condition:Callable[[Tensor, Tensor], Tensor]):
        """
        A resampling algorithm that conditionally only resamples a subset of the particles from each batch.

        Notes
        -----
        We generally do not recommend conditionally resampling in a batched DPF context. Classically, conditional resampling is motivated by computation time, resampling algorithms are typically expensive compared to the
        other components of the filtering loop. However, in a parallelised setting each batch can be resampled in parallel, and it is likely that at least one batch does not satisfy the condition. For this reason it is likely that
        the overhead of calculating and applying the condition will more than cancel out any time savings.

        Parameters
        ----------
        resampler : Resampler
            The base resampling algorithm to use.

        condition : Callable[[Tensor, Tensor], Tensor]
            A function that takes a weighted population and returns a 1D boolean tensor where True indicates that resampling should be performed for the given batch and False that it should not.
        """
        super().__init__()
        self.resampler = resampler
        self.condition = condition
        self.cache = {}

    def forward(self, state: Tensor, weight: Tensor) -> WeightedSample:
        with torch.no_grad():
            resample_mask = self.condition(state, weight)
        if not torch.any(resample_mask):
            return state.clone(), weight.clone()
        masked_state = state[resample_mask]
        masked_weight = weight[resample_mask]
        out_state = state.clone()
        out_weight = weight.clone()
        resampled_state , resampled_weight = self.resampler(masked_state, masked_weight)
        out_state[resample_mask] = resampled_state
        out_weight[resample_mask] = resampled_weight
        self.cache = self.resampler.cache
        self.cache['mask'] = resample_mask
        return out_state, out_weight

def ESS_Condition(threshold):
    def forward(state, weight):
        return (1/torch.sum(torch.exp(2*weight), dim=1)) < threshold*weight.size(1)
    return forward