from .base import Module
import torch
from torch import Tensor
from typing import Tuple, Callable


class ConditionalResampler(Module):
    def __init__(self, resampler:Module,  condition:Callable):
        """
        A resampling algorithm that conditionally only resamples a subset of the particles from each batch.

        Parameters
        ----------
        resampler : Resampler
            The base resampling algorithm to use.
        condition : Callable
            A function that take the time-step data and returns a 1D boolean tensor where True indicates that resampling should be performed for the given batch and False that it should not.

        Notes
        -----
        We generally do not recommend conditionally resampling in a batched DPF context. Classically, conditional resampling is motivated by computation time, resampling algorithms are typically expensive compared to the
        other components of the filtering loop. However, in a parallelised setting each batch can be resampled in parallel, and it is likely that at least one batch does not satisfy the condition.
        For this reason it is likely that the overhead of calculating and applying the condition will more than cancel out any time savings.
        """
        super().__init__()
        self.resampler = resampler
        self.condition = condition
        self.cache = {}

    def _mask_data(self, mask, **data):
        masked_data = {}
        for k, v in data.items():
            masked_data[k] = v[mask]
        return masked_data


    def forward(self, state: Tensor, weight: Tensor, **data) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            resample_mask = self.condition(state=state, weight=weight, **data)
        if not torch.any(resample_mask):
            return state.clone(), weight.clone()
        masked_state = state[resample_mask]
        masked_weight = weight[resample_mask]
        out_state = state.clone()
        out_weight = weight.clone()
        masked_data = self._mask_data(resample_mask, **data)
        resampled_state , resampled_weight = self.resampler(state=masked_state, weight=masked_weight, **masked_data)
        out_state[resample_mask] = resampled_state
        out_weight[resample_mask] = resampled_weight
        self.cache = self.resampler.cache
        t = weight.clone()
        t[resample_mask] = self.cache['used_weight'][resample_mask]
        self.cache['used_weight'] = t
        self.cache['mask'] = resample_mask
        return out_state, out_weight

def ESS_Condition(threshold):
    r"""The effective sample size criterion.


    Parameters
    ----------
    threshold: float
        The relative ESS threshold below which resampling should be performed.

    Returns
    -------
    resample_mask:
        A mask tensor that is true if a filter in the batch should be resampled.

    Notes
    -----
    The relative ESS is equal to:

    .. math:: \frac{1}{K\sum^{K}_{k=1}(w^{(k)}_{t})^{2}}

    See [1]_.

    References
    ----------
    .. [1] Elivra V, Martino L, Robert CP (2022), "Rethinking the effective sample size", Int. Stat. Review, 90: 525–550.
    """
    def forward(weight, **data):
        return (1/torch.sum(torch.exp(2*weight), dim=1)) < threshold*weight.size(1)
    return forward