"""
This Module implements methods for modifying the gradient of the particles. This is most often done for stability.
"""

from .filtering import SIS
from typing import Tuple, Any
import torch
from torch import Tensor




class GradientRegularisedSIS(torch.autograd.Function):

    """
    Given an SIS algorithm, this class defines a new SIS algorithm with some regularising function on the parameters.
    The new algorithm uses the same underlying modules as the original SIS algorithm, but leaves the original unchanged.
    The gradient regularising function can take hyperparameters, but it is never possible to differentiate w.r.t these hyperparameters.
    """

    @staticmethod
    def forward(ctx: Any, state: Tensor, weight: Tensor, prev_state: Tensor, prev_weight: Tensor) -> Any:
        ctx.save_for_backward(state, weight, prev_state, prev_weight)
        return state, weight

    @staticmethod
    def backward(ctx: Any, dstate: Tensor, dweight: Any) -> Tuple[Tensor, Any]:
        pass

def ClipByElement(clip_threshold: float) -> type(GradientRegularisedSIS):
    """
        Clips the per-element gradient of the loss due to the state/weights to some constant value at each time-step.

        Parameters
        ----------
        clip_threshold: float
        The threshold above which to clip.
    """
    class _ClipByElement(GradientRegularisedSIS):
        @staticmethod
        def forward(ctx: Any, state:Tensor, weight:Tensor, prev_state: Tensor, prev_weight: Tensor) -> Tuple[Tensor, Tensor]:
            return state, weight

        @staticmethod
        def backward(ctx: Any, dstate:Tensor, dweight:Any):
            return torch.clip(dstate, clip_threshold, clip_threshold), torch.clip(dweight, -clip_threshold, clip_threshold), None, None

    return _ClipByElement

def ClipByNorm(clip_threshold: float) -> type(GradientRegularisedSIS):
    """
        Clips the gradient of the particles and their weights to some constant value.

        Parameters
        ----------

        clip_threshold: float
        The threshold above which to clip.
    """
    class _ClipByNorm(GradientRegularisedSIS):
        @staticmethod
        def forward(ctx: Any, state: Tensor, weight: Tensor, prev_state: Tensor, prev_weight: Tensor) -> Tuple[Tensor, Tensor]:
            return state, weight

        @staticmethod
        def backward(ctx: Any, dstate: Tensor, dweight: Tensor):
            dstate_norm = torch.linalg.vector_norm(dstate)
            dweight_norm = torch.linalg.norm(dweight)
            if dstate_norm > clip_threshold:
                dstate = dstate/dstate_norm
            if dweight_norm > clip_threshold:
                dweight = dweight/dweight_norm
            return dstate, dweight , None, None
    return _ClipByNorm


def ClipByParticle(clip_threshold: float) -> type(GradientRegularisedSIS):
    """
    Clips the norm of the gradient assigned per-particle.

    Parameters
    ----------

    clip_threshold: float
    The threshold above which to clip.
    """
    class _ClipByParticle(GradientRegularisedSIS):
        @staticmethod
        def forward(ctx: Any, state: Tensor, weight: Tensor, prev_state: Tensor, prev_weight: Tensor) -> Tuple[Tensor, Tensor]:
            ctx.save_for_backward(state, weight)
            return state, weight

        @staticmethod
        def backward(ctx: Any, dstate: Tensor, dweight: Tensor):
            state, weight = ctx.saved_tensors
            exp_weights = torch.exp(weight).unsqueeze(-1)
            dparticle = dstate/exp_weights + dweight.unsqueeze(-1)/(exp_weights*state)
            mag_dparticle = torch.linalg.vector_norm(dparticle, -1, keepdim=True)/2
            zero_mask = (exp_weights > 1e-7)
            too_big_mask = torch.logical_and(mag_dparticle > clip_threshold, zero_mask)
            dstate = torch.where(zero_mask, torch.where(too_big_mask, dstate/mag_dparticle, dstate), 0.)
            dweight = torch.where(zero_mask.squeeze(), torch.where(too_big_mask.squeeze(), dweight/mag_dparticle.squeeze(), dweight), 0.)
            return dstate, dweight, None, None
    return _ClipByParticle