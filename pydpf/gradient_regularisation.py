"""This Module implements methods for modifying the gradient of the particles. This is most often done for stability."""

from typing import Tuple, Any
import torch
from torch import Tensor

def ClipByElement(clip_threshold: float) -> torch.autograd.Function:
    """Clips the per-element gradient of the loss due to the state/weights to some constant value at each time-step.

        Parameters
        ----------
        clip_threshold: float
        The threshold above which to clip.
    """
    class _ClipByElement(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, state:Tensor, weight:Tensor, time_data: dict) -> Tuple[Tensor, Tensor]:
            return state, weight

        @staticmethod
        def backward(ctx: Any, dstate:Tensor, dweight:Any):
            return torch.clip(dstate, clip_threshold, clip_threshold), torch.clip(dweight, -clip_threshold, clip_threshold), None

    return _ClipByElement

def ClipByNorm(clip_threshold: float) -> torch.autograd.Function:
    """Clips the gradient of the particles and their weights to some constant value.

        Parameters
        ----------

        clip_threshold: float
        The threshold above which to clip.
    """
    class _ClipByNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, state: Tensor, weight: Tensor, time_data: dict) -> Tuple[Tensor, Tensor]:
            return state, weight

        @staticmethod
        def backward(ctx: Any, dstate: Tensor, dweight: Tensor):
            dstate_norm = torch.linalg.vector_norm(dstate)
            dweight_norm = torch.linalg.norm(dweight)
            if dstate_norm > clip_threshold:
                dstate = dstate/dstate_norm
            if dweight_norm > clip_threshold:
                dweight = dweight/dweight_norm
            return dstate, dweight , None
    return _ClipByNorm


def ClipByParticle(clip_threshold: float) -> torch.autograd.Function:
    """Clips the norm of the gradient assigned per-particle.

    Parameters
    ----------

    clip_threshold: float
    The threshold above which to clip.
    """
    class _ClipByParticle(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, state: Tensor, weight: Tensor, time_data: dict) -> Tuple[Tensor, Tensor]:
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
            return dstate, dweight, None
    return _ClipByParticle

def TBPTT(propagation_length: int):
    """Truncates the computation graph every ``propagation_length`` time-steps.

    Parameters
    ----------
    propagation_length: int
        The amount of time-steps after which to halt the gradient flow.
    """

    #Truncating can be done more efficiently on the forward pass, so create a class that looks like an autograd function
    #to the filtering script but only acts on the forward pass.
    class _TBPTT:
        @staticmethod
        def apply(state: Tensor, weight: Tensor, time_data: dict):
            if time_data['t'] % propagation_length == 0:
                return state.detach(), weight.detach()
            return state, weight

    return _TBPTT

