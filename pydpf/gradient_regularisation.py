"""
This Module implements methods for modifying the gradient of the particles. This is most often done for stability.
"""

from .filtering import SIS
from typing import Union, Type, Tuple, Any
from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
from .custom_types import ImportanceKernelLikelihood
from .base import Module
from .utils import doc_function




class GradientRegularisedSIS(SIS, metaclass=ABCMeta):

    """
    Given an SIS algorithm, this class defines a new SIS algorithm with some regularising function on the parameters.
    The new algorithm uses the same underlying modules as the original SIS algorithm, but leaves the original unchanged.
    The gradient regularising function can take hyperparameters, but it is never possible to differentiate w.r.t these hyperparameters.
    """

    @abstractmethod
    def make_grad_fun(self, *args, **kwargs):
        raise NotImplementedError('Method not implemented')

    @staticmethod
    def make_prop(base_prop: ImportanceKernelLikelihood, grad_fun: torch.autograd.Function):
        class new_proposal(Module):
            def __init__(self):
                super().__init__()
                self.base_prop = base_prop

            def forward(self, state: Tensor, weight: Tensor, data: Tensor, time: int) -> Tuple[Tensor, Tensor, Tensor]:
                new_state, new_weight, likelihood = base_prop(state, weight, data, time)
                new_state, new_weight, likelihood = grad_fun.apply(new_state, new_weight, likelihood, state, weight)
                return new_state, new_weight, likelihood
        return new_proposal()

    def __init__(self, base_SIS: SIS, *args, **kwargs):
        grad_fun = self.make_grad_fun(*args, **kwargs)
        super().__init__(base_SIS.initial_proposal, self.make_prop(base_SIS.proposal, grad_fun))


class ClipByElement(GradientRegularisedSIS):

    @doc_function
    def __init__(self, base_SIS: SIS, clip_threshold:float):
        """
        Clips the per-element gradient of the loss due to the state/weights to some constant value at each time-step.

        Parameters
        ----------
        base_SIS : SIS
        The sequential importance sampler to modify the gradient of.

        clip_threshold: float
        The threshold above which to clip.
        """
        pass

    def make_grad_fun(self, clip_threshold:float):
        class GradFun(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, state: Tensor, weight: Tensor, likelihood, prev_state: Tensor, prev_weight: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
                return state, weight, likelihood

            @staticmethod
            def backward(ctx: Any, dstate: Tensor, dweight: Tensor, dLikelihood: Tensor):
                return torch.clip(dstate, clip_threshold, clip_threshold), torch.clip(dweight, -clip_threshold, clip_threshold), dLikelihood, None, None
        return GradFun


class ClipByNorm(GradientRegularisedSIS):
    """
    Clips the gradient of the particles and their weights to some constant value.

    Parameters
    ----------
    base_SIS : SIS
    The sequential importance sampler to modify the gradient of.

    clip_threshold: float
    The threshold above which to clip.
    """

    def make_grad_fun(self, clip_threshold: float):
        class gradient_fun(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, state: Tensor, weight: Tensor, likelihood, prev_state: Tensor, prev_weight: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
                return state, weight, likelihood

            @staticmethod
            def backward(ctx: Any, dstate: Tensor, dweight: Tensor, dLikelihood: Tensor):
                dstate_norm = torch.linalg.vector_norm(dstate)
                dweight_norm = torch.linalg.norm(dweight)
                if dstate_norm > clip_threshold:
                    dstate = dstate/dstate_norm
                if dweight_norm > clip_threshold:
                    dweight = dweight/dweight_norm
                return dstate, dweight, dLikelihood
        return gradient_fun


class ClipByParticle(GradientRegularisedSIS):
    """
    Clips the norm of the gradient assigned per-particle.

    Parameters
    ----------
    base_SIS : SIS
    The sequential importance sampler to modify the gradient of.

    clip_threshold: float
    The threshold above which to clip.
    """

    def make_grad_fun(self, clip_threshold: float):
        class gradient_fun(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, state: Tensor, weight: Tensor, likelihood, prev_state: Tensor, prev_weight: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
                ctx.save_for_backward(state, weight)
                return state, weight, likelihood

            @staticmethod
            def backward(ctx: Any, dstate: Tensor, dweight: Tensor, dLikelihood: Tensor):
                state, weight = ctx.saved_tensors
                exp_weights = torch.exp(weight).unsqueeze(-1)
                dparticle = dstate/exp_weights + dweight.unsqueeze(-1)/(exp_weights*state)
                mag_dparticle = torch.linalg.vector_norm(dparticle, -1, keepdim=True)/2
                zero_mask = (exp_weights > 1e-7)
                too_big_mask = torch.logical_and(mag_dparticle > clip_threshold, zero_mask)
                dstate = torch.where(zero_mask, torch.where(too_big_mask, dstate/mag_dparticle, dstate), 0.)
                dweight = torch.where(zero_mask.squeeze(), torch.where(too_big_mask.squeeze(), dweight/mag_dparticle.squeeze(), dweight), 0.)
                return dstate, dweight, dLikelihood, None, None
        return gradient_fun