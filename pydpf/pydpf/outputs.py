import torch
from torch import Tensor
from typing import Callable
from .base import Module
from .distributions import KernelMixture

class FilteringMean(Module):
    """Get an estimate of the filtering mean of a function of the latent state.

    Parameters
    ----------
    function: Callable[[Tensor], Tensor]. Default identity function.
        The function of the latent state to estimate.

    Notes
    -----
    function can be (explicitly) a function of the state only.
    """
    def __init__(self,function: Callable[[Tensor], Tensor] = lambda x: x):
        super().__init__()
        self.function = function

    def forward(self, *, state: Tensor, weight: Tensor, **data) -> Tensor:
        """Get the filtering mean for a given time step."""
        return torch.einsum('ij..., ij -> i... ', self.function(state), torch.exp(weight))

class MSE_Loss(Module):
    """Get the per-timestep mean squared error of a function of the latent state compared to ground truth over a batch of filters.

    Parameters
    ----------
    function: Callable[[Tensor], Tensor]. Default identity function.
        The function of the latent state to estimate.
    """
    def __init__(self,function: Callable[[Tensor], Tensor] = lambda x: x):

        super().__init__()
        self.mean = FilteringMean(function)

    def forward(self, *, state: Tensor, weight: Tensor, ground_truth, **data):
        """Get the MSE loss for a time-step"""
        filter_mean = self.mean(state = state, weight = weight)
        return torch.mean(torch.sum((ground_truth - filter_mean) ** 2, dim=-1))

class LogLikelihoodFactors(Module):

    def __init__(self):
        """Get the log observation likelihood factor for each time step. Such that an estimate of the log likelihood over the trajectory may be given by a sum of these factors."""
        super().__init__()

    def forward(self, likelihood, **kwargs) -> Tensor:
        """Get the log likelihood factor for a time step."""
        return likelihood

class ElBO_Loss(Module):
    """Get the factors of the ELBO loss per-timestep for a batch of filters. The complete ELBO loss may be given by a sum of these factors. This is the negative of the ELBO.

    Notes
    -----
    As an average of the log-likelihood, the ELBO estimates a Jensens' inequality lower bound to the mean log of the likelihood.
    The SMC ELBO is not exactly analogous to the variation auto-encoder ELBO, see [1]_ for more information.

    References
    ----------
    .. [1] Naesseth C, Linderman S, Ranganath R, Blei D (2018). “Variational sequential monte carlo.” In Proc. Int. Conf. Art. Int. and Stat. (AISTATS), pp. 968–977. PMLR, Lanzarote, Canary Islands.
    """
    def __init__(self):
        super().__init__()

    def forward(self, likelihood, **kwargs):
        """Get the negative ELBO factor for a time step."""
        return -torch.mean(likelihood)


class PredictiveMean(Module):
    """Predict the state n steps ahead.

    Parameters
    ----------
    prediction_kernel: Module
        A function to importance sample from the predictive distribution n-steps ahead. Typically, this will entail be applying the bootstrap proposal n-times.
    lag: int
        How many steps ahead the prediction is being made.
    function: Callable[[Tensor], Tensor]
        The function of the latent state to estimate.
    """
    def __init__(self, prediction_kernel: Callable, lag: int, function: Callable[[Tensor], Tensor] = lambda x: x):
        super().__init__()
        self.prediction_kernel = prediction_kernel
        self.lag = lag
        self.function = function

    def forward(self, *, state: Tensor, norm_weights: Tensor, data, time, **kwargs):
        prediction, new_weights = self.prediction_kernel(state, norm_weights, data[time:time+self.lag].squeeze(), time)
        return torch.einsum('ij...,ij...->i...', self.function(prediction), torch.exp(new_weights))


class NegLogDataLikelihood_Loss(Module):
    """Get the negative log data likelihood per-timestep under a kernel density estimator.
    This function applies a kernel density estimator over the particles and calculates the log likelihood of the ground truth given the KDE.

    Parameters
    ----------
    kernel: KernelMixture
        The kernel density estimator.
    """
    def __init__(self, kernel: KernelMixture):

        super().__init__()
        self.KDE = kernel


    def forward(self, *, state: Tensor, weight: Tensor, ground_truth, **kwargs):
        """Get the negative log data likelihood factor under the a KDE and given a time-step"""
        return -self.KDE.log_density(ground_truth, state, weight)

class State(Module):
    """Module that returns the particle states when passed as an `aggregation_function`."""

    def __init__(self):
      super().__init__()

    def forward(self, *, state: Tensor, **data):
        """Return the particle states"""
        return state


class Weight(Module):
    """Module that returns the particle weights when passed as an `aggregation_function`."""
    def __init__(self):
        super().__init__()

    def forward(self, *, weight: Tensor, **data):
        """Return the particle weights"""
        return weight