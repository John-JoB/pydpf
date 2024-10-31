import torch
from torch import Tensor
from typing import Callable, Tuple
from .base import Module
from .distributions import KernelMixture
from .custom_types import ImportanceKernel

class FilteringMean(Module):
    def __init__(self,function: Callable[[Tensor], Tensor] = lambda x: x):
        """
        Get an estimate of the filtering mean of a function of the latent state.

        Parameters
        ----------
        function: Callable[[Tensor], Tensor]
            The function of the latent state to estimate.
        """
        super().__init__()
        self.function = function

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time) -> Tensor:
        return torch.einsum('ij..., ij -> i... ', self.function(state), torch.exp(norm_weights))

class MSE_Loss(Module):

    def __init__(self, ground_truth: Tensor, function: Callable[[Tensor], Tensor] = lambda x: x):
        """
        Get the per-timestep mean squared error of a function of the latent state compared to ground truth over a batch of filters.

        Parameters
        ----------
        ground_truth: Tensor
            The ground truth target values.
        function: Callable[[Tensor], Tensor]
            The function of the latent state to estimate.
        """
        super().__init__()
        self.ground_truth = ground_truth
        self.mean = FilteringMean(function)

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time):
        filter_mean = self.mean(state, norm_weights, likelihood, data, time)
        return torch.sum(torch.mean((self.ground_truth[time] - filter_mean) ** 2, dim=0))

class LogLikelihoodFactors(Module):

    def __init__(self):
        """
        Get the log observation likelihood factor for each time step.
        Such that an estimate of the log likelihood over the trajectory may be given by a sum of these factors.
        """
        super().__init__()

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time) -> Tensor:
        return likelihood

class ElBO_Loss(Module):

    def __init__(self):
        """
        Get the factors of the ELBO loss per-timestep for a batch of filters.
        The complete ELBO loss may be given by a sum of these factors.

        Notes
        -----
        As an average of the log-likelihood, the ELBO estimates a Jensens' inequality lower bound to the mean log of the likelihood.
        See C. Naesseth, S. Linderman, R. Ranganath, and D. Blei, 'Variational Sequential Monte-Carlo', 2018 for more information.
        """
        super().__init__()

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time):
        return -torch.mean(likelihood)


class PredictiveMean(Module):

    def __init__(self, prediction_kernel: ImportanceKernel, lag: int, function: Callable[[Tensor], Tensor] = lambda x: x):
        """
        Predict the state n steps ahead.

        Parameters
        ----------
        prediction_kernel: ImportanceKernel
            A function to importance sample from the predictive distribution n-steps ahead. Typically, this will entail be applying the bootstrap proposal n-times.
        lag: int
            How many steps ahead the prediction is being made.
        function: Callable[[Tensor], Tensor]
            The function of the latent state to estimate.
        """
        super().__init__()
        self.prediction_kernel = prediction_kernel
        self.lag = lag
        self.function = function

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time):
        prediction, new_weights = self.prediction_kernel(state, norm_weights, data[time:time+self.lag].squeeze(), time)
        return torch.einsum('ij...,ij...->i...', self.function(prediction), torch.exp(new_weights))


class PredictionMSE_Loss(Module):

    def __init__(self, ground_truth: Tensor, prediction_kernel: ImportanceKernel, lag: int, function: Callable[[Tensor], Tensor] = lambda x: x):
        """
        Get the per-timestep mean squared error of a function of the latent state compared to ground truth over a batch of filters for an n-step ahead prediction.

        Parameters
        ----------
        ground_truth: Tensor
            The ground truth target values. The first ground truth value is assumed to align with time zero.
        prediction_kernel: ImportanceKernel
            A function to importance sample from the predictive distribution n-steps ahead. Typically, this will entail be applying the bootstrap proposal n-times.
        lag: int
            How many steps ahead the prediction is being made.
        function: Callable[[Tensor], Tensor]
            The function of the latent state to estimate.
        """
        super().__init__()
        self.prediction_kernel = prediction_kernel
        self.ground_truth = ground_truth
        self.lag = lag
        self.function = function

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time):
        prediction, new_weights = self.prediction_kernel(state, norm_weights, data[time:time + self.lag].squeeze(), time)
        mean_pred = torch.einsum('ij...,ij...->i...', self.function(prediction), torch.exp(new_weights))
        return torch.sum(torch.mean((self.ground_truth[time+self.lag] - mean_pred) ** 2, dim=0))

        

class NegLogDataLikelihood_Loss(Module):

    def __init__(self, ground_truth: Tensor, kernel: KernelMixture):
        """
        Get the negative log data likelihood per-timestep for a batch of kernel filters.
        This function applies a kernel density estimator over the particles and calculates the log likelihood of the ground truth given the KDE.

        Parameters
        ----------
        ground_truth: Tensor
            The ground truth target values
        kernel: KernelMixture
            The kernel density estimator.
        """
        super().__init__()
        self.KDE = kernel
        self.ground_truth = ground_truth

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time):
        return -self.KDE.log_density(self.ground_truth[time], state, norm_weights)