import torch
from torch import Tensor
from typing import Callable, Tuple
from .base import Module
from .distributions import KernelMixture

class FilteringMean(Module):
    def __init__(self,function: Callable[[Tensor], Tensor] = lambda x: x):
        super().__init__()
        self.function = function

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time) -> Tensor:
        return torch.einsum('ij..., ij -> i... ', self.function(state), torch.exp(norm_weights))

class MSE_Loss(Module):
    def __init__(self, ground_truth: Tensor, function: Callable[[Tensor], Tensor] = lambda x: x):
        super().__init__()
        self.ground_truth = ground_truth
        self.mean = FilteringMean(function)

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time):
        filter_mean = self.mean(state, norm_weights, likelihood, data, time)
        return torch.sum(torch.mean((self.ground_truth[time] - filter_mean) ** 2, dim=0))

class LogLikelihoodFactors(Module):
    def __init__(self):
        super().__init__()

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time) -> Tensor:
        return likelihood

class ElBO_Loss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time):
        return -torch.mean(likelihood)


class PredictiveMean(Module):
    def __init__(self, ground_truth: Tensor, prediction_kernel: Callable[[Tensor, Tensor, Tensor, int], Tuple[Tensor, Tensor]], lag: int, function: Callable[[Tensor], Tensor] = lambda x: x):
        super().__init__()
        self.prediction_kernel = prediction_kernel
        self.ground_truth = ground_truth
        self.lag = lag
        self.function = function

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time):
        prediction, new_weights = self.prediction_kernel(state, norm_weights, data[time:time+self.lag].squeeze(), time)
        return torch.einsum('ij...,ij...->i...', self.function(prediction), torch.exp(new_weights))


class PredictionMSE_Loss(Module):
    def __init__(self, ground_truth: Tensor, prediction_kernel: Callable[[Tensor, Tensor, Tensor, int], Tuple[Tensor, Tensor]], lag: int, function: Callable[[Tensor], Tensor] = lambda x: x):
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
        super().__init__()
        self.KDE = kernel
        self.ground_truth = ground_truth

    def forward(self, state: Tensor, norm_weights: Tensor, likelihood, data, time):
        return -self.KDE.log_density(self.ground_truth[time], state, norm_weights)