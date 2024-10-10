from .base import Module
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple
import torch
from enum import StrEnum

"""
Module to contain implementations of commonly utilised distributions.
This is to provide a convenient API, similar to torch.distribution.
If a user wants to create a custom distribution it will almost always be easier to subclass Module 
and manually implement sample and log_density methods rather than to subclass Distribution.
"""


class Distribution(Module, metaclass=ABCMeta):

    class GradientEstimator(StrEnum):
        reparameterisation = 'reparameterisation'
        score = 'score'
        none = 'none'

    def __init__(self, device: Union[str, torch.device], gradient_estimator: str, generator: Union[torch.Generator, None], *args, **kwargs) -> None:
        super().__init__()
        self.generator = generator
        self.grad_est = self.GradientEstimator(gradient_estimator)
        self.device = device

    def __call__(self, *args, **kwargs) -> None:
        # Do not implement a forward method for a Distribution
        pass

    def forward(self, *args, **kwargs) -> None:
        #Do not implement a forward method for a Distribution
        pass

    @abstractmethod
    def _sample(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError('Sampling not implemented for this distribution')

    def sample(self, *args, **kwargs) -> torch.Tensor:
        output = self._sample(*args, **kwargs)
        if self.training or self.grad_est == self.GradientEstimator.none:
            return output.detach()
        if self.grad_est == self.GradientEstimator.reparameterisation:
            return output
        if self.grad_est == self.GradientEstimator.score:
            log_dens = self.log_density(output, *args, **kwargs)
            return output.detach() * torch.exp(log_dens - log_dens.detach())


    @abstractmethod
    def log_density(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError('Density/Mass function not implemented for this distribution')

    def set_generator(self, generator: Union[torch.Generator, None]):
        self.generator = generator

    def set_gradient_estimator(self, gradient_estimator: str):
        self.grad_est = self.GradientEstimator(gradient_estimator)

    def set_device(self, device):
        self.device = device
        self.to(self.device)


class MultivariateGaussian(Distribution):

    half_log_2pi = 1/2 * torch.log(torch.tensor(2*torch.pi))

    def __init__(self, device: Union[str, torch.device], gradient_estimator: str, generator: Union[None, torch.Generator],  mean: torch.Tensor, cholesky_covariance: torch.Tensor) -> None:
        super().__init__(device, gradient_estimator, generator)
        self.mean = mean
        self.cholesky_covariance = cholesky_covariance
        self._update()

    def _update(self):
        with torch.no_grad():
            self.cholesky_covariance.diagonal().mul_(self.cholesky_covariance.diagonal().sign())
        self.inv_cholesky_cov, _ = torch.linalg.inv_ex(self.cholesky_covariance)
        _, self.half_log_det_cov = torch.linalg.slogdet(self.inv_cholesky_cov)

    def update(self):
        super().update()
        self._update()


    def _sample(self, batch_size: Union[Tuple[int], None], *args, **kwargs) -> torch.Tensor:
        if batch_size is None:
            sample_size = self.mean.size()
        else:
            sample_size = list(batch_size) + list(self.mean.size())
        output = torch.normal(0, 1, device=self.device, size=sample_size, generator=self.generator)
        output = self.mean + torch.einsum('...j, ij', output, self.cholesky_covariance)
        return output

    def log_density(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        prefactor = -1/2 * sample.size(-1) - MultivariateGaussian.half_log_2pi - self.half_log_det_cov
        residuals = sample - self.mean
        exponent = -1/2 * torch.sum(torch.einsum('...i, ij', residuals, self.inv_cholesky_cov) ** 2, dim=-1)
        return prefactor + exponent

    def set_mean(self, mean: torch.Tensor) -> None:
        self.mean = mean

    def set_cholesky_covariance(self, cholesky_covariance: torch.Tensor) -> None:
        self.cholesky_covariance = cholesky_covariance
        self._update()


class LinearGaussian(Distribution):
    def __init__(self, device: Union[str, torch.device], gradient_estimator: str, generator: Union[torch.Generator, None], weight: torch.Tensor, bias: torch.Tensor, cholesky_covariance: torch.Tensor) -> None:
        super().__init__(device, gradient_estimator, generator)
        self.weight = weight
        self.bias = bias
        self.dist = MultivariateGaussian(device, gradient_estimator, generator, mean = torch.empty(1), cholesky_covariance=cholesky_covariance)

    def set_weight(self, weight: torch.Tensor) -> None:
        self.weight = weight

    def set_bias(self, bias: torch.Tensor) -> None:
        self.bias = bias

    def set_cholesky_covariance(self, cholesky_covariance: torch.Tensor) -> None:
        self.dist.set_cholesky_covariance(cholesky_covariance)

    def _sample(self, condition_on: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.dist.set_mean(condition_on @ self.weight + self.bias)
        return self.dist._sample(batch_size=None)

    def log_density(self, sample: torch.Tensor, condition_on: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.dist.set_mean(condition_on @ self.weight + self.bias)
        return self.dist.log_density(sample)
