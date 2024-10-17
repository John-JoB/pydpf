from .base import Module
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple
import torch
from enum import StrEnum
from .resampling import multinomial
from .utils import multiple_unsqueeze

"""
Module to contain implementations of commonly utilised distributions.
This is to provide a convenient API, similar to torch.distribution.
If a user wants to create a custom distribution it will almost always be easier to subclass Module 
and manually implement sample and log_density methods rather than to subclass Distribution.
"""

#Consistency rules for distribution dimensions
#There is always one data dimension
#Sample should take conditioning variables with dimensions (*B, *C), sample size S and return a Tensor of size (*B,*S,D)
#The specific size C depends on the distribution, it should be the size of a single, un-batched, copy of the conditioning variable.
#log_density takes a sample of size (*B, *S, D) and conditioning variables (*B, *C) and returns the density (*B, *S)


class Distribution(Module, metaclass=ABCMeta):

    class GradientEstimator(StrEnum):
        reparameterisation = 'reparameterisation'
        score = 'score'
        none = 'none'

    @staticmethod
    def get_batch_size(size: Tuple[int, ...], data_size) -> Tuple[int, ...]:
        return tuple(list(size)[:-data_size])

    @staticmethod
    def _unsqueeze_to_size(parameter: torch.Tensor, sample: torch.Tensor, data_dims: int = 1) -> torch.Tensor:
        return multiple_unsqueeze(parameter, sample.dim() - parameter.dim(), -data_dims)

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

    def append_sizes(self, batch_size: Tuple[int], sample_size: Union[Tuple[int], None]):
        if sample_size is None:
            return batch_size
        else:
            return list(batch_size) +  list(sample_size)

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
    def log_density(self, *args, **kwargs) -> torch.Tensor:
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
            self.cholesky_covariance.data = torch.tril(self.cholesky_covariance)
            self.cholesky_covariance.diagonal().mul_(self.cholesky_covariance.diagonal().sign())
        self.inv_cholesky_cov, _ = torch.linalg.inv_ex(self.cholesky_covariance)
        _, self.half_log_det_cov = torch.linalg.slogdet(self.inv_cholesky_cov)

    def update(self):
        super().update()
        self._update()

    def _sample(self, sample_size: Union[Tuple[int, ...], None] = None) -> torch.Tensor:
        if sample_size is None:
            true_sample_size = self.mean.size()
        else:
            true_sample_size = (*sample_size, self.mean.size(-1))
        output = torch.normal(0, 1, device=self.device, size= true_sample_size, generator=self.generator)
        output = self.mean + output @ self.cholesky_covariance.T
        return output

    def log_density(self, sample: torch.Tensor) -> torch.Tensor:
        prefactor = -1/2 * sample.size(-1) - MultivariateGaussian.half_log_2pi - self.half_log_det_cov
        residuals = sample - self.mean
        exponent = -1/2 * torch.sum((residuals @ self.inv_cholesky_cov.T)**2, dim=-1)
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
        self.dist = MultivariateGaussian(device, gradient_estimator, generator, mean = torch.zeros((self.weight.size(-1),), device = self.weight.device), cholesky_covariance=cholesky_covariance)

    def set_weight(self, weight: torch.Tensor) -> None:
        self.weight = weight

    def set_bias(self, bias: torch.Tensor) -> None:
        self.bias = bias

    def set_cholesky_covariance(self, cholesky_covariance: torch.Tensor) -> None:
        self.dist.set_cholesky_covariance(cholesky_covariance)

    def _sample(self, condition_on: torch.Tensor, sample_size: Union[Tuple[int, ...], None] = None) -> torch.Tensor:
        batch_size = self.get_batch_size(condition_on.size(), 1)
        if sample_size is None:
            sample = self.dist._sample(sample_size=batch_size)
        else:
            sample = self.dist._sample(sample_size=(*batch_size, *sample_size))
        means = condition_on @ self.weight + self.bias
        return sample + self._unsqueeze_to_size(means, sample)

    def log_density(self, sample: torch.Tensor, condition_on: torch.Tensor) -> torch.Tensor:
        means = condition_on @ self.weight + self.bias
        return self.dist.log_density(sample - self._unsqueeze_to_size(means, sample))


class KernelMixture(Distribution):
    '''
        Class for KDE mixtures.
    '''

    def __init__(self, device: Union[str, torch.device], gradient_estimator: str, generator: Union[torch.Generator, None], kernel: Distribution):
        super().__init__(device, gradient_estimator, generator)
        self.kernel = kernel
        self.mn_sampler = multinomial(generator)

    def _sample(self, sample_size: Union[Tuple[int], None], loc: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        #Multinomial resampling is sampling from a KDE with a dirac kernel.
        sampled_locs, _, _ = self.mn_sampler(loc, weight)
        batch_size = self.get_batch_size(sampled_locs, 1)
        if sample_size is None:
            sample = self.kernel._sample(sample_size=batch_size)
        else:
            sample = self.kernel._sample(sample_size=(*batch_size, *sample_size))
        return loc + sample

    def log_density(self, sample: torch.Tensor, loc: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        densities = self.kernel.log_density(sample.unsqueeze(-2) - self._unsqueeze_to_size(loc, sample.unsqueeze(-2), 2))
        return torch.logsumexp(densities + weight, dim=-1)
