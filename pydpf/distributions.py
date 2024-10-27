from .base import Module, cached_property, constrained_parameter
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple, Iterable
import torch
from torch import Tensor
from enum import StrEnum
from .resampling import multinomial
from .utils import multiple_unsqueeze, doc_function

"""
Module to contain implementations of commonly utilised distributions.
This is to provide a convenient API, similar to torch.distribution.
If a user wants to create a custom distribution it will almost always be easier to subclass Module 
and manually implement sample and log_density methods rather than to subclass Distribution.

Distribution samples have the following dimension order: Batch X Samples X Data

Data is always a single dimension and is inferred from the distribution parameters, which must not be batched.

Batch can be any number of batch dimensions, it is inferred from the conditioning variables. For unconditional distributions the batch size
is always 0.

Samples can be any number of dimensions, it is manually supplied when calling Distribution.sample(). When calling Distribution.log_density() it
is inferred as the dimensions of the supplied sample that aren't Batch or Data.

Log_density returns the log density of each datum in a sample in a batch. Rather than reducing over the sample.
"""


class Distribution(Module, metaclass=ABCMeta):
    """
    Abstract base class for all distributions.
    """

    conditional = False

    class GradientEstimator(StrEnum):
        reparameterisation = 'reparameterisation'
        score = 'score'
        none = 'none'

    def check_sample(self, sample):
        if sample.size(-1) != self.dim:
            raise ValueError(f"Sample must have dimension equal to dimensionality of distribution, found {sample.size(-1)} and {self.dim}")
        if sample.device != sample.device:
            raise ValueError(f"Sample must have device equal to device of distribution parameters, found {sample.device} and {self.device}")

    @staticmethod
    def get_batch_size(size: Tuple[int, ...], data_dims:int) -> Tuple[int, ...]:
        """
        Get the size of a tensor excluding the last (data_size) dimensions.

        Parameters
        ----------
        size : Tuple[int}
            The size to extract the batch dimensions from.
        data_dims : int
            The number of non-batch dimensions to extract.

        Returns
        -------
        batch_size : Tuple[int]
            The extracted batch dimensions.
        """
        return tuple(list(size)[:-data_dims])

    @staticmethod
    def _unsqueeze_to_size(parameter: Tensor, sample: Tensor, data_dims: int = 1) -> Tensor:
        """
        Unsqueeze a tensor to the same number of dimensions as another.

        Parameters
        ----------
        parameter: Tensor
            The tenosr to unsqueeze.
        sample: Tensor
            The tensor of the size parameter is to be unsqueezed to.
        data_dims: data_dims
            The number of dimensions from the last to unsqueeze at.

        Returns
        -------
        unsqueezed_parameter: Tensor
            parameter unsqueezed to the required size.

        """
        return multiple_unsqueeze(parameter, sample.dim() - parameter.dim(), -data_dims)

    def __init__(self, gradient_estimator: str, generator: Union[torch.Generator, None], *args, **kwargs) -> None:
        super().__init__()
        self.dim = None
        self.device = None
        self.reparameterisable = False
        self.generator = generator
        try:
            self.grad_est = self.GradientEstimator(gradient_estimator)
        except ValueError:
            raise ValueError(f'gradient_estimator must be one of reparameterisation, score or none. Found {gradient_estimator}.')
        self.dim = 0

    def __call__(self, *args, **kwargs) -> None:
        # Do not implement a forward method for a Distribution
        pass

    def forward(self, *args, **kwargs) -> None:
        #Do not implement a forward method for a Distribution
        pass

    @abstractmethod
    def _sample(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError('Sampling not implemented for this distribution')

    def sample(self, *args, **kwargs) -> Tensor:
        output = self._sample(*args, **kwargs)
        if not self.training or self.grad_est == self.GradientEstimator.none or not torch.is_grad_enabled():
            return output.detach()
        if self.grad_est == self.GradientEstimator.reparameterisation:
            if not self.reparameterisable:
                raise ValueError(f'No reparameterisation method exists for this distribution, {type(self)}.\nTry a score based gradient estimator or running without gradient.')
            return output
        if self.grad_est == self.GradientEstimator.score:
            log_dens = self.log_density(output, *args, **kwargs)
            return output.detach() * torch.exp(log_dens - log_dens.detach())

    @abstractmethod
    def log_density(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError('Density/Mass function not implemented for this distribution')


class MultivariateGaussian(Distribution):
    conditional = False

    half_log_2pi = 1/2 * torch.log(torch.tensor(2*torch.pi))

    def __init__(self, mean: Tensor, cholesky_covariance: Tensor, gradient_estimator: str, generator: Union[None, torch.Generator]) -> None:
        """
            A multivariate Gaussian distribution.

            Parameters
            ----------
            mean: Tensor
                1D tensor specifying the mean.
            cholesky_covariance: Tensor
                2D tensor specifying the (lower) Cholesky decomposition of the covariance matrix. If the upper triangular section has non-zero
                values these will be ignored.
            gradient_estimator : str
                The gradient estimator to use.
            generator : Union[torch.Generator, None]
                The generator to control the rng when sampling kernels from the mixture.
        """

        super().__init__(gradient_estimator, generator)
        self.reparameterisable = True
        self.mean = mean
        self.cholesky_covariance_ = cholesky_covariance
        self.dim = mean.size(0)
        self.device = mean.device
        if cholesky_covariance.device != mean.device:
            raise ValueError(f'Mean and Covariance should be on the same device, found {mean.device} and {cholesky_covariance.device}')
        if (cholesky_covariance.size(0) != self.dim) or (cholesky_covariance.size(1) != self.dim):
            raise ValueError(f'Covariance must have the same dimensions as the mean, found {self.dim} and {cholesky_covariance.size()}.')

    @constrained_parameter
    def cholesky_covariance(self) -> Tuple[Tensor, Tensor]:
        tril = torch.tril(self.cholesky_covariance_)
        return self.cholesky_covariance_, tril * tril.diagonal().sign()

    @cached_property
    def inv_cholesky_cov(self):
        return torch.linalg.inv_ex(self.cholesky_covariance)[0]

    @cached_property
    def half_log_det_cov(self):
        return torch.linalg.slogdet(self.cholesky_covariance)[1]

    def _sample(self, sample_size: Union[Tuple[int, ...], None] = None) -> Tensor:
        if sample_size is None:
            true_sample_size = self.mean.size()
        else:
            true_sample_size = (*sample_size, self.mean.size(-1))
        output = torch.normal(0, 1, device=self.device, size= true_sample_size, generator=self.generator)
        output = self.mean + output @ self.cholesky_covariance.T
        return output


    @doc_function
    def sample(self, sample_size: Union[Tuple[int, ...], None] = None) -> Tensor:
        """
        Sample a Multivariate Gaussian distribution.

        Parameters
        ----------
        sample_size: Union[Tuple[int, ...], None]
            The size of the sample to draw. Draw a single sample without a sample dimension if None.

        Returns
        -------
        sample: Tensor
            A multivariate Gaussian sample.
        """
        pass

    def log_density(self, sample: Tensor) -> Tensor:
        """
        Returns the log density of a sample

        Parameters
        ----------
        sample: Tensor
            The sample to get the density of.

        Returns
        -------
        sample log_density: Tensor
            The log density of each datum in the sample.
        """
        self.check_sample(sample)
        prefactor = -1/2 * sample.size(-1) - MultivariateGaussian.half_log_2pi - self.half_log_det_cov
        residuals = sample - self.mean
        exponent = -1/2 * torch.sum((residuals @ self.inv_cholesky_cov.T)**2, dim=-1)
        return prefactor + exponent


class LinearGaussian(Distribution):
    conditional = True

    def __init__(self, weight: Tensor, bias: Tensor, cholesky_covariance: Tensor, gradient_estimator: str, generator: Union[torch.Generator, None]) -> None:
        """
            A Gaussian conditional distribution, where the means of the Gaussian are conditional on a supplied variable, X, through the linear map WX + B. Where W is the weights and B is the bias.

            Parameters
            ----------
            weight: Tensor
                2D tensor specifying the weight matrix.
            bias: Tensor
                1D tensor specifying the bias.
            cholesky_covariance: Tensor
                2D tensor specifying the (lower) Cholesky decomposition of the covariance matrix. If the upper triangular section has non-zero
                values these will be ignored.
            gradient_estimator : str
                The gradient estimator to use.
            generator : Union[torch.Generator, None]
                The generator to control the rng when sampling kernels from the mixture.
        """

        super().__init__(gradient_estimator, generator)
        self.weight = weight
        self.bias = bias
        self.dim = self.weight.size(0)
        self.reparameterisable = True
        self.device = weight.device
        if (cholesky_covariance.size(0) != self.dim) or (cholesky_covariance.size(1) != self.dim):
            raise ValueError(f'Covariance must have the same dimensions as the weights first dimension, found {cholesky_covariance.size()} and {weight.size()}.')
        if bias.size(0) != self.dim:
            raise ValueError(f'Bias must have the same dimensions as the weights first dimension, found {bias.size()} and {weight.size()}')
        if cholesky_covariance.device != self.device:
            raise ValueError(f'Weight and Covariance should be on the same device, found {self.device} and {cholesky_covariance.device}')
        if cholesky_covariance.device != self.device:
            raise ValueError(f'Weight and bias should be on the same device, found {self.device} and {bias.device}')
        self.dist = MultivariateGaussian(torch.zeros((self.dim,), device = self.device), cholesky_covariance, gradient_estimator, generator)

    def set_cholesky_covariance(self, cholesky_covariance: Tensor) -> None:
        self.dist.set_cholesky_covariance(cholesky_covariance)

    def _check_conditions(self, condition_on: Tensor) -> None:
        if condition_on.size(-1) != self.weight.size(0):
            raise ValueError(f'condition_on must be batch multiply-able with the distribution weights, found sizes {condition_on.size()} and {self.weight.size()}.')
        if condition_on.device != self.device:
            raise ValueError(f'condition_on should be on the same device as the distribution parameters, found {condition_on} and {self.device}.')

    def _sample(self, condition_on: Tensor, sample_size: Union[Tuple[int, ...], None] = None) -> Tensor:
        self._check_conditions(condition_on)
        batch_size = self.get_batch_size(condition_on.size(), 1)
        if sample_size is None:
            sample = self.dist._sample(sample_size=batch_size)
        else:
            sample = self.dist._sample(sample_size=(*batch_size, *sample_size))
        means = condition_on @ self.weight + self.bias
        return sample + self._unsqueeze_to_size(means, sample)

    @doc_function
    def sample(self, condition_on: Tensor, sample_size: Union[Tuple[int, ...], None] = None) -> Tensor:
        """
        Sample a multivariate Linear Gaussian.
        The means of the Gaussian are calculated as condition_on @ self.weight + self.bias.

        Parameters
        ----------
        condition_on: Tensor
            The vector to condition the distribution on.

        sample_size: Union[Tuple[int, ...], None]
            The size of the sample to draw. If None then a single sample is drawn per batch dimension and no sample dimension is used.

        Returns
        -------
        sample: Tensor
            A sample of a multivariate Linear Gaussian, conditioned on condition_on.

        """
        pass

    def log_density(self, sample: Tensor, condition_on: Tensor) -> Tensor:
        """
        Evaluate the log density of a sample.
        The means of the Gaussian are calculated as condition_on @ self.weight + self.bias.

        Parameters
        ----------
        sample: Tensor
            The sample to get the density of.

        condition_on: Tensor
            The vector to condition the distribution on.

        Returns
        -------
        sample log_density: Tensor
            The log density of each datum in the sample.

        """
        self._check_conditions(condition_on)
        try:
            means = condition_on @ self.weight + self.bias
        except RuntimeError as e:
            raise RuntimeError(f'Failed to apply condition with error: \n {e}. \n This is likely to a mismatch in batch dimensions between the conditioning variables and the sample.')
        return self.dist.log_density(sample - self._unsqueeze_to_size(means, sample))


class CompoundDistribution(Distribution):
    """
        Class for when the desired random variable can be expressed as a set of independent components.
        Primarily included to allow kernels that have different distributions along different dimensions.
    """
    conditional = False

    def __init__(self, distributions: Iterable[Distribution], gradient_estimator: str, generator: Union[torch.Generator, None]):
        """
        A wrapper that concatenates several distributions into a single distribution object.
        Interdependence between the distributions is not permitted.

        Parameters
        ----------
        distributions: Iterable[Distribution]
            An iterable of the distributions that form the components of the compound.
        gradient_estimator : str
            The gradient estimator to use.
        generator : Union[torch.Generator, None]
            The generator to control the rng when sampling kernels from the mixture.
        """
        super().__init__(gradient_estimator, generator)
        self.dists = distributions
        self.reparameterisable = True
        self.dim = 0
        self.device = None
        for dist in self.dists:
            self.dim += dist.dim
            self.reparameterisable = self.reparameterisable and dist.reparameterisable
            if type(dist).conditional:
                raise TypeError(f'None of the component distributions may be conditional, detected {type(dist)} which is.')
            if self.device is not None and self.device != dist.device:
                raise ValueError(f'All component distributions must have all parameters on the same device, found {self.device} and {dist.device}.')
            self.device = dist.device

    @doc_function
    def sample(self, sample_size: Union[Tuple[int], None]) -> Tensor:
        """
        Sample a Compound distribution.
        The sample is the concatenation of samples from the components distributions along the last axis.

        Parameters
        ----------
        sample_size : Union[Tuple[int], None]
            The size of the sample to draw. If None then a single sample is drawn and no sample dimension is used.

        Returns
        -------
        sample: Tensor
            The resulting sample.
        """
        pass

    def _sample(self, sample_size: Union[Tuple[int], None]) -> Tensor:
        samples = []
        for dist in self.dists:
            samples.append(dist.sample(sample_size))
        return torch.cat(samples, dim=-1)

    def log_density(self, sample: Tensor) -> Tensor:
        """
        Evaluate the log density of a sample.

        Parameters
        ----------
        sample: Tensor
            The sample to get the density of.

        Returns
        -------
        sample log_density: Tensor
            The log density of each datum in the sample.

        """
        output = None
        dim_count = 0
        for dist in self.dists:
            if output is None:
                output = dist.log_density(sample[..., dim_count:dim_count+dist.dim])
            else:
                output += dist.log_density(sample[..., dim_count:dim_count+dist.dim])
            dim_count += dist.dim
        return output



class KernelMixture(Distribution):
    '''
        Class for KDE mixtures.
    '''
    conditional = True

    def __init__(self, kernel: Distribution, gradient_estimator: str, generator: Union[torch.Generator, None]):
        """
        Create a kernel density mixture.
        The parameter kernel is an unconditional distribution which will be convolved over the kernel density mixture.
        The resultant distribution is conditional on the locations and weights of the kernels.

        Parameters
        ----------
        kernel: Distribution
            The kernel to use.
        gradient_estimator : str
            The gradient estimator to use.
        generator : Union[torch.Generator, None]
            The generator to control the rng when sampling kernels from the mixture.
        """
        super().__init__(gradient_estimator, generator)
        self.kernel = kernel
        self.reparameterisable = False
        self.dim = self.kernel.dim
        self.device = kernel.device
        self.mn_sampler = multinomial(self.generator)
        if type(self.kernel).conditional:
            raise ValueError(f'The kernel distribution cannot be conditional, detected {type(self.kernel)} which is.')

    def _check_conditions(self, loc: Tensor, weight: Tensor) -> None:
        if loc.device != self.device:
            raise ValueError(f'loc must be on the same device as the distribution, found {loc.device} and {self.device}.')
        if weight.device != self.device:
            raise ValueError(f'weight must be on the same device as the distribution, found {weight.device} and {self.device}.')
        if loc.size(-1) != self.dim:
            raise ValueError(f'It is not permitted for the kernel to have a different dimension as the space it is convolved over, found {loc.size(-1)} and {self.dim}.')
        if weight.size(-1) != loc.size(-2):
            raise ValueError(f'Differing number of kernels locations{loc.size(-2)} and weights {weight.size(-1)}.')

    @doc_function
    def sample(self,  loc: Tensor, weight: Tensor, sample_size: Union[Tuple[int], None]) -> Tensor:
        """
        Sample a KDE mixture

        Parameters
        ----------
        loc : Tensor
            Locations of the Kernels

        weight  : Tensor
            Weights of the Kernels

        sample_size : Union[Tuple[int], None]
            The size of the sample to draw. If None then a single sample is drawn per batch dimension and no sample dimension is used.

        Returns
        -------
        Sample: Tensor
         A sample from the KDE mixture.
        """
        pass

    def _sample(self,  loc: Tensor, weight: Tensor, sample_size: Union[Tuple[int], None]) -> Tensor:
        #Multinomial resampling is sampling from a KDE with a dirac kernel.
        self._check_conditions(loc, weight)
        try:
            sampled_locs, _, _ = self.mn_sampler(loc, weight)
        except Exception as e:
            raise RuntimeError(f'Failed to sample kernels with error: \n {e} \n This is likely due to a mismatch in batch dimensions.')
        batch_size = self.get_batch_size(sampled_locs, 1)
        if sample_size is None:
            sample = self.kernel._sample(sample_size=batch_size)
        else:
            sample = self.kernel._sample(sample_size=(*batch_size, *sample_size))
        return loc + sample

    def log_density(self, sample: Tensor, loc: Tensor, weight: Tensor) -> Tensor:
        """
        Evaluate the log density of a sample.

        Parameters
        ----------
        sample: Tensor
            The sample to get the density of.

        loc : Tensor
            Locations of the Kernels

        weight  : Tensor
            Weights of the Kernels.

        Returns
        -------
        Sample: Tensor
         The log density of each datum in the sample.
        """
        self._check_conditions(loc, weight)
        try:
            densities = self.kernel.log_density(sample.unsqueeze(-2) - self._unsqueeze_to_size(loc, sample.unsqueeze(-2), 2))
            return torch.logsumexp(densities + weight, dim=-1)
        except RuntimeError as e:
            raise RuntimeError(f'Failed to apply condition with error: \n {e} \n This is likely to a mismatch in batch dimensions.')