from .base import Distribution
from torch import Tensor
from typing import Union
import torch
from ..utils import batched_select
from ..resampling import MultinomialResampler
from ..base import Module


class CompoundDistribution(Distribution):
    """A wrapper that concatenates several distributions into a single distribution object.
        Interdependence between the distributions is not permitted.

        Parameters
        ----------
        distributions: list[Distribution|Module, ...]
            An iterable of the distributions that form the components of the compound.
        generator : Unused parameter

        Notes
        -----
        If a distribution in the list distributions is not a Distribution subclass (i.e. it is a custom implementation) then it must have the following attributes::

        dim : the dimension of the distribution.

        sample() : method that takes the parameter `sample_size` and returns a tensor of that size with an extra final dimension of size dim.

        log_density() : method that takes the parameter `sample` and returns a tensor of the same size without the final dimension, assumed to be of size dim.
    """
    conditional = False

    def __init__(self, distributions: list[Distribution|Module, ...], generator):
        super().__init__(torch.default_generator)
        self.dists = distributions
        self.dim = 0
        #register submodules
        self.dists = torch.nn.ModuleList(distributions)
        for dist in self.dists:
            if not isinstance(dist, Distribution):
                if not hasattr(dist, "sample"):
                    raise AttributeError('Custom distributions must have an "sample" method.')
                if not hasattr(dist, "log_density"):
                    raise AttributeError('Custom distributions must have an "log_density" method.')
                if not hasattr(dist, "dim"):
                    raise AttributeError('Custom distributions must have an "dim" attribute.')
            else:
                if type(dist).conditional:
                    raise TypeError(f'None of the component distributions may be conditional, detected {type(dist)} which is.')
                if self.device != dist.device:
                    raise ValueError(f'All component distributions must have all parameters on the same device, found {self.device} and {dist.device}.')
            self.dim += dist.dim


    def sample(self, sample_size: tuple[int, ...]|None) -> Tensor:
        """Sample a Compound distribution.
        The sample is the concatenation of samples from the components distributions along the last axis.

        Parameters
        ----------
        sample_size : tuple[int, ...]|None
            The size of the sample to draw. If None then a single sample is drawn and no sample dimension is used.

        Returns
        -------
        sample: Tensor
            The resulting sample.
        """
        samples = []
        for dist in self.dists:
            samples.append(dist.sample(sample_size))
        return torch.cat(samples, dim=-1)

    def log_density(self, sample: Tensor) -> Tensor:
        """Evaluate the log density of a sample.

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


def sample_only_multinomial(state: Tensor, weights: Tensor, generator) -> Tensor:
    with torch.no_grad():
        sampled_indices = torch.multinomial(torch.exp(weights), weights.size(1), replacement=True, generator=generator).detach()
    return batched_select(state, sampled_indices)


class KernelMixture(Distribution):
    """Create a kernel density mixture.
    The parameter kernel is an unconditional distribution which will be convolved over the kernel density mixture.
    The resultant distribution is conditional on the locations and weights of the kernels.

    Parameters
    ----------
    kernel: list[tuple[str,int], ...]|Distribution
        The kernel to convolve over the particles to form the KDE sampling distribution.
    generator : Union[torch.Generator, None]
        The generator to control the rng when sampling kernels from the mixture.

    Notes
    -----
    If the kernel is not a Distribution subclass (i.e. it is a custom implementation) then it must have the following attributes:

    dim : the dimension of the distribution.

    sample() : method that takes the parameter `sample_size` and returns a tensor of that size with an extra final dimension of size dim.

    log_density() : method that takes the parameter `sample` and returns a tensor of the same size without the final dimension, assumed to be of size dim.
    """
    conditional = True

    def __init__(self, kernel: list[Distribution|Module], generator: Union[torch.Generator, None]):
        super().__init__(generator)
        self.resampler = MultinomialResampler(generator=generator)
        if not isinstance(kernel, Distribution):
            if not hasattr(kernel, "sample"):
                raise AttributeError('Custom distributions must have an "sample" method.')
            if not hasattr(kernel, "log_density"):
                raise AttributeError('Custom distributions must have an "log_density" method.')
            if not hasattr(kernel, "dim"):
                raise AttributeError('Custom distributions must have an "dim" attribute.')
        self.kernel = kernel
        self.dim = self.kernel.dim
        if isinstance(kernel, Distribution) and type(self.kernel).conditional:
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


    def sample(self,  loc: Tensor, weight: Tensor, sample_size: tuple[int,...]|None) -> Tensor:
        """Sample a KDE mixture

        Parameters
        ----------
        loc : Tensor
            Locations of the Kernels
        weight  : Tensor
            Weights of the Kernels
        sample_size : tuple[int,...]|None
            The size of the sample to draw. If None then a single sample is drawn per batch dimension and no sample dimension is used.

        Returns
        -------
        Sample: Tensor
            A sample from the KDE mixture.
        """
        #Multinomial resampling is sampling from a KDE with a dirac kernel.
        self._check_conditions(loc, weight)
        try:
            sampled_locs, _ = self.resampler(loc, weight, generator=self.generator)
        except Exception as e:
            raise RuntimeError(f'Failed to sample kernels with error: \n {e} \n This is likely due to a mismatch in batch dimensions.')
        batch_size = self.get_batch_size(sampled_locs.size(), 2)
        if sample_size is None:
            sample = self.kernel.sample(sample_size=batch_size)
        else:
            sample = self.kernel.sample(sample_size=(*batch_size, *sample_size))
        return sampled_locs + sample

    def log_density(self, sample: Tensor, loc: Tensor, weight: Tensor) -> Tensor:
        """Evaluate the log density of a sample.

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
        try:
            self._check_conditions(loc, weight)
            densities = self.kernel.log_density(sample.unsqueeze(-2) - self._unsqueeze_to_size(loc, sample.unsqueeze(-2), 2))
            return torch.logsumexp(densities + weight.unsqueeze(-2), dim=-1)
        except RuntimeError as e:
            raise RuntimeError(f'Failed to apply condition with error: \n {e} \n This is likely to a mismatch in batch dimensions.')