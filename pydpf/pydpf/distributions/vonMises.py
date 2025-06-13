import torch
from ..base import constrained_parameter, cached_property
from .base import Distribution
from torch import Tensor
from torch.distributions.von_mises import VonMises as torch_VM


class VonMises(Distribution):
    """von Mises distribution in radians

    Parameters
    ----------
    mean: Tensor
        The mean of the distribution.
    concentration: Tensor
        The concentration of the distribution. All entries must be positive.
    generator: torch.Generator
        The generator object used to control the RNG.

    Notes
    -----
    This distribution is not reparameterisable as implemented. There is no way to correlate the dimensions of the output sample.
    Each dimension is sampled from a separate von Mises distribution.
    """

    conditional = False

    def __init__(self, mean: Tensor, concentration: Tensor, generator:torch.Generator):
        super().__init__(generator)
        self.mean_ = mean
        self.concentration_ = concentration
        self.generator = generator
        self.dim = mean.size(0)
        self.device = mean.device
        self.torch_dist = torch_VM(self.mean, self.concentration)

    @cached_property
    def _double_concentration(self):
        return self.concentration.to(torch.double)

    @cached_property
    def _double_mean(self):
        return self.mean.to(torch.double)

    @cached_property
    def _proposal_r(self):
        kappa = self._double_concentration
        tau = 1 + (1 + 4 * kappa ** 2).sqrt()
        rho = (tau - (2 * tau).sqrt()) / (2 * kappa)
        _proposal_r = (1 + rho ** 2) / (2 * rho)
        # second order Taylor expansion around 0 for small kappa
        _proposal_r_taylor = 1 / kappa + kappa
        return torch.where(kappa < 1e-5, _proposal_r_taylor, _proposal_r)

    def _rejection_sample(self, x):
        done = torch.zeros(x.shape, dtype=torch.bool, device=self.device)
        while not done.all():
            u = torch.rand((3,) + x.shape, dtype=torch.double, device=self.device, generator=self.generator)
            u1, u2, u3 = u.unbind()
            z = torch.cos(torch.pi * u1)
            f = (1 + self._proposal_r * z) / (self._proposal_r + z)
            c = self._double_concentration * (self._proposal_r - f)
            accept = ((c * (2 - c) - u2) > 0) | ((c / u2).log() + 1 - c >= 0)
            if accept.any():
                x = torch.where(accept, (u3 - 0.5).sign() * f.acos(), x)
                done = done | accept
        return (x + torch.pi + self._double_mean) % (2 * torch.pi) - torch.pi

    @constrained_parameter
    def mean(self):
        mod_angle = torch.remainder(self.mean_, 2*torch.pi)
        return self.mean_, torch.where(mod_angle > torch.pi, mod_angle - 2*torch.pi, mod_angle)

    @constrained_parameter
    def concentration(self):
        return self.concentration_, torch.abs(self.concentration_)

    #The torch implementation has no way to use a generator so we modify their implementation
    @torch.no_grad()
    def sample(self, sample_size: tuple[int,...]|None = None) -> Tensor:
        """Sample a Von-Mises distribution.

        Parameters
        ----------
        sample_size: tuple[int,...]|None
            The size of the sample to draw. Draw a single sample without a sample dimension if None.

        Returns
        -------
        sample: Tensor
            A multivariate Von-Mises sample.
        """
        if sample_size is None:
            true_sample_size = self.mean.size()
        else:
            true_sample_size = (*sample_size, self.mean.size(-1))
        out = torch.empty(true_sample_size, dtype=torch.double, device=self.device)
        return self._rejection_sample(out).to(dtype=self.mean.dtype)

    def log_density(self, sample: Tensor) -> Tensor:
        """Get the density of a Von-Mises distribution.

        pi and -pi are identified.

        Parameters
        ----------
        sample: Tensor
            The sample to get the density of.

        Returns
        -------
        sample log_density: Tensor
            The log density of each datum in the sample.
        """
        return torch.sum(self.torch_dist.log_prob(sample), dim=-1)