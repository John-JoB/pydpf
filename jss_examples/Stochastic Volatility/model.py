from sympy.strategies.branch import condition

import pydpf
import torch

class StochasticVolatility_Prior(pydpf.Module):
    @pydpf.cached_property
    def sd(self):
        return torch.sqrt(torch.ones((1,1), device=self.alpha.device)*(self.sigma**2/(1-self.alpha**2)))

    @pydpf.constrained_parameter
    def alpha(self):
        return self.alpha_, torch.clip(self.alpha_, 1e-3, 1-1e-3)

    def __init__(self, sigma, alpha, generator):
        super().__init__()
        self.sigma = sigma
        self.alpha_ = alpha
        self.dist = pydpf.MultivariateGaussian(torch.zeros(1, device=generator.device), torch.ones((1,1), device=generator.device), generator=generator)

    def sample(self, batch_size: int, n_particles: int, **data):
        return self.dist.sample(sample_size=(batch_size, n_particles)) * self.sd

    def log_density(self, state, **data):
        return self.dist.log_density(sample=state/self.sd) - torch.log(self.sd)

class StochasticVolatility_Dynamic(pydpf.Module):
    def __new__(cls, sigma, alpha, generator):
        return pydpf.LinearGaussian(weight = alpha, bias = torch.zeros(1, device=generator.device), cholesky_covariance=sigma, generator=generator)

class StochasticVolatility_Observation(pydpf.Module):
    def __init__(self, beta, generator):
        super().__init__()
        self.beta = beta
        self.dist = pydpf.MultivariateGaussian(torch.zeros(1, device=generator.device), torch.ones((1,1), device=generator.device), generator=generator)

    def sample(self, state, **data):
        sample  = self.dist.sample((state.size(0),))
        return sample * torch.exp(state) * self.beta

    def score(self, observation, state, **data):
        sd = torch.exp(state) * self.beta
        return self.dist.log_density(observation.unsqueeze(1)/sd) - torch.log(sd).squeeze()


def make_SSM(sigma, alpha, beta, device):
    return pydpf.FilteringModel(prior_model=StochasticVolatility_Prior(sigma, alpha, torch.Generator(device).manual_seed(0)),
                                dynamic_model=StochasticVolatility_Dynamic(sigma, alpha, torch.Generator(device).manual_seed(0)),
                                observation_model=StochasticVolatility_Observation(beta, torch.Generator(device).manual_seed(0)))