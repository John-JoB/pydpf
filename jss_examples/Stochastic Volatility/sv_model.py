import pydpf
import torch

class StochasticVolatility_Prior(pydpf.Module):
    @pydpf.cached_property
    def sd(self):
        i1 = torch.ones((1,1), device=self.alpha.device)
        return torch.sqrt(i1*(self.sigma**2/(1-self.alpha**2)))

    @pydpf.constrained_parameter
    def alpha(self):
        return self.alpha_, torch.clip(self.alpha_, 1e-2, 1-1e-2)

    def __init__(self, sigma, alpha, generator):
        super().__init__()
        self.sigma = sigma
        self.alpha_ = alpha
        i1 = torch.ones((1, 1), device=generator.device)
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(1, device=generator.device), cholesky_covariance=i1, generator=generator)

    def sample(self, batch_size: int, n_particles: int, **data):
        return self.dist.sample(sample_size=(batch_size, n_particles)) * self.sd

    def log_density(self, state, **data):
        return self.dist.log_density(sample=state/self.sd) - torch.log(self.sd)

class StochasticVolatility_Dynamic(pydpf.Module):
    def __new__(cls, sigma, alpha, generator):
        return pydpf.LinearGaussian(weight = alpha, bias = torch.zeros(1, device=generator.device), cholesky_covariance=sigma, generator=generator)

class StochasticVolatility_Observation(pydpf.Module):

    @pydpf.constrained_parameter
    def beta(self):
        return self.beta_, torch.clip(self.beta_, 1e-3)

    def __init__(self, beta, generator):
        super().__init__()
        self.beta_ = beta
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(1, device=generator.device), cholesky_covariance=torch.ones((1, 1), device=generator.device), generator=generator)

    def sample(self, state, **data):
        sample = self.dist.sample((state.size(0), state.size(1)))
        return sample * torch.exp(state/2) * self.beta

    def score(self, observation, state, **data):
        sd = torch.exp(state/2) * self.beta
        #With this simple SV model there's not a convenient way to disallow very small volatilities, clip them to avoid numerical errors.
        sd = torch.clip(sd, 1e-7)
        return self.dist.log_density(observation.unsqueeze(1)/sd) - torch.log(sd).squeeze()


def make_SSM(sigma, alpha, beta, device, generator=None):
    if generator is None:
        generator = torch.Generator(device).manual_seed(0)
    return pydpf.FilteringModel(prior_model=StochasticVolatility_Prior(sigma, alpha, generator),
                                dynamic_model=StochasticVolatility_Dynamic(sigma, alpha, generator),
                                observation_model=StochasticVolatility_Observation(beta, generator))