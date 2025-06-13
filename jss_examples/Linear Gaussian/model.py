from pydpf import pydpf
import torch

class GaussianDynamic(pydpf.Module):
    def __new__(cls, dx:int, generator):
        device = generator.device
        dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1) - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        dynamic_offset = torch.zeros(dx, device=device)
        return pydpf.LinearGaussian(weight=dynamic_matrix, bias=dynamic_offset, cholesky_covariance=torch.eye(dx, device=device), generator=generator)

class GaussianObservation(pydpf.Module):
    def __new__(cls, dx:int, dy:int, generator):
        device = generator.device
        observation_matrix = torch.zeros((dy, dx), device=device)
        for i in range(dy):
            observation_matrix[i, i] = 1
        observation_offset = torch.zeros(dy, device=device)
        return pydpf.LinearGaussian(weight=observation_matrix, bias=observation_offset, cholesky_covariance=torch.eye(dy, device=device), generator=generator)

class GaussianPrior(pydpf.Module):
    def __new__(cls, dx:int, generator):
        device = generator.device
        return pydpf.MultivariateGaussian(torch.zeros(dx, device=device), torch.eye(dx, device=device), generator=generator)

class GaussianOptimalProposal(pydpf.Module):
    def __init__(self, dx:int, dy:int, generator):
        super().__init__()
        device = generator.device
        covariance = torch.eye(dx, device=device)
        self.dx = dx
        self.dy = dy
        for i in range(dy):
            covariance[i,i] = .5
        self.dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1) - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(dx, device=device), cholesky_covariance=torch.sqrt(covariance), generator=generator)

    def sample(self, observation, prev_state, **data):
        sample = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1)
        mean[:,:,:self.dy] = (mean[:,:,:self.dy] + observation.unsqueeze(1))/2
        return mean + sample

    def log_density(self, observation, prev_state, state, **data):
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1)
        mean[:, :, :self.dy] = (mean[:, :, :self.dy] + observation.unsqueeze(1))/2
        sample = state - mean
        return self.dist.log_density(sample)

class GaussianLearnedProposal(pydpf.Module):
    def __init__(self, dx:int, dy:int, generator):
        super().__init__()
        device = generator.device
        cov = torch.nn.Parameter(torch.eye(dx, device=device))
        self.dx = dx
        self.dy = dy
        self.dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1) - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(dx, device=device), cholesky_covariance=cov, generator=generator)
        self.x_weight = torch.nn.Parameter(torch.ones(dx, device=device))
        self.y_weight = torch.nn.Parameter(torch.zeros(dy, device=device))
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(dx, device=device), cholesky_covariance=cov, generator=generator, diagonal_cov=True)

    def sample(self, observation, prev_state, **data):
        sample = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1) * self.x_weight
        mean[:,:,:self.dy] = mean[:,:,:self.dy] + observation.unsqueeze(1) * self.y_weight
        return mean + sample

    def log_density(self, observation, prev_state, state, **data):
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1) * self.x_weight
        mean[:, :, :self.dy] = mean[:, :, :self.dy] + observation.unsqueeze(1) * self.y_weight
        sample = state - mean
        return self.dist.log_density(sample)
