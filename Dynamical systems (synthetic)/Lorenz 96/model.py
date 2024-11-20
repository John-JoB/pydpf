import pydpf
import torch
from typing import Union
from torch import Tensor

class LorenzMeasurement(pydpf.Module):
    def __init__(self, state_dim:int, obs_dim:int,  device: Union[str, torch.device] = "cpu", generator: torch.Generator = torch.default_generator):
        super().__init__()
        root_measurement_cov = torch.nn.Parameter(torch.ones(size=(obs_dim, obs_dim), device=device) * torch.rand(size=(obs_dim,obs_dim), device = device, generator=generator))
        observation_matrix = torch.nn.Parameter(torch.ones(size=(obs_dim, state_dim), device=device) * torch.rand(size=(obs_dim,state_dim), device = device, generator=generator))
        self.observation_dist = pydpf.distributions.LinearGaussian(observation_matrix, torch.zeros(size=(obs_dim,), device=device), root_measurement_cov, diagonal_cov=False, generator=generator)

    def score(self, state:Tensor, observation:Tensor) -> Tensor:
        self.observation_dist.log_density(observation, state)


class LorenzDynamic(pydpf.Module):
    def __init__(self, state_dim:int, time_gap:float = 0.01, sub_steps:int = 1, device: Union[str, torch.device] = "cpu", generator: torch.Generator = torch.default_generator):
        super().__init__()
        self.forcing = torch.nn.Parameter(torch.rand(device = device)*10)
        self.time_gap = time_gap
        self.sub_steps = sub_steps
        self.jiggling_cov = torch.nn.Parameter(torch.eye(state_dim, device = state_dim) * torch.rand(size= (state_dim,1), device = device) * 0.1)
        self.jiggling_dist = pydpf.distributions.MultivariateGaussian(mean=torch.zero((state_dim,), device = device), cholesky_covariance=self.jiggling_cov, diagonal_cov=True, generator=generator)

    def Lorenz_derivative(self, state:Tensor, forcing:Tensor)->Tensor:
        derivative = torch.empty_like(state)
        derivative[:, :, -1] = -state[:, :, -3] * state[:, :, -2] + state[:, :, -2] * state[:, :, 0] - state[:, :, -1]
        derivative[:, :, 0] = -state[:, :, -2] * state[:, :, -1] + state[:, :, -1] * state[:, :, 1] - state[:, :, 0]
        derivative[:, :, 1] = -state[:, :, -1] * state[:, :, 0] + state[:, :, 0] * state[:, :, 2] - state[:, :, 1]
        derivative[:,:,2:-1] = -state[:, :, :-3] * state[:, :, 1:-2] + state[:, :, 1:-2] * state[:, :, 3:] - state[:, :, 2:-1]
        return derivative + forcing + self.jiggling_dist.sample((derivative.shape[0], derivative.shape[1]))


    def runge_kutta_update(self, state:Tensor, time_gap:float, sub_steps:int):
        state_now = state
        step_size = time_gap / sub_steps
        for i in range(sub_steps):
            k1 = self.Lorenz_derivative(state_now, self.forcing)
            k2 = self.Lorenz_derivative(state_now + step_size*k1/2, self.forcing)
            k3 = self.Lorenz_derivative(state_now + step_size*k2/2, self.forcing)
            k4 = self.Lorenz_derivative(state_now + step_size*k3, self.forcing)
            state_now = state_now + step_size*(k1 + 2*k2 + 2*k3 + k4)/6
        return state_now

    def predict(self, state:Tensor):
        return self.runge_kutta_update(state, self.time_gap, self.sub_steps)

class LorenzPrior(pydpf.Module):
    def __init__(self, state_dim: int, device: Union[str, torch.device] = "cpu", generator: torch.Generator = torch.default_generator):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.initial_range = torch.nn.Parameter(torch.rand(device = device, generator = generator)*0.1)

    def sample(self, batch_size:int, n_particles:int):

        initial_state = torch.ones(size=(batch_size, n_particles, self.state_dim), device = self.device)
        initial_state[:, :, 0] = torch.rand(size=(self.size(0), n_particles), device = self.device) * self.initial_range