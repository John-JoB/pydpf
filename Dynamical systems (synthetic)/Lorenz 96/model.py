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

    def score(self, state:Tensor, observation:Tensor, time, prev_time = None) -> Tensor:
       return self.observation_dist.log_density(observation.unsqueeze(1), state)


class LorenzDynamic(pydpf.Module):
    def __init__(self, state_dim:int, time_gap:float = 0.01, sub_steps:int = 1, device: Union[str, torch.device] = "cpu", generator: torch.Generator = torch.default_generator):
        super().__init__()
        self._forcing = torch.nn.Parameter(torch.rand(size=(1,), device = device, generator=generator).squeeze()*10)
        self.time_gap = time_gap
        self.sub_steps = sub_steps
        self.jiggling_cov = torch.nn.Parameter(torch.eye(state_dim, device = device) * torch.rand(size= (state_dim  ,1), device = device, generator=generator) * 0.1)
        self.jiggling_dist = pydpf.distributions.MultivariateGaussian(mean=torch.zeros((state_dim,), device = device), cholesky_covariance=self.jiggling_cov, diagonal_cov=True, generator=generator)

    @pydpf.constrained_parameter
    def forcing(self):
        return self._forcing, torch.abs(self._forcing)

    def Lorenz_derivative(self, state:Tensor, forcing:Tensor)->Tensor:
        derivative = torch.empty_like(state)
        derivative[:, :, -1] = -state[:, :, -3] * state[:, :, -2] + state[:, :, -2] * state[:, :, 0] - state[:, :, -1]
        derivative[:, :, 0] = -state[:, :, -2] * state[:, :, -1] + state[:, :, -1] * state[:, :, 1] - state[:, :, 0]
        derivative[:, :, 1] = -state[:, :, -1] * state[:, :, 0] + state[:, :, 0] * state[:, :, 2] - state[:, :, 1]
        derivative[:,:,2:-1] = -state[:, :, :-3] * state[:, :, 1:-2] + state[:, :, 1:-2] * state[:, :, 3:] - state[:, :, 2:-1]
        return derivative + forcing

    def runge_kutta_update(self, state:Tensor, time_gap:float, sub_steps:int):
        state_now = state
        step_size = time_gap / sub_steps
        for i in range(sub_steps):
            k1 = self.Lorenz_derivative(state_now, self.forcing)
            k2 = self.Lorenz_derivative(state_now + step_size*k1/2, self.forcing)
            k3 = self.Lorenz_derivative(state_now + step_size*k2/2, self.forcing)
            k4 = self.Lorenz_derivative(state_now + step_size*k3, self.forcing)
            state_now = state_now + step_size*(k1 + 2*k2 + 2*k3 + k4)/6
        return state_now + self.jiggling_dist.sample((state_now.shape[0], state_now.shape[1]))

    def sample(self, prev_state:Tensor, time, prev_time):
        return self.runge_kutta_update(prev_state, (time - prev_time).unsqueeze(1).unsqueeze(2), self.sub_steps)

class LorenzPrior(pydpf.Module):
    def __init__(self, state_dim: int, device: Union[str, torch.device] = "cpu", generator: torch.Generator = torch.default_generator):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.gen = generator
        self.initial_range = torch.nn.Parameter(torch.rand(size=(1,), device = device, generator = generator).squeeze()*0.1)

    def sample(self, batch_size:int, n_particles:int, time):
        initial_state = torch.ones(size=(batch_size, n_particles, self.state_dim), device = self.device)
        initial_state[:, :, 0] = torch.rand(size=(batch_size, n_particles), device = self.device, generator=self.gen) * self.initial_range
        return initial_state