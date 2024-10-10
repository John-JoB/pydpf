from typing import Callable, Tuple
import torch
from .utils import normalise
from .base import Module

class SIS(Module):
    """
    SMC filters can, in general, be described as special cases of sequential importance sampling (SIS).
    We provide this generic SIS class that can be extended for a given use case, or used by directly supplying the relevant functions.
    SIS iteratively importance samples a Markov-Chain.
    An SIS algorithm is defined by supplying an initial distribution and a Markov kernel.
    """
    def __init__(self, prior: Callable[[int, torch.tensor], Tuple[torch.tensor, torch.tensor]],
                 sampler: Callable[[torch.tensor, torch.tensor, torch.tensor, int], Tuple[torch.tensor, torch.tensor]]
                 ):
        super().__init__()
        self.sampler = sampler
        self.prior = prior

    def initialise(self, n_particles:int, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, weight = self.prior(n_particles, data[0])
        weight, weight_magnitude = normalise(weight)
        return state, weight, weight_magnitude

    def advance_once(self, state: torch.Tensor, weight: torch.Tensor, time: int, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_state, new_weight = self.sampler(state, weight, data[time], time)
        new_weight, new_weight_magnitude = normalise(new_weight)
        return new_state, new_weight, new_weight_magnitude

    def forward(self, data: torch.Tensor,
                n_particles: int,
                time_extent: int,
                aggregation_function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]) -> torch.Tensor:
        #Typically one does not need to store the population of all particles at each timestep
        #Use a function to aggregate over particles to save memory
        state, weight, weight_magnitude = self.initialise(n_particles, data)
        temp = aggregation_function(state, weight, weight_magnitude, data[0], 0)
        output = torch.empty((time_extent+1, *temp.size()), device = data.device, dtype=torch.float32)
        output[0] = temp
        for time in range(time_extent):
            state, weight, weight_magnitude = self.advance_once(state, weight, time, data)
            output[time] = aggregation_function(state, weight, weight_magnitude, data[time], time)
        return output


class ParticleFilter(SIS):
    """
        Helper class for a common case of the SIS, the particle filter (Doucet and Johansen 2008), (Chopin and Papaspiliopoulos 2020).
        Applies a resampling step prior to sampling from the proposal kernel.
    """
    def __init__(self, prior: Callable[[int, torch.tensor], Tuple[torch.tensor, torch.tensor]],
                 resampler: Callable[[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor, torch.tensor]],
                 proposal: Callable[[torch.tensor, torch.tensor, torch.tensor, int], Tuple[torch.tensor, torch.tensor]]) -> None:
        def PF_sampler(x: torch.tensor,
                       w: torch.tensor,
                       data_: torch.tensor,
                       t: int) -> Tuple[torch.tensor, torch.tensor]:
            resampled_x, resampled_w, resampled_indices = resampler(x, w)
            return proposal(resampled_x, resampled_w, data_, t)

        super().__init__(prior, PF_sampler)











