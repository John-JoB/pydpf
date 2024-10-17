from typing import Callable, Tuple
import torch

from .utils import normalise
from .base import Module
from .resampling import systematic, soft, optimal_transport, stop_gradient

class SIS(Module):
    """
    SMC filters can, in general, be described as special cases of sequential importance sampling (SIS).
    We provide this generic SIS class that can be extended for a given use case, or used by directly supplying the relevant functions.
    SIS iteratively importance samples a Markov-Chain.
    An SIS algorithm is defined by supplying an initial distribution and a Markov kernel.
    """
    def __init__(self, prior: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                 sampler: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
                 ):
        super().__init__()
        self.sampler = sampler
        self.prior = prior

    def initialise(self, n_particles:int, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, weight = self.prior(n_particles, data[0])
        weight, weight_magnitude = normalise(weight)
        return state, weight, weight_magnitude

    def advance_once(self, state: torch.Tensor, weight: torch.Tensor, time: int, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_state, new_weight = self.sampler(state, weight, data, time)
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
            state, weight, weight_magnitude = self.advance_once(state, weight, time, data[time])
            output[time] = aggregation_function(state, weight, weight_magnitude, data[time], time)
        return output


class ParticleFilter(SIS):
    """
        Helper class for a common case of the SIS, the particle filter (Doucet and Johansen 2008), (Chopin and Papaspiliopoulos 2020).
        Applies a resampling step prior to sampling from the proposal kernel.
    """
    def __init__(self, initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                 resampler: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                 proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]) -> None:
        def PF_sampler(x: torch.Tensor,
                       w: torch.Tensor,
                       data_: torch.Tensor,
                       t: int) -> Tuple[torch.Tensor, torch.Tensor]:
            resampled_x, resampled_w, resampled_indices = resampler(x, w)
            return proposal(resampled_x, resampled_w, data_, t)

        super().__init__(initial_proposal, PF_sampler)

class DPF(ParticleFilter):
    """
    Basic 'differentiable' particle filter, as described in R. Jonschkowski, D. Rastogi and O. Brock
    'Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors' 2018.
    """

    def __init__(self, initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                 proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]],
                 resampling_generator: torch.Generator = torch.default_generator) -> None:
        """
        Basic 'differentiable' particle filter, as described in R. Jonschkowski, D. Rastogi, O. Brock
        'Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors' 2018.

        Parameters
        ----------
        initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
            Importance sampler from the proposal kernel, takes the state, weights and data and returns the new states and weights.
        """
        super().__init__(initial_proposal, systematic(resampling_generator), proposal)

class SoftDPF(ParticleFilter):
    """
    Differentiable particle filter with soft-resampling (P. Karkus, D. Hsu and W. S. Lee
    'Particle Filter Networks with Application to Visual Localization' 2018).
    """

    def __init__(self, initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                 proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]],
                 softness: float,
                 resampling_generator: torch.Generator = torch.default_generator) -> None:
        """
        Differentiable particle filter with soft-resampling.

        Parameters
        ----------
        initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
            Importance sampler from the proposal kernel, takes the state, weights and data and returns the new states and weights.
        softness: float
            The trade-off parameter between a uniform and the usual resampling distribution.
        """
        super().__init__(initial_proposal, soft(softness, resampling_generator), proposal)

class OptimalTransportDPF(ParticleFilter):
    """
    Differentiable particle filter with optimal transport resampling (A. Corenflos, J. Thornton, G. Deligiannidis and A. Doucet
    'Differentiable Particle Filtering via Entropy-Regularized Optimal Transport' 2021).
    """
    def __init__(self, initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                 proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]],
                 regularisation: float,
                 step_size: float,
                 min_update_size: float = 0.01,
                 max_iterations: int = 100,
                 transport_gradient_clip: float = 1.
                 ) -> None:
        """
        Differentiable particle filter with optimal transport resampling.

        Parameters
        ----------
        initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
            Importance sampler from the proposal kernel, takes the state, weights and data and returns the new states and weights.
        regularisation: float
            The maximum strength of the entropy regularisation, in our implementation regularisation automatically chosen per sample and
             annealed.
        step_size: float
            The factor by which to decrease the entropy regularisation per Sinkhorn loop.
        min_update_size: float
            The size of update to the transport potentials below which iteration should stop.
        max_iterations: int
            The maximum number iterations of the Sinkhorn loop, before stopping. Regardless of convergence.
        transport_gradient_clip: float
            The maximum per-element gradient of the transport matrix that should be passed. Higher valued gradients will be clipped to this value.
        """
        super().__init__(initial_proposal, optimal_transport(regularisation, min_update_size, max_iterations, step_size, transport_gradient_clip), proposal)

class StopGradientDPF(ParticleFilter):
    """
    Differentiable particle filter with stop-gradient resampling (A. Scibor and F. Wood
    'Differentiable Particle Filtering without Modifying the Forward Pass' 2021).
    """
    def __init__(self, initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                 proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]],
                 resampling_generator: torch.Generator = torch.default_generator) -> None:
        """
        Differentiable particle filter with stop-gradient resampling.

        Parameters
        ----------
        initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
            Importance sampler from the proposal kernel, takes the state, weights and data and returns the new states and weights.
        """
        super().__init__(initial_proposal, stop_gradient(resampling_generator), proposal)