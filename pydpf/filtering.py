from typing import Callable, Tuple
import torch

from .utils import normalise
from .base import Module
from .resampling import systematic, soft, optimal_transport, stop_gradient

'''
Python module for the core filtering algorithms. 

In pydpf a filtering algorithm is defined by it's sampling procedures rather than the underlying model. 
For example, in order sequential importance sampling, instead of providing a prior, dynamic kernel and observation model; 
one provides two procedures: importance sampling from the posterior at time zero and importance sampling from the posterior at subsequent time
steps. We motivate this choice in two respects, foremostly the flexibility provided with this design we permit the user to easily use 
whatever proposal strategy their use case calls for without the package getting in the way. Secondarily, it is more conceptually inline
with the set-up of most current DPF problems. One learns an algorithm that performs well on the desired metrics, not a model that is
close to the truth.

The downside to this strategy being it requires some extra work from the user in certain cases. For, example should the user want to try 
slightly differing filtering algorithms on the same underlying parameterisation, then it is fully on them to design their code to make that 
process easy.

Note on Callables: most Modules in this file take at least one Callable argument, these must be instantiated Module (or less preferably 
torch.nn.Module) classes with the forward() method defined if the function contains learnable parameters.

Note on data: data should be treated similarly abstractly to the algorithm/model. The data passed to the proposal routines at each timestep
should be whatever data is required to sample the desired posterior. In a vanilla (bootstrap) filtering scenario this the observation at 
that time-step, but it doesn't have to be in general. pydpf makes no distinction between observations (past, present or future) and 
exogenous variables, all are treated as non-random inputs.
'''

class SIS(Module):
    """
    SMC filters can, in general, be described as special cases of sequential importance sampling (SIS).
    We provide this generic SIS class that can be extended for a given use case, or used by directly supplying the relevant functions.
    SIS iteratively importance samples a Markov-Chain.
    An SIS algorithm is defined by supplying an initial distribution and a Markov kernel.
    """
    def __init__(self, initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                 proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
                 ):
        """
        Module that represents a sequential importance sampling (SIS) algorithm. A SIS algorthm is fully specified by its importance sampling
        procedures, the user should supply a proposal kernel that may depend on the time-step; and a special case for time 0.

        Notes
        -----
        This implementation is more general than the standard SIS algorithm. There is no independence requirements for the samples within a
        batch. This means that the particles can be drawn from an arbitrary joint distribution on depended on the data and the particles at
        the previous time-step. This means that both the usual particle filter and interacting multiple model particle filter are special
        cases of this algorithm. It's also possible to make the filters within a batch depend on each other, but don't do that.


        Parameters
        ----------
        initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            A callable object that takes the number of particles and the data/observations at time-step zero and returns an importance sample
            of the posterior, i.e. particle position and log weights.

        proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
            A callable object that implements the proposal kernel. Takes the state and log weights at the previous time step,
            the discreet time index i.e. how many iterations the filter has run for; and the data/observations at the current time-step.
            And returns an importance sample of the posterior at the current time step, i.e. particle position and log weights.
        """
        super().__init__()
        self.initial_proposal = initial_proposal
        self.proposal = proposal

    def initialise(self, n_particles:int, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialise the SIS filter by drawing the particles at time zero.

        Parameters
        ----------
        n_particles: int
            The number of particles to draw per filter.
        data: torch.Tensor
            The data associated to time-step zero.

        Returns
        -------
            state: torch.Tensor
                The locations of the particles at time zero.
            weights: torch.Tensor
                The log normalised weights of the particles at time zero.
            weight_magnitude: torch.Tensor
                The log of the sum of the unnormalised weights of the particles at time zero.
        """
        state, weight = self.initial_proposal(n_particles, data[0])
        weight, weight_magnitude = normalise(weight)
        return state, weight, weight_magnitude

    def advance_once(self, state: torch.Tensor, weight: torch.Tensor, time: int, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advance the filter one time-step by drawing from the proposal kernel.

        Parameters
        ----------
        state: torch.Tensor
            The locations of the particles at the previous time-step.
        weight: torch.Tensor
            The log normalised weights of the particles at the previous time-step.
        time: int
            The current time-step.
        data: torch.Tensor
            The data associated to the current time-step.

        Returns
        -------
            state: torch.Tensor
                The locations of the particles at the current time-step.
            weights: torch.Tensor
                The log normalised weights of the particles at the current time-step.
            weight_magnitude: torch.Tensor
                The log of the sum of the unnormalised weights of the particles at the current time-step.
        """

        new_state, new_weight = self.proposal(state, weight, data, time)
        new_weight, new_weight_magnitude = normalise(new_weight)
        return new_state, new_weight, new_weight_magnitude

    def forward(self, data: torch.Tensor,
                n_particles: int,
                time_extent: int,
                aggregation_function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]) -> torch.Tensor:
        """
        Run a forward pass of the SIS filter. To save memory during inference runs we allow the user to pass a function that takes a population
        of particles and processes this into an output for each time-step. For example, if the goal was the filtering mean then it would be
        wasteful to store the full population of the particles for every time-step. Memory is not saved during training, because the
        computation graph is stored.

        Parameters
        ----------
        data: torch.Tensor
            The data needed to run the filter.
        n_particles: int
            The number of particles to draw per filter.
        time_extent: int
            The maximum time-step to run to, including time 0, the filter will draw {time_extent + 1} importance sample populations.
        aggregation_function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]
            A Callable that processes the filtering outputs (the particle locations, the normalised log weights,
            the log sum of the unormalised weights, and the time-step) into an output per time-step.

        Returns
        -------
        output: torch.Tensor
            The output of the filter, formed from stacking the output of aggregation_function for every time-step.
        """

        state, weight, weight_magnitude = self.initialise(n_particles, data)
        temp = aggregation_function(state, weight, weight_magnitude, data[0], 0)
        output = torch.empty((time_extent+1, *temp.size()), device = data.device, dtype=torch.float32)
        output[0] = temp
        for time in range(1, time_extent+1):
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
        """
        The standard particle filter is a special case of the SIS algorithm. We construct the particle filtering proposal by first
        resampling particles from their population, then applying a proposal kernel restricted such that the particles depend only on the
        population at the previous time-step through the particle at the same index.

        Parameters
        ----------
        initial_proposal: Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            A callable object that takes the number of particles and the data/observations at time-step zero and returns an importance sample
            of the posterior, i.e. particle position and log weights.

        resampler: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            The resampling algorithm to use. Takes teh state and log weights at the previous time-step and returns the state and log weights
            after resampling. Resampling algorithms must also return a third tensor. Used to report extra information about how the particles
            were chosen, in most cases the resampled indices; this is often useful for diagnostics. But, this basic implementation discards
            this tensor.

        proposal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
            A callable object that implements the proposal kernel. Takes the state and log weights at the previous time step,
            the discreet time index i.e. how many iterations the filter has run for; and the data/observations at the current time-step.
            And returns an importance sample of the posterior at the current time step, i.e. particle position and log weights. For the
            resultant algorithm to be properly a particle filter, this kernel should be restricted such that the particles depend only on the
            population at the previous time-step through the particle at the same index.
        """
        class PF_sampler(Module):
            def __init__(self):
                super().__init__()
                self.resampler = resampler
                self.proposal = proposal

            def forward(self, x, w, data_, t):
                resampled_x, resampled_w, _ = self.resampler(x, w)
                return self.proposal(resampled_x, resampled_w, data_, t)

        super().__init__(initial_proposal, PF_sampler())

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