from typing import Tuple, Callable, Union
from .custom_types import Resampler, ImportanceKernel, ImportanceSampler, Aggregation, ImportanceSamplerLikelihood, ImportanceKernelLikelihood
import torch
from torch import Tensor
from .utils import normalise
from .base import Module
from .resampling import systematic, soft, optimal_transport, stop_gradient, kernel_resampling
from .distributions import KernelMixture
from .utils import batched_select

"""
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
"""

class SIS(Module):
    """
    SMC filters can, in general, be described as special cases of sequential importance sampling (SIS).
    We provide this generic SIS class that can be extended for a given use case, or used by directly supplying the relevant functions.
    SIS iteratively importance samples a Markov-Chain.
    An SIS algorithm is defined by supplying an initial distribution and a Markov kernel.
    """
    def __init__(self, initial_proposal: ImportanceSamplerLikelihood, proposal: ImportanceKernelLikelihood):
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
        initial_proposal: ImportanceKernelLikelihood
            A callable object that takes the number of particles and the data/observations at time-step zero and returns an importance sample
            of the posterior, i.e. particle position and log weights. Also returns the observation likelihood (if applicable).

        proposal: ImportanceKernelLikelihood
            A callable object that implements the proposal kernel. Takes the state and log weights at the previous time step,
            the discreet time index i.e. how many iterations the filter has run for; and the data/observations at the current time-step.
            And returns an importance sample of the posterior at the current time step, i.e. particle position and log weights.
            Also returns the observation likelihood (if applicable).
        """
        super().__init__()
        self.initial_proposal = initial_proposal
        self.proposal = proposal
        self.aggregation_function = None

    def forward(self, data: Tensor, n_particles: int, time_extent: int, aggregation_function: Aggregation) -> Tensor:
        """
        Run a forward pass of the SIS filter. To save memory during inference runs we allow the user to pass a function that takes a population
        of particles and processes this into an output for each time-step. For example, if the goal was the filtering mean then it would be
        wasteful to store the full population of the particles for every time-step. Memory is not saved during training, because the
        computation graph is stored.

        Parameters
        ----------
        data: Tensor
            The data needed to run the filter.
        n_particles: int
            The number of particles to draw per filter.
        time_extent: int
            The maximum time-step to run to, including time 0, the filter will draw {time_extent + 1} importance sample populations.
        aggregation_function: Aggregation
            A Callable that processes the filtering outputs (the particle locations, the normalised log weights,
            the log sum of the unormalised weights, the data, the time-step) into an output per time-step.

        Returns
        -------
        output: Tensor
            The output of the filter, formed from stacking the output of aggregation_function for every time-step.
        """
        #Register any parameters
        self.aggregation_function = aggregation_function
        state, weight, likelihood = self.initial_proposal(n_particles, data[0])
        temp = aggregation_function(state, weight, likelihood, data[0], 0)
        output = torch.empty((time_extent+1, *temp.size()), device = data.device, dtype=torch.float32)
        output[0] = temp
        for time in range(1, time_extent+1):
            state, weight, likelihood = self.proposal(state, weight, data[time], time)
            output[time] = aggregation_function(state, weight, likelihood, data[time], time)
        return output


class ParticleFilter(SIS):
    """
        Helper class for a common case of the SIS, the particle filter (Doucet and Johansen 2008), (Chopin and Papaspiliopoulos 2020).
        Applies a resampling step prior to sampling from the proposal kernel.
    """
    def __init__(self, initial_proposal: ImportanceSampler, resampler: Resampler, proposal: ImportanceKernel) -> None:
        """
        The standard particle filter is a special case of the SIS algorithm. We construct the particle filtering proposal by first
        resampling particles from their population, then applying a proposal kernel restricted such that the particles depend only on the
        population at the previous time-step through the particle at the same index.

        Parameters
        ----------
        initial_proposal: ImportanceSampler
            A callable object that takes the number of particles and the data/observations at time-step zero and returns an importance sample
            of the posterior, i.e. particle position and log weights.

        resampler: Resampler:
            The resampling algorithm to use. Takes teh state and log weights at the previous time-step and returns the state and log weights
            after resampling. Resampling algorithms must also return a third tensor. Used to report extra information about how the particles
            were chosen, in most cases the resampled indices; this is often useful for diagnostics. But, this basic implementation discards
            this tensor.

        proposal: ImportanceKernel
            A callable object that implements the proposal kernel. Takes the state and log weights at the previous time step,
            the discreet time index i.e. how many iterations the filter has run for; and the data/observations at the current time-step.
            And returns an importance sample of the posterior at the current time step, i.e. particle position and log weights. For the
            resultant algorithm to be properly a particle filter, this kernel should be restricted such that the particles depend only on the
            population at the previous time-step through the particle at the same index.
        """
        class PF_initial_sampler(Module):
            def __init__(self):
                super().__init__()
                self.initial_proposal = initial_proposal

            def forward(self, n_particles: int, data_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
                state, weight = self.initial_proposal(n_particles, data_)
                weight, likelihood = normalise(weight)
                return state, weight, likelihood


        class PF_sampler(Module):
            def __init__(self):
                super().__init__()
                self.resampler = resampler
                self.proposal = proposal

            def forward(self, x, w, data_, t):
                resampled_x, resampled_w = self.resampler(x, w)
                initial_likelihood = torch.logsumexp(resampled_w, dim=-1)
                state, weight = self.proposal(resampled_x, resampled_w, data_, t)

                weight , likelihood = normalise(weight)
                return state, weight, likelihood - initial_likelihood

        super().__init__(PF_initial_sampler(), PF_sampler())

class DPF(ParticleFilter):
    """
    Basic 'differentiable' particle filter, as described in R. Jonschkowski, D. Rastogi and O. Brock
    'Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors' 2018.
    """

    def __init__(self, initial_proposal: ImportanceSampler, proposal: ImportanceKernel, resampling_generator: torch.Generator = torch.default_generator) -> None:
        """
        Basic 'differentiable' particle filter, as described in R. Jonschkowski, D. Rastogi, O. Brock
        'Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors' 2018.

        Parameters
        ----------
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        """
        super().__init__(initial_proposal, systematic(resampling_generator), proposal)

class SoftDPF(ParticleFilter):
    """
    Differentiable particle filter with soft-resampling (P. Karkus, D. Hsu and W. S. Lee
    'Particle Filter Networks with Application to Visual Localization' 2018).
    """

    def __init__(self, initial_proposal: ImportanceSampler, proposal: ImportanceKernel, softness: float, resampling_generator: torch.Generator = torch.default_generator) -> None:
        """
        Differentiable particle filter with soft-resampling.

        Parameters
        ----------
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        softness: float
            The trade-off parameter between a uniform and the usual resampling distribution.
        """
        super().__init__(initial_proposal, soft(softness, resampling_generator), proposal)

class OptimalTransportDPF(ParticleFilter):
    """
    Differentiable particle filter with optimal transport resampling (A. Corenflos, J. Thornton, G. Deligiannidis and A. Doucet
    'Differentiable Particle Filtering via Entropy-Regularized Optimal Transport' 2021).
    """
    def __init__(self, initial_proposal: ImportanceSampler,
                 proposal: ImportanceKernel,
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
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
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
        super().__init__(initial_proposal, optimal_transport(regularisation, step_size, min_update_size, max_iterations, transport_gradient_clip), proposal)

class StopGradientDPF(ParticleFilter):
    """
    Differentiable particle filter with stop-gradient resampling (A. Scibor and F. Wood
    'Differentiable Particle Filtering without Modifying the Forward Pass' 2021).
    """
    def __init__(self, initial_proposal: ImportanceSampler, proposal: ImportanceKernel, resampling_generator: torch.Generator = torch.default_generator) -> None:
        """
        Differentiable particle filter with stop-gradient resampling.

        Parameters
        ----------
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernel
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        """
        super().__init__(initial_proposal, stop_gradient(resampling_generator), proposal)

class StabilisedStopGradientDPF(SIS):
    def __init__(self,
                 initial_proposal: ImportanceSampler,
                 proposal: Callable[[Tensor, Tensor, int], Tensor],
                 log_proposal_density: Callable[[Tensor, Tensor, Tensor, int], Tensor],
                 log_posterior_density: Union[Callable[[Tensor, Tensor, Tensor, int] ,Tensor], None] = None,
                 resampling_generator: torch.Generator = torch.default_generator) -> None:
        """
        The variance reduced version of the Stop Gradient DPF (A. Scibor and F. Wood
        'Differentiable Particle Filtering without Modifying the Forward Pass' 2021).

        This is less general than the Stop Gradient DPF and requires a specific form for the proposal.

        Unlike most the Stop Gradient resampler the computational cost is quadratic in the number of particles.

        Notes
        -----
        This particle filter directly targets the per-time-step marginal posterior rather than the posterior of complete trajectories.
        In the bootstrap case the two are equivalent in their forward pass. But in general this filter should not be used as part of a procedure to sample trajectories.

        Parameters
        ----------
        initial_proposal: ImportanceSampler
            Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
            and returns the importance sampled state and weights
        proposal: ImportanceKernels
            Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
        log_proposal_density: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
            Returns the density of the proposal model given the new state, old state, data, and discrete time.
        log_posterior_density: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
            Returns the (unnormalised) density of the posterior model given the new state, old state, data, and discrete time.
        resampling_generator:
            The generator to track the resampling rng.
        """

        def _systematic(state: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            with torch.no_grad():
                offset = torch.rand((weights.size(0),), device=state.device, generator=resampling_generator)
                cum_probs = torch.cumsum(torch.exp(weights), dim=1)
                # No index can be above 1. and the last index must be exactly 1.
                # Fix this in case of numerical errors
                cum_probs = torch.where(cum_probs > 1., 1., cum_probs)
                cum_probs[:, -1] = 1.
                resampling_points = torch.arange(weights.size(1), device=state.device) + offset.unsqueeze(1)
                sampled_indices = torch.searchsorted(cum_probs * weights.size(1), resampling_points)
            return batched_select(state, sampled_indices), torch.zeros_like(weights), sampled_indices

        def resampling(state: Tensor, weight: Tensor):
            state, no_grad_weights, sampled_indices = _systematic(state, weight)
            # Save computation if gradient is not required
            resampled_weights = batched_select(weight, sampled_indices)
            return state, resampled_weights - resampled_weights.detach()

        class PF_initial_sampler(Module):
            def __init__(self):
                super().__init__()
                self.initial_proposal = initial_proposal

            def forward(self, n_particles: int, data_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
                state, weight = self.initial_proposal(n_particles, data_)
                weight, likelihood = normalise(weight)
                return state, weight, likelihood



        class StabPFSampler(Module):
            def __init__(self):
                super().__init__()
                self.log_proposal_density = log_proposal_density
                self.log_posterior_density = log_posterior_density
                self.proposal = proposal


            def forward(self, x, w, data_, t):
                resampled_x, resampled_w = resampling(x, w)
                state = self.proposal(resampled_x, data_, t)
                expanded_state = state.unsqueeze(2).expand(-1, -1, resampled_x.size(1), -1)
                resampled_x = resampled_x.unsqueeze(1).expand(-1, state.size(1), -1, -1)
                weight_numerator = self.log_posterior_density(expanded_state, resampled_x, data_, t)
                weight_denominator = self.log_proposal_density(expanded_state, resampled_x, data_, t)
                resampled_w = resampled_w.unsqueeze(-1)
                weight = torch.logsumexp(resampled_w + weight_numerator, dim =1)  - torch.logsumexp(resampled_w.detach() + weight_denominator, dim = 1)
                weight, likelihood = normalise(weight)
                return state, weight, likelihood

        super().__init__(PF_initial_sampler(), StabPFSampler())

class KernelDPF(ParticleFilter):

    def __init__(self, initial_proposal: ImportanceSampler, proposal: ImportanceKernel, kernel: KernelMixture) -> None:
        """
            Differentiable particle filter with mixture kernel resampling (Younis and Sudderth 'Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes' 2024).



            Parameters
            ----------
            initial_proposal: ImportanceSampler
                Importance sampler for the initial distribution, takes the number of particles and the data at time 0,
                and returns the importance sampled state and weights
            proposal: ImportanceKernel
                Importance sampler for the proposal kernel, takes the state, weights and data and returns the new states and weights.
            kernel: KernelMixture
                The kernel mixture to convolve over the particles to form the KDE sampling distribution.

            Returns
            -------
            kernel_resampler: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
                A Module whose forward method implements kernel resampling.
        """

        super().__init__(initial_proposal, kernel_resampling(kernel), proposal)