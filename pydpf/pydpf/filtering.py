"""Python module for the core filtering algorithms."""

import torch
from torch import Tensor
from .utils import normalise
from .base import Module
from .resampling import SystematicResampler, SoftResampler, OptimalTransportResampler, StopGradientResampler, KernelResampler, MultinomialResampler
from .distributions import KernelMixture
from .model_based_api import FilteringModel
from .base import DivergenceError
from warnings import warn
from .conditional_resampling import ConditionalResampler
from copy import copy
from typing import Callable



class SIS(Module):
    """Module that represents a sequential importance sampling (SIS) algorithm. A SIS algorthm is fully specified by its importance sampling
        procedures, the user should supply a proposal kernel that may depend on the time-step; and a special case for time 0.

    Parameters
    ----------
    initial_proposal: Module | None
        A callable object that takes the number of particles and the data/observations at time-step zero and returns an importance sample
        of the posterior, i.e. particle position and log weights. Also returns the observation likelihood (if applicable).
    proposal: Module | None
        A callable object that implements the proposal kernel. Takes the state and log weights at the previous time step,
        the discreet time index i.e. how many iterations the filter has run for; and the data/observations at the current time-step.
        And returns an importance sample of the posterior at the current time step, i.e. particle position and log weights.
        Also returns the observation factor likelihood.

    Notes
    -----
    SMC filters can, in general, be described as special cases of sequential importance sampling (SIS).
    We provide this generic SIS class that can be extended for a given use case, or used by directly supplying the relevant functions.
    SIS iteratively importance samples a Markov-Chain.
    An SIS algorithm is defined by supplying an initial distribution and a Markov kernel.

    This implementation is more general than the standard SIS algorithm. There is no independence requirements for the samples within a
    batch. This means that the particles can be drawn from an arbitrary joint distribution on depended on the data and the particles at
    the previous time-step. Both the usual particle filter [1]_ and interacting multiple model particle filter [2]_ are special
    cases of this algorithm.

    References
    ----------
    .. [1] Chopin N, Papaspiliopoulos O (2020). An Introduction to Sequential Monte Carlo, chapter
       Particle Filtering, pp. 129–165. Springer.

    .. [2] Boers Y, Driessen J (2003). “Interacting multiple model particle filter.” IEE Proc. Radar,
       Sonar Nav., 150, 344–349. ISSN 1350-2395.
    """
    def __init__(self, *, initial_proposal: Module | None = None, proposal: Module | None = None):
        super().__init__()
        if initial_proposal is not None:
            self._register_functions(initial_proposal, proposal)


    def _register_functions(self, initial_proposal: Module, proposal: Module):
        self.initial_proposal = initial_proposal
        self.proposal = proposal
        self.aggregation_function = None

    @staticmethod
    def _get_time_data(t: int, **data) -> dict:
        time_dict = {k:v[t] for k, v in data.items() if k != 'series_metadata' and k != 'state' and v is not None}
        time_dict['t'] = t
        if data['time'] is not None and t>0:
            time_dict['prev_time'] = data['time'][t-1]
        if data['series_metadata'] is not None:
            time_dict['series_metadata'] = data['series_metadata']
        return time_dict

    class _EmptyGradReg:
        @staticmethod
        def apply(state, weight, time_data):
            return state, weight

    def forward(self,
                n_particles: int,
                time_extent: int,
                aggregation_function: dict[str, Module] | Module,
                observation: Tensor,
                *,
                gradient_regulariser: torch.autograd.Function | None = None,
                ground_truth: Tensor | None = None,
                control: Tensor | None = None,
                time: Tensor | None = None,
                series_metadata: Tensor | None = None) -> dict[str, Tensor] | Tensor :
        """Run a forward pass of the SIS filter.

        Parameters
        ----------
        n_particles: int
            The number of particles to draw per filter.
        time_extent: int
            The maximum time-step to run to, including time 0, the filter will draw {time_extent + 1} importance sample populations.
        aggregation_function: dict[str, Module] or Module
            A module that's forward function processes the filtering outputs (the particle locations, the normalised log weights,
            the log sum of the unnormalised weights, the data, the time-step) into an output per time-step.
            Or a string indexed dictionary of such items.
        observation: Tensor
            The observations of the hidden variable system.
        gradient_regulariser: torch.autograd.Function or None. Default None.
            A autograd function to apply to the particles at the start of every time-step. It should leave the forward pass unchanged but may modify the gradients during the backward pass
            the intended usage is to regularise the gradient in some way. Optional.
        ground_truth: Tensor or None. Default None.
            The ground truth latent state. Should only be pass to the aggregation function and not used in the proposal functions. Optional.
        control: Tensor or None. Default None.
            The control actions. Optional.
        time: Tensor or None. Default None.
            The continuous time each time-step occurs at. Optional.
        series_metadata: Tensor or None. Default None.
            The series_metadata. Optional.

        Returns
        -------
        output: Tensor or dict[str, Tensor]
            The output of the filter, formed from stacking the output of aggregation_function for every time-step.
            Or if aggregation_function is a dictionary, a dictionary of these output Tensors one for each aggregation function.

        Notes
        -----
        To save memory during inference runs we allow the user to pass a function that takes a population
        of particles and processes this into an output for each time-step. For example, if the goal was the filtering mean then it would be
        wasteful to store the full population of the particles for every time-step. The memory savings are most impactful during inference as
        pytorch retains many intermediate tensors for gradient computation otherwise.
        """
        #Register module should the aggregation function have learnable parameters
        if isinstance(aggregation_function,dict):
            self.aggregation_function = torch.nn.ModuleDict(aggregation_function)
            output_dict = True
        else:
            self.aggregation_function = aggregation_function
            output_dict = False

        if gradient_regulariser is None or not (self.training and torch.is_grad_enabled()):
            gradient_regulariser = self._EmptyGradReg

        gt_exists = False
        if ground_truth is not None:
            gt_exists = True
        time_data = self._get_time_data(0, observation = observation, control = control, time = time, series_metadata = series_metadata)
        state, weight, likelihood = self.initial_proposal(n_particles = n_particles, **time_data)
        likelihood = likelihood - torch.log(torch.tensor(n_particles, dtype=torch.float32, device=observation.device))

        if output_dict:
            output = {}
            for name, function in aggregation_function.items():
                if gt_exists:
                    temp = function(state=state, weight=weight, likelihood=likelihood, ground_truth=ground_truth[0], **time_data)
                else:
                    temp = function(state=state, weight=weight, likelihood=likelihood, **time_data)
                output[name] = torch.empty((time_extent+1, *temp.size()), device = observation.device, dtype=torch.float32)
                output[name][0] = temp
        else:
            if gt_exists:
                temp = aggregation_function(state = state, weight = weight, likelihood = likelihood, ground_truth = ground_truth[0], **time_data)
            else:
                temp = aggregation_function(state = state, weight = weight, likelihood = likelihood, **time_data)
            output = torch.empty((time_extent+1, *temp.size()), device = observation.device, dtype=torch.float32)
            output[0] = temp
        for t in range(1, time_extent+1):
            try:
                time_data = self._get_time_data(t, observation = observation, control = control, time = time, series_metadata = series_metadata)
                prev_state, prev_weight = gradient_regulariser.apply(state, weight, time_data)
                state, weight, likelihood = self.proposal(prev_state = prev_state, prev_weight = prev_weight, **time_data)
                if output_dict:
                    for name, function in aggregation_function.items():
                        if gt_exists:
                            output[name][t] = function(state=state, weight=weight, likelihood=likelihood, ground_truth=ground_truth[t], **time_data)
                        else:
                            output[name][t] = function(state=state, weight=weight, likelihood=likelihood, **time_data)
                else:
                    if gt_exists:
                        output[t] = aggregation_function(state=state, weight=weight, likelihood=likelihood, ground_truth = ground_truth[t], **time_data)
                    else:
                        output[t] = aggregation_function(state=state, weight=weight, likelihood=likelihood, **time_data)
            except DivergenceError as e:
                warn(f'Detected divergence at time-step {t} with message:\n    {e} \nStopping iteration early.')
                if isinstance(output, dict):
                    return {name: value[:t-1] for name, value in output.items()}
                else:
                    return output[:t-1]
        return output


class ParticleFilter(SIS):
    """The standard SIR particle filter


    Parameters
    ----------
    resampler: Module
        The resampling algorithm to use. Takes the state and log weights at the previous time-step and returns the state and log weights
        after resampling.
    SSM: FilteringModel
        A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
    use_REINFORCE_for_proposal: bool
        Whether to use the REINFORCE estimator for the gradient due to the particle proposal process. Applying REINFORCE to only some components of the
        state space is not permitted with this API, such a use case would require a custom SIS process.

    Notes
    -----
        The standard particle filter is a special case of the SIS algorithm. We construct the particle filtering proposal by first
        resampling particles from their population, then applying a proposal kernel restricted such that the particles depend only on the
        population at the previous time-step through the particle at the same index [1]_. This Class forms the base for the majority of
        filtering algorithms packaged with PyDPF.

    References
    ----------
    .. [1] Chopin N, Papaspiliopoulos O (2020). An Introduction to Sequential Monte Carlo, chapter Particle Filtering, pp. 129–165. Springer.
    """
    def __init__(self, resampler: Module = None, SSM: FilteringModel = None, *, use_REINFORCE_for_proposal:bool=False, use_REINFORCE_for_initial_proposal:bool=False) -> None:
        self.REINFORCE_i = use_REINFORCE_for_initial_proposal
        self.REINFORCE = use_REINFORCE_for_proposal

        super().__init__()
        if resampler is not None:
            self._register_functions(resampler=resampler, SSM=SSM)

    @staticmethod
    def _make_PF_prior(SSM, REINFORCE_i:bool):
        if SSM.has_initial_proposal:
            if REINFORCE_i:
                def prior(n_particles, observation, **data):
                    with torch.no_grad():
                        state = SSM.initial_proposal_model.sample(batch_size = observation.size(0), n_particles = n_particles, observation=observation, **data)
                        prop_density = SSM.initial_proposal_model.log_density(state = state, observation = observation, **data)
                    weight = (SSM.observation_model.score(state = state, observation = observation, **data)
                              - prop_density
                              + SSM.prior_model.log_density(state = state, **data))
                    return state, weight
            else:
                def prior(n_particles, observation, **data):
                    state = SSM.initial_proposal_model.sample(batch_size = observation.size(0), n_particles = n_particles, observation=observation, **data)
                    weight = (SSM.observation_model.score(state = state, observation = observation, **data)
                              - SSM.initial_proposal_model.log_density(state = state, observation = observation, **data)
                              + SSM.prior_model.log_density(state = state, **data))
                    return state, weight
        else:
            if REINFORCE_i:
                def prior(n_particles, observation, **data):
                    with torch.no_grad():
                        state = SSM.prior_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data)
                    density = SSM.prior_model.log_density(state = state, **data)
                    weight = SSM.observation_model.score(state = state, observation = observation, **data) + density - density.detach()
                    return state, weight
            else:
                def prior(n_particles, observation, **data):
                    state = SSM.prior_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data)
                    weight = SSM.observation_model.score(state = state, observation = observation, **data)
                    return state, weight

        return prior

    @staticmethod
    def _make_PF_proposal(SSM, REINFORCE:bool):
        if SSM.has_proposal:
            if REINFORCE:
                def prop(prev_state, prev_weight, observation, **data):
                    with torch.no_grad():
                        new_state = SSM.proposal_model.sample(prev_state = prev_state, observation=observation, **data)
                        prop_density = SSM.proposal_model.log_density(state = new_state, prev_state = prev_state, observation = observation, **data)
                    new_weight = (prev_weight + SSM.observation_model.score(state = new_state, observation = observation, **data)
                                  - prop_density
                                  + SSM.dynamic_model.log_density(state = new_state, prev_state = prev_state, **data))
                    return new_state, new_weight
            else:
                def prop(prev_state, prev_weight, observation, **data):
                    new_state = SSM.proposal_model.sample(prev_state = prev_state, observation=observation, **data)
                    new_weight = (prev_weight + SSM.observation_model.score(state = new_state, observation = observation, **data)
                                  - SSM.proposal_model.log_density(state = new_state, prev_state = prev_state, observation = observation, **data)
                                  + SSM.dynamic_model.log_density(state = new_state, prev_state = prev_state, **data))
                    return new_state, new_weight
        else:
            if REINFORCE:
                def prop(prev_state, prev_weight, observation, **data):
                    with torch.no_grad():
                        new_state = SSM.dynamic_model.sample(prev_state = prev_state, **data)
                    density = SSM.dynamic_model.log_density(state=new_state, prev_state=prev_state, **data)
                    new_weight = prev_weight + SSM.observation_model.score(state=new_state, observation = observation, **data) + density - density.detach()
                    return new_state, new_weight
            else:
                def prop(prev_state, prev_weight, observation, **data):
                    new_state = SSM.dynamic_model.sample(prev_state = prev_state, **data)
                    new_weight = prev_weight + SSM.observation_model.score(state=new_state, observation = observation, **data)
                    return new_state, new_weight

        return prop


    def _register_functions(self, resampler: Module, SSM: FilteringModel):
        self.SSM = SSM
        self.resampler = resampler

        if self.REINFORCE:
            if not hasattr(SSM.dynamic_model, 'log_density'):
                raise AttributeError("The dynamic model must implement a 'log_density' method for REINFORCE.")
        if self.REINFORCE_i:
            if not hasattr(SSM.prior_model, 'log_density'):
                raise AttributeError("The prior model must implement a 'log_density' method for REINFORCE.")

        prior = ParticleFilter._make_PF_prior(SSM, self.REINFORCE_i)
        prop = ParticleFilter._make_PF_proposal(SSM, self.REINFORCE)

        def initial_sampler(n_particles: int, **data):
            state, weight = prior(n_particles=n_particles, **data)
            weight, likelihood = normalise(weight)
            return state, weight, likelihood

        def pf_sampler(prev_state, prev_weight, **data):
            resampled_x, resampled_w = self.resampler(prev_state, prev_weight, **data)
            state, weight = prop(prev_state=resampled_x, prev_weight=resampled_w, **data)
            try:
                weight, likelihood = normalise(weight)
            except ValueError:
                raise DivergenceError('Found batch where all weights are small.')
            return state, weight, likelihood
        super()._register_functions(initial_sampler, pf_sampler)


class MarginalParticleFilter(SIS):
    """The marginal particle filter

    Parameters
    ----------
    resampler: Module
        The resampling algorithm to use. Takes the state and log weights at the previous time-step and returns the state and log weights
        after resampling.
    SSM: FilteringModel
        A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
    use_REINFORCE_for_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
    use_REINFORCE_for_initial_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.
    optimise_for_bootstrap: bool. Default True.
        If the system is bootstrap then the filter can be optimised due to not computing terms which are guaranteed to cancel. Set to True to take advantage of these optimisations.

    Notes
    -----
    The marginal particle filter is a special case of the SIS algorithm. The unlike standard particle filter the marginal particle filter considers resampling in its evaluation of the proposal distribution by
    accounting for the possibility that any new particle could have been derived from any particle at the previous time-step instead of only considering it's genealogical path. This is done by marginalising over the
    ancestral indices at each time-step, hence the name. [1]_

    The optimisations taken when ``optimise_for_bootstrap`` is True are only valid if the algorithm is truly bootstrap aside from the first time-step, i.e. the proposal and resampling both do not induce importance
    weights. However, it is permitted for ``SSM.initial_proposal_model`` to be non-bootstrap. If ``SSM.proposal_model`` is non-None then this is detected and the value of ``optimise_for_bootstrap`` is ignored.
    However, there is no efficient way to detect if the resampler modifies the weights so it is on the user to manually set ``optimise_for_bootstrap`` to False in this case, otherwise this filter will silently use
    an algorithm with undefined behaviour.

    .. warning:: Setting the parameter ``optimise_for_bootstrap`` to True can silently invoke undefined behaviour if the resampler is non-standard, see the Notes section for detail.


    References
    ----------
    .. [1] Klaas M, de Freitas N, Doucet A (2005). “Toward Practical N 2 Monte Carlo: the Marginal
       Particle Filter.” In Proc. Conf. Uncert. Art. Intell. (UAI), pp. 308–315. Arlington, Virginia.
    """


    def __init__(self,
                 resampler: Module|None = None,
                 SSM: FilteringModel = None,
                 *,
                 use_REINFORCE_for_proposal: bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False,
                 optimise_for_bootstrap: bool = True):
        super().__init__()
        self.REINFORCE_i = use_REINFORCE_for_initial_proposal
        self.REINFORCE = use_REINFORCE_for_proposal
        self.optimise = optimise_for_bootstrap and (not SSM.has_proposal)
        if self.optimise:
            self.PF = ParticleFilter(resampler, SSM)
        if resampler is not None:
            self._register_functions(resampler=resampler, SSM=SSM)


    def forward(self, *args, **kwargs):
        if self.optimise and not torch.is_grad_enabled():
            return self.PF.forward(*args, **kwargs)
        return super().forward(*args, **kwargs)


    @staticmethod
    def _make_MPF_proposal(SSM, resampler, REINFORCE, optimise) -> Callable:

        if SSM.has_proposal:
            if REINFORCE:
                def prop(prev_state, prev_weight, observation, **data):
                    resampled_state, resampled_weight = resampler(prev_state, prev_weight, **data)
                    used_weight = resampler.cache['used_weight']
                    with torch.no_grad():
                        state = SSM.proposal_model.sample(prev_state=resampled_state, observation=observation, **data).detach()
                    expanded_state = state.unsqueeze(2).expand(-1, -1, state.size(1), -1).flatten(1, 2)
                    expanded_prev_state = prev_state.unsqueeze(1).expand(-1, state.size(1), -1, -1).flatten(1, 2)
                    dynamic_log_density = SSM.dynamic_model.log_density(state=expanded_state, prev_state=expanded_prev_state, **data).reshape(state.size(0), state.size(1), state.size(1))
                    with torch.no_grad():
                        proposal_log_density = SSM.proposal_model.log_density(state=expanded_state, prev_state=expanded_prev_state, observation=observation, **data).reshape(state.size(0), state.size(1), state.size(1))
                    obs_score = SSM.observation_model.score(state=state, observation=observation, **data)
                    weight = (torch.logsumexp(prev_weight.unsqueeze(1) + dynamic_log_density, dim=-1)
                              - torch.logsumexp(used_weight.unsqueeze(1) + proposal_log_density, dim=-1)
                              + obs_score)
                    return state, weight, obs_score, dynamic_log_density, proposal_log_density, resampled_weight
            else:
                def prop(prev_state, prev_weight, observation, **data):
                    resampled_state, resampled_weight = resampler(prev_state, prev_weight, **data)
                    used_weight = resampler.cache['used_weight']
                    state = SSM.proposal_model.sample(prev_state=resampled_state, observation=observation, **data)
                    expanded_prev_state = prev_state.unsqueeze(1).expand(-1, state.size(1), -1, -1).flatten(1, 2)
                    expanded_state = state.unsqueeze(2).expand(-1, -1, state.size(1), -1).flatten(1, 2)
                    dynamic_log_density = SSM.dynamic_model.log_density(state=expanded_state, prev_state=expanded_prev_state, **data).reshape(state.size(0), state.size(1), state.size(1))
                    proposal_log_density = SSM.proposal_model.log_density(state=expanded_state, prev_state=expanded_prev_state, observation=observation, **data).reshape(state.size(0), state.size(1), state.size(1))
                    obs_score = SSM.observation_model.score(state=state, observation=observation, **data)
                    weight = (torch.logsumexp(prev_weight.unsqueeze(1) + dynamic_log_density, dim=-1)
                              - torch.logsumexp(used_weight.unsqueeze(1) + proposal_log_density, dim=-1)
                              + obs_score)
                    return state, weight, obs_score, dynamic_log_density, proposal_log_density, resampled_weight

        else:
            if REINFORCE:
                def prop(prev_state, prev_weight, observation, **data):
                    resampled_state, resampled_weight = resampler(prev_state, prev_weight, **data)
                    used_weight = resampler.cache['used_weight']
                    with torch.no_grad():
                        state = SSM.dynamic_model.sample(prev_state=resampled_state, **data)
                    expanded_prev_state = prev_state.unsqueeze(1).expand(-1, state.size(1), -1, -1).flatten(1, 2)
                    expanded_state = state.unsqueeze(2).expand(-1, -1, state.size(1), -1).flatten(1, 2)
                    dynamic_log_density = SSM.dynamic_model.log_density(state=expanded_state, prev_state=expanded_prev_state, **data).reshape(state.size(0), state.size(1), state.size(1))
                    obs_score = SSM.observation_model.score(state=state, observation=observation, **data)
                    detach_dynamic = dynamic_log_density.detach()
                    weight = (torch.logsumexp(prev_weight.unsqueeze(1) + dynamic_log_density, dim=-1)
                              - torch.logsumexp(used_weight.unsqueeze(1) + detach_dynamic, dim=-1)
                              + obs_score)
                    return state, weight, obs_score, dynamic_log_density, detach_dynamic, resampled_weight
            else:
                if optimise:
                    def prop(prev_state, prev_weight, observation, **data):
                        resampled_state, resampled_weight = resampler(prev_state, prev_weight, **data)
                        used_weight = resampler.cache['used_weight']
                        state = SSM.dynamic_model.sample(prev_state=resampled_state, **data)
                        expanded_prev_state = prev_state.unsqueeze(1).expand(-1, state.size(1), -1, -1).flatten(1, 2)
                        expanded_state = state.unsqueeze(2).expand(-1, -1, state.size(1), -1).flatten(1, 2)
                        obs_score = SSM.observation_model.score(state=state, observation=observation, **data)
                        # Detach to save computing redundant gradient terms.
                        with torch.no_grad():
                            dynamic_log_density = SSM.dynamic_model.log_density(state=expanded_state, prev_state=expanded_prev_state, **data).reshape(state.size(0), state.size(1), state.size(1))
                        weight = (torch.logsumexp(prev_weight.unsqueeze(1) + dynamic_log_density, dim=-1)
                                  - torch.logsumexp(used_weight.unsqueeze(1) + dynamic_log_density, dim=-1)
                                  + obs_score)
                        return state, weight, obs_score, 0, 0, resampled_weight
                else:
                    def prop(prev_state, prev_weight, observation, **data):
                        resampled_state, resampled_weight = resampler(prev_state, prev_weight, **data)
                        used_weight = resampler.cache['used_weight']
                        state = SSM.dynamic_model.sample(prev_state=resampled_state, **data)
                        expanded_prev_state = prev_state.unsqueeze(1).expand(-1, state.size(1), -1, -1).flatten(1, 2)
                        expanded_state = state.unsqueeze(2).expand(-1, -1, state.size(1), -1).flatten(1, 2)
                        dynamic_log_density = SSM.dynamic_model.log_density(state=expanded_state, prev_state=expanded_prev_state, **data).reshape(state.size(0), state.size(1), state.size(1))
                        obs_score = SSM.observation_model.score(state=state, observation=observation, **data)
                        weight = (torch.logsumexp(prev_weight.unsqueeze(1) + dynamic_log_density, dim=-1)
                                  - torch.logsumexp(used_weight.unsqueeze(1) + dynamic_log_density, dim=-1)
                                  + obs_score)
                        return state, weight, obs_score, 0, 0, resampled_weight

        return prop



    def _register_functions(self, resampler: Module, SSM: FilteringModel):
        self.resampler = resampler
        self.SSM = SSM

        prior = ParticleFilter._make_PF_prior(SSM, self.REINFORCE_i)
        prop = MarginalParticleFilter._make_MPF_proposal(SSM, resampler, self.REINFORCE_i, self.optimise)



        def initial_sampler(n_particles: int, **data):
            state, weight = prior(n_particles=n_particles, **data)
            weight, likelihood = normalise(weight)
            return state, weight, likelihood

        if isinstance(self.resampler, ConditionalResampler):
            def mpf_sampler(prev_state, prev_weight, observation, **data):
                state, weight, obs_score, dynamic_log_density, proposal_log_density, resampled_weight = prop(prev_state, prev_weight, observation, **data)
                weight = torch.where(self.resampler.cache['mask'], weight, resampled_weight + obs_score + torch.diag(dynamic_log_density) - torch.diag(proposal_log_density))
                try:
                    weight, likelihood = normalise(weight)
                except ValueError:
                    raise DivergenceError('Found batch where all weights are small.')

                return state, weight, likelihood
        else:
            def mpf_sampler(prev_state, prev_weight, observation, **data):
                state, weight, _, _, _, _ = prop(prev_state, prev_weight, observation, **data)
                try:
                    weight, likelihood = normalise(weight)
                except ValueError:
                    raise DivergenceError('Found batch where all weights are small.')
                return state, weight, likelihood

        super()._register_functions(initial_sampler, mpf_sampler)



class DPF(ParticleFilter):
    """Basic 'differentiable' particle filter.

    Parameters
    ----------
    SSM: FilteringModel
        A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
    resampling_generator: torch.Generator
        Generator to track the resampling rng.
    multinomial: bool. Default False.
        If true then use multinomial resampling. Otherwise, use systematic resampling.
    use_REINFORCE_for_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
    use_REINFORCE_for_initial_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.

        Notes
        -----
        This DPF does not use a differentiable resampling method, instead the gradients are detached at each time-step. See [1]_ for detail.

        References
        ----------
        .. [1] Jonschkowski R, Rastogi D, Brock O (2018). “Differentiable Particle Filters: End-to-End
           Learning with Algorithmic Priors.” In Proc. Robot.: Sci. Syst. Pittsburgh, PA, USA.
        """

    def __init__(self,
                 SSM: FilteringModel = None,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial:bool = False,
                 *,
                 use_REINFORCE_for_proposal: bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False
                 ) -> None:

        resampler = MultinomialResampler(resampling_generator) if multinomial else SystematicResampler(resampling_generator)
        super().__init__(resampler, SSM, use_REINFORCE_for_proposal=use_REINFORCE_for_proposal, use_REINFORCE_for_initial_proposal=use_REINFORCE_for_initial_proposal)
        temp = copy(self.proposal)
        self.proposal = lambda prev_state, prev_weight, **data: temp(prev_state.detach(), prev_weight.detach(), **data)

class SoftDPF(ParticleFilter):
    """Differentiable particle filter with soft-resampling. [1]_

    Parameters
    ----------
    SSM: FilteringModel
        A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
    resampling_generator: torch.Generator
        Generator to track the resampling rng.
    multinomial: bool. Default False.
        If true then use multinomial resampling. Otherwise, use systematic resampling.
    use_REINFORCE_for_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
    use_REINFORCE_for_initial_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.

    Notes
    -----
    Applies a non-target proposal to resampling, replace the resampling weights :math:`w^{i}_{t}` with:

    .. math:: (\text{softness}) w^{i}_{t} + \frac{(1-\text{softness})}{\text{n_particles}}

    With ``softness`` = 1 then this is equivelent to the straight-through estimator used in [2]_.

    References
    ----------
    .. [1] Karkus P, Hsu D, Lee WS (2018). “Particle filter networks with application to visual localization.” In Proc. Conf. Robot Learn., pp. 169–178. PMLR, Zurich, CH.

    .. [2] Naesseth C, Linderman S, Ranganath R, Blei D (2018). “Variational sequential monte carlo.” In Proc. Int. Conf. Art. Int. and Stat. (AISTATS), pp. 968–977. PMLR, Lanzarote, Canary Islands.
    """

    def __init__(self, SSM: FilteringModel = None,
                 softness: float = 0.7,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial: bool = False,
                 *,
                 use_REINFORCE_for_proposal: bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False,
                 ) -> None:

        base_resampler = MultinomialResampler(resampling_generator) if multinomial else SystematicResampler(resampling_generator)
        resampler = SoftResampler(softness, base_resampler, resampling_generator.device)
        super().__init__(resampler, SSM, use_REINFORCE_for_proposal=use_REINFORCE_for_proposal, use_REINFORCE_for_initial_proposal=use_REINFORCE_for_initial_proposal)

class OptimalTransportDPF(ParticleFilter):
    """Differentiable particle filter with optimal transport resampling.

    Parameters
    ----------
    SSM: FilteringModel
        A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
    regularisation: float
        The maximum strength of the entropy regularisation, in our implementation the initial regularisation automatically chosen per sample and exponentially decreased to the given regularisation.
    decay_rate: float
        The factor by which to decrease the entropy regularisation per Sinkhorn loop.
    min_update_size: float
        The size of update to the transport potentials below which the algorithm is considered converged and iteration should stop.
    max_iterations: int
        The maximum number iterations of the Sinkhorn loop, before stopping. Regardless of convergence.
    transport_gradient_clip: float
        The maximum per-element gradient of loss due to the transport matrix that should be passed. Higher valued gradients will be clipped to this value.
    use_REINFORCE_for_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
    use_REINFORCE_for_initial_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.

    Notes
    -----
    See [1]_. It is not recommended to set ``use_REINFORCE_for_proposal`` to True as it will lose the convergence properties of the reparameterised algorithm but retain the instability and costly runtime of OT filtering.
    Consider other algorithms if the proposal cannot be reparameterised.

    References
    ----------
    .. [1] Corenflos A, Thornton J, Deligiannidis G, Doucet A (2021). “Differentiable Particle Filtering via Entropy-Regularized Optimal Transport.” In Proc. Int. Conf. on Machine Learn. (ICML), pp. 2100–2111. Online.
    """
    def __init__(self, SSM: FilteringModel = None,
                 regularisation: float = 0.1,
                 decay_rate: float = 0.9,
                 min_update_size: float = 0.01,
                 max_iterations: int = 100,
                 transport_gradient_clip: float = 1.,
                 *,
                 use_REINFORCE_for_proposal: bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False
                 ) -> None:


        super().__init__(OptimalTransportResampler(regularisation, decay_rate, min_update_size, max_iterations, transport_gradient_clip), SSM, use_REINFORCE_for_proposal = use_REINFORCE_for_proposal, use_REINFORCE_for_initial_proposal=use_REINFORCE_for_initial_proposal)


class StopGradientDPF(ParticleFilter):
    """Differentiable particle filter with stop-gradient resampling.

    Parameters
    ----------
    SSM: FilteringModel
        A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
    resampling_generator: torch.Generator
        Generator to track the resampling rng.
    multinomial: bool. Default False.
        If true then use multinomial resampling. Otherwise, use systematic resampling.
    use_REINFORCE_for_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
    use_REINFORCE_for_initial_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.

    Notes
    -----
    See [1]_. If ``use_REINFORCE_for_proposal`` is ``True`` then this DPF is as described in [1]_ and the proposal model cannot be learned. Otherwise, it is as in [2]_.

    References
    ----------
    .. [1] Scibior A, Wood F (2021). “Differentiable particle filtering without modifying the forward pass.” arXiv:2106.10314

    .. [2] Cox B, P´erez-Vieites S, Zilberstein N, Sevilla M, Segarra S, Elvira V (2024). “End-to-End Learning of Gaussian Mixture Proposals Using Differentiable Particle Filters and Neural Networks.”
    In Int. Conf. Acoustics, Speech and Sig. Proc. (ICASSP), pp. 9701–9705.
    """
    def __init__(self, SSM: FilteringModel = None,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial:bool = False,
                 *,
                 use_REINFORCE_for_proposal: bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False
                 ) -> None:
        base_resampler = MultinomialResampler(resampling_generator) if multinomial else SystematicResampler(resampling_generator)
        resampler = StopGradientResampler(base_resampler)
        super().__init__(resampler, SSM, use_REINFORCE_for_proposal=use_REINFORCE_for_proposal, use_REINFORCE_for_initial_proposal=use_REINFORCE_for_initial_proposal)

class MarginalStopGradientDPF(MarginalParticleFilter):
    """Differentiable particle filter with marginal stop-gradient resampling.

    Parameters
    ----------
    SSM: FilteringModel
        A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
    resampling_generator: torch.Generator
        Generator to track the resampling rng.
    multinomial: bool. Default False.
        If true then use multinomial resampling. Otherwise, use systematic resampling.
    use_REINFORCE_for_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
    use_REINFORCE_for_initial_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.

    Notes
    -----
    See [1]_. If ``use_REINFORCE_for_proposal`` is ``True`` then this DPF is as described in [1]_ and the proposal model cannot be learned. Otherwise, if the model is bootstrap it is as in [2]_, if
    the SSM has a proposal then this is a non-published scheme.

    References
    ----------
    .. [1] Scibior A, Wood F (2021). “Differentiable particle filtering without modifying the forward pass.” arXiv:2106.10314

    .. [2] Brady JJ, et al. (2024). “Differentiable Interacting Multiple Model Particle Filtering.” arXiv preprint arXiv:2410.00620.
    """

    def __init__(self,
                 SSM: FilteringModel = None,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial = False,
                 *,
                 use_REINFORCE_for_proposal:bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False,
                 ) -> None:

        base_resampler = MultinomialResampler(resampling_generator) if multinomial else SystematicResampler(resampling_generator)
        resampler = StopGradientResampler(base_resampler)
        super().__init__(resampler, SSM, use_REINFORCE_for_proposal=use_REINFORCE_for_proposal, use_REINFORCE_for_initial_proposal=use_REINFORCE_for_initial_proposal)


class KernelDPF(ParticleFilter):
    """Differentiable particle filter with mixture kernel resampling.

    Parameters
    ----------
    SSM: FilteringModel
        A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
    Kernel: KernelMixture
        The Kernel mixture to resample from.
    use_REINFORCE_for_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
    use_REINFORCE_for_initial_proposal: bool. Default False.
        Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.


    Notes
    -----
    See [1]_. Currently only Gaussian Kernels are supported.

    References
    ----------
    .. [1] Younis A, Sudderth E (2023). “Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes.” In Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), volume 36. New Orleans, LA, USA.
    """


    def __init__(self,
                 SSM: FilteringModel = None,
                 kernel: KernelMixture = None,
                 *,
                 use_REINFORCE_for_proposal: bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False) -> None:

        if kernel is None:
            raise ValueError('Must specify a kernel mixture')

        super().__init__(KernelResampler(kernel), SSM, use_REINFORCE_for_proposal=use_REINFORCE_for_proposal, use_REINFORCE_for_initial_proposal=use_REINFORCE_for_initial_proposal)


class MarginalDPF(MarginalParticleFilter):
    """Differentiable particle filter based on the marginal particle filter.

        Notes
        -----
        Analagous to the basic 'differentiable' particle filter, as described in R. Jonschkowski, D. Rastogi and O. Brock
        'Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors' 2018 but based on the marginal particle filter.

        Parameters
        ----------
        SSM: FilteringModel
            A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
        resampling_generator: torch.Generator
            Generator to track the resampling rng.
        multinomial: bool. Default False.
            If true then use multinomial resampling. Otherwise, use systematic resampling.
        use_REINFORCE_for_proposal: bool. Default False.
            Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
        use_REINFORCE_for_initial_proposal: bool. Default False.
            Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.

        See Also
        --------
        MarginalParticleFilter : The marginal particle filter base implementation.
        """

    def __init__(self,
                 SSM: FilteringModel = None,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial: bool = False,
                 *,
                 use_REINFORCE_for_proposal: bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False
                 ) -> None:
        resampler = MultinomialResampler(resampling_generator) if multinomial else SystematicResampler(resampling_generator)
        super().__init__(resampler, SSM, use_REINFORCE_for_proposal=use_REINFORCE_for_proposal, use_REINFORCE_for_initial_proposal=use_REINFORCE_for_initial_proposal)
        temp = copy(self.proposal)
        self.proposal = lambda prev_state, prev_weight, **data: temp(prev_state.detach(), prev_weight.detach(), **data)

class StraightThroughDPF(ParticleFilter):
    """Similar to the DPF but the gradient of the state is passed through resampling without modification. (T. Le et al. 'Auto-encoding sequential monte carlo' 2018,
        C. Maddison et al. ' Filtering variational objectives' 2018, and C. Naesseth et al. 'Variational sequential monte carlo' 2018.) Equivelant to soft resampling
        with a softness parameter of 1.

        Parameters
        ----------
        SSM: FilteringModel
            A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
        resampling_generator: torch.Generator
            Generator to track the resampling rng.
        multinomial: bool. Default False.
            If true then use multinomial resampling. Otherwise, use systematic resampling.
        use_REINFORCE_for_proposal: bool. Default False.
            Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
        use_REINFORCE_for_initial_proposal: bool. Default False.
            Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.
    """

    def __init__(self,
                 SSM: FilteringModel = None,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial:bool = False,
                 *,
                 use_REINFORCE_for_proposal: bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False) -> None:
        """


        Parameters
        ----------
        SSM: FilteringModel
            A FilteringModel that represents the SSM (and optionally a proposal model). See the documentation of FilteringModel for more complete information.
            If this parameter is not None then the values of initial_proposal and proposal are ignored.
        resampling_generator:
            The generator to track the resampling rng.
        """
        resampler = MultinomialResampler(resampling_generator) if multinomial else SystematicResampler(resampling_generator)
        super().__init__(resampler, SSM, use_REINFORCE_for_proposal=use_REINFORCE_for_proposal, use_REINFORCE_for_initial_proposal=use_REINFORCE_for_initial_proposal)


class MarginalStraightThroughDPF(MarginalParticleFilter):
    """The marginal particle filter with the straight through gradient estimator.


        Parameters
        ----------
        SSM: FilteringModel
            A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
        resampling_generator: torch.Generator
            Generator to track the resampling rng.
        multinomial: bool. Default False.
            If true then use multinomial resampling. Otherwise, use systematic resampling.
        use_REINFORCE_for_proposal: bool. Default False.
            Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
        use_REINFORCE_for_initial_proposal: bool. Default False.
            Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.

        See Also
        --------
        MarginalParticleFilter : The marginal particle filter base implementation.
        StraightThroughDPF : The DPF with the straight through gradient estimator.
    """

    def __init__(self,
                 SSM: FilteringModel = None,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial: bool = False,
                 *,
                 use_REINFORCE_for_proposal: bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False) -> None:
        resampler = MultinomialResampler(resampling_generator) if multinomial else SystematicResampler(resampling_generator)
        super().__init__(resampler, SSM, use_REINFORCE_for_proposal=use_REINFORCE_for_proposal, use_REINFORCE_for_initial_proposal=use_REINFORCE_for_initial_proposal)




class MarginalSoftDPF(MarginalParticleFilter):
    """Marginal particle filter with soft-resampling.

        Parameters
        ----------
        SSM: FilteringModel
            A ``FilteringModel`` that represents the SSM (and optionally a proposal model). See the documentation of ``FilteringModel`` for more complete information.
        resampling_generator: torch.Generator
            Generator to track the resampling rng.
        multinomial: bool. Default False.
            If true then use multinomial resampling. Otherwise, use systematic resampling.
        use_REINFORCE_for_proposal: bool. Default False.
            Whether to use the REINFORCE estimator for the gradient due to the particle proposal process.
        use_REINFORCE_for_initial_proposal: bool. Default False.
            Whether to use the REINFORCE estimator for the gradient due to the initial particle proposal process.

        See Also
        --------
        MarginalParticleFilter : The marginal particle filter base implementation.
        SoftDPF : The soft DPF
        """

    def __init__(self, SSM: FilteringModel = None,
                 softness: float = 0.7,
                 resampling_generator: torch.Generator = torch.default_generator,
                 multinomial: bool = False,
                 *,
                 use_REINFORCE_for_proposal: bool = False,
                 use_REINFORCE_for_initial_proposal: bool = False
                 ) -> None:
        base_resampler = MultinomialResampler(resampling_generator) if multinomial else SystematicResampler(resampling_generator)
        resampler = SoftResampler(softness, base_resampler, resampling_generator.device)
        super().__init__(resampler, SSM, use_REINFORCE_for_proposal=use_REINFORCE_for_proposal, use_REINFORCE_for_initial_proposal=use_REINFORCE_for_initial_proposal, optimise_for_bootstrap=False)