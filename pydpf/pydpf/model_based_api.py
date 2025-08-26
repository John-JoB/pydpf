"""Module that defines the FilteringModel class"""
from .base import Module
from torch import Tensor
from .distributions import Distribution

class _obs_from_model(Module):
    def __init__(self, dist:Distribution):
        super().__init__()
        self.dist = dist

    def score(self, state:Tensor, observation:Tensor, **data) -> Tensor:
        return self.dist.log_density(sample = observation.unsqueeze(1), condition_on=state)

    def sample(self, state:Tensor, **data) -> Tensor:
        return self.dist.sample(condition_on = state)

class _dyn_from_model(Module):
    def __init__(self, dist:Distribution):
        super().__init__()
        self.dist = dist

    def log_density(self, prev_state:Tensor, state:Tensor, **data)->Tensor:
        return self.dist.log_density(condition_on=prev_state, sample=state)

    def sample(self, prev_state:Tensor, **data) -> Tensor:
        return self.dist.sample(condition_on=prev_state)

class _prior_from_model(Module):
    def __init__(self, dist:Distribution):
        super().__init__()
        self.dist = dist

    def log_density(self, state:Tensor, **data)->Tensor:
        return self.dist.log_density(sample=state)

    def sample(self, batch_size:int, n_particles:int, **data) -> Tensor:
        return self.dist.sample(sample_size=(batch_size, n_particles))


class FilteringModel(Module):
    r"""The class for grouping model components to define a state-space model with optional proposal distributions.

    Parameters
    ----------
    dynamic_model:Module|Distribution
        The dynamic model.
    observation_model:Module|Distribution
        The observation model.
    prior_model:Module|Distribution
        The prior model.
    initial_proposal_model:Module|Distribution|None. Default None
        The initial proposal model.
    proposal_model:Module|Distribution|None. Default None
        The proposal model.

    Notes
    -----
    Any of the model components can be Distribution objects. However, distributions defined by Distribution objects can only be dependent on at most one variable.
    That variable is:
    For the prior_model and initial_proposal_model: None.
    For the dynamic_model and proposal_model: ``prev_state``.
    For the observation_model: ``state``.

    If the components are not Distributions then they must be Modules with that will be accessed through specific methods. Not all of these methods need to be defined depending on the use case. Let B be the size of the
    batch dimension, K be the size of the particle dimension, D-x be the size of the inherent dimension for data type x. Starred (\*) arguments are always passed, unstarred arguments are only passed if they exist.

    ``prior_model``:

        ``log_density()`` Parameters: \*state, time, control, series_metadata. Output: the probability density of the state, Tensor of size (B X K). Required for non-bootstrap filtering.

        ``sample()`` Parameters: \*batch_size := B, \*n_particles := K, time, control, series_metadata. Output a sample from the prior, Tensor of size (B X K X D-state). Required for data generation and bootstrap filtering.

    ``dynamic_model``:

        ``log_density()`` Parameters: \*prev_state, \*state, prev_time, time, control, series_metadata, \*t. Output: the probability density of the state, Tensor of size (B X K). Required for non-bootstrap filtering.

        ``sample()`` Parameters: \*prev_state, prev_time, time, control, series_metadata, \*t. Output: a sample from the dynamic model, Tensor of size (B X K X D-state). Required for data generation and bootstrap filtering.

    ``observation_model``:

        ``score()`` Parameters: \*state, \*observation, prev_time, time, control, series_metadata, \*t. Output: The score of an observation given the latent state, usually the log-density, Tensor fo size (B X K). Required for filtering.

        ``sample()`` Parameters: \*state, prev_time, time, control, series_metadata, \*t. Output: a sample from the dynamic model, Tensor of size (B X K X D-observation). Required for data generation.

    ``inital_proposal_model``:

        ``log_density()`` Parameters: \*state, \*observation, time, control, series_metadata. Output: the probability density of the state under the initial proposal, Tensor of size (B X K). Required for non-bootstrap filtering.

        ``sample()`` Parameters: \*batch_size := B, \*n_particles := K, \*observation, time, control, series_metadata. Output a sample from the initial proposal, Tensor of size (B X K X D-state). Required non-bootstrap filtering.

    ``proposal_model``:

        ``log_density()`` Parameters: \*prev_state, \*state, \*observation, prev_time, time, control, series_metadata, \*t. Output: the probability density of the state under the proposal, Tensor of size (B X K). Required for non-bootstrap filtering.

        ``sample()`` Parameters: \*prev_state, \*observation, prev_time, time, control, series_metadata, \*t. Output: a sample from the proposal model, Tensor of size (B X K X D-state). Required for non-bootstrap filtering.


    We check that the components have the components required for SIRS particle filtering. Additional components may be required depending on the loss function or when generating data and these will not be caught.
    """
    def __init__(self,
                 *,
                 dynamic_model: Module|Distribution,
                 observation_model: Module|Distribution,
                 prior_model: Module|Distribution,
                 initial_proposal_model: Module|Distribution|None = None,
                 proposal_model: Module|Distribution|None = None):
        super().__init__()
        if isinstance(observation_model, Distribution):
           self.observation_model = _obs_from_model(observation_model)
        else:
            self.observation_model = observation_model
        if isinstance(dynamic_model, Distribution):
            self.dynamic_model = _dyn_from_model(dynamic_model)
        else:
            self.dynamic_model = dynamic_model
        if isinstance(prior_model, Distribution):
            self.prior_model = _prior_from_model(prior_model)
        else:
            self.prior_model = prior_model
        if isinstance(initial_proposal_model, Distribution):
            self.initial_proposal_model = _prior_from_model(initial_proposal_model)
        else:
            self.initial_proposal_model = initial_proposal_model
        if isinstance(proposal_model, Distribution):
            self.proposal_model = _dyn_from_model(proposal_model)
        else:
                self.proposal_model = proposal_model
        if not hasattr(self.observation_model, 'score'):
            raise AttributeError("The observation model must implement a 'score' method")
        
        if self.proposal_model is None:
            if not hasattr(self.dynamic_model, 'sample'):
                raise AttributeError("The dynamic model must implement a 'sample' method")
        else:
            if not hasattr(self.dynamic_model, 'log_density'):
                raise AttributeError("The dynamic model must implement a 'log_density' method")
            if not hasattr(self.proposal_model, 'log_density'):
                raise AttributeError("The proposal model must implement a 'log_density' method")
            if not hasattr(self.proposal_model, 'sample'):
                raise AttributeError("The proposal model must implement a 'sample' method")
            
        if self.initial_proposal_model is None:
            if not hasattr(self.prior_model, 'sample'):
                raise AttributeError("The observation model must implement a 'sample' method")
        else:
            if not hasattr(self.prior_model, 'log_density'):
                raise AttributeError("The prior model must implement a 'log_density' method")
            if not hasattr(self.initial_proposal_model, 'propose'):
                raise AttributeError("The initial sample model must implement a 'sample' method")
            if not hasattr(self.initial_proposal_model, 'log_density'):
                raise AttributeError("The initial proposal model must implement a 'log_density' method")

    @property
    def is_bootstrap(self):
        """True if the model doesn't have either an initial_proposal_model or proposal_model"""
        return self.proposal_model is None and self.initial_proposal_model is None

    @property
    def has_proposal(self):
        """True if the model has a proposal_model"""
        return self.proposal_model is not None

    @property
    def has_initial_proposal(self):
        """True if the model has an initial_proposal_model"""
        return self.initial_proposal_model is not None