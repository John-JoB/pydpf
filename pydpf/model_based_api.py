from .base import Module
from typing import Union
from torch import Tensor
from .distributions import Distribution

class obs_from_model(Module):
    def __init__(self, dist:Distribution):
        super().__init__()
        self.dist = dist

    def score(self, state:Tensor, observation:Tensor, **data) -> Tensor:
        return self.dist.log_density(sample = observation.unsqueeze(1), condition_on=state)

    def sample(self, state:Tensor, **data) -> Tensor:
        return self.dist.sample(condition_on = state)

class dyn_from_model(Module):
    def __init__(self, dist:Distribution):
        super().__init__()
        self.dist = dist

    def log_density(self, prev_state:Tensor, state:Tensor, **data)->Tensor:
        return self.dist.log_density(condition_on=prev_state, sample=state)

    def sample(self, prev_state:Tensor, **data) -> Tensor:
        return self.dist.sample(condition_on=prev_state)

class prior_from_model(Module):
    def __init__(self, dist:Distribution):
        super().__init__()
        self.dist = dist

    def log_density(self, state:Tensor, **data)->Tensor:
        return self.dist.log_density(sample=state)

    def sample(self, batch_size:int, n_particles:int, **data) -> Tensor:
        return self.dist.sample(sample_size=(batch_size, n_particles))


class FilteringModel(Module):
    def __init__(self, *, dynamic_model:Union[Module,Distribution],
                 observation_model:[Module,Distribution],
                 prior_model:[Module,Distribution],
                 initial_proposal_model: Module = None,
                 proposal_model: Module= None):
        super().__init__()
        if isinstance(observation_model, Distribution):
           self.observation_model = obs_from_model(observation_model)
        else:
            self.observation_model = observation_model
        if isinstance(dynamic_model, Distribution):
            self.dynamic_model = dyn_from_model(dynamic_model)
        else:
            self.dynamic_model = dynamic_model
        if isinstance(prior_model, Distribution):
            self.prior_model = prior_from_model(prior_model)
        else:
            self.prior_model = prior_model
        if isinstance(initial_proposal_model, Distribution) or isinstance(proposal_model, Distribution):
            #Don't allow proposal to be a Distribution as there's no obvious/intuitive way to marry the APIs.
            raise TypeError('The non-bootstrap proposals cannot be Distribution objects.')
        self.proposal_model =  proposal_model
        self.initial_proposal_model = initial_proposal_model
        if not hasattr(self.observation_model, 'score'):
            raise AttributeError("The observation model must implement a 'score' method")
        
        if self.proposal_model is None:
            if not hasattr(self.dynamic_model, 'sample'):
                raise AttributeError("The dynamic model must implement a 'predict' method")
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
        return self.proposal_model is None and self.initial_proposal_model is None

    @property
    def has_proposal(self):
        return self.proposal_model is not None

    @property
    def has_initial_proposal(self):
        return self.initial_proposal_model is not None

    def get_prior_IS(self):
        if self.initial_proposal_model is None:
            def prior(n_particles, observation, **data):
                state = self.prior_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data)
                weight = self.observation_model.score(state = state, observation = observation, **data)
                return state, weight
        else:
            def prior(n_particles, observation, **data):
                state = self.initial_proposal_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data)
                weight = (self.observation_model.score(state = state, observation = observation, **data)
                          - self.initial_proposal_model.log_density(state = state, observation = observation, **data)
                          + self.prior_model.log_density(state = state, **data))
                return state, weight
        return prior


    def get_prop_IS(self):
        if self.proposal_model is None:
            def prop(prev_state, prev_weight, observation, **data):
                new_state = self.dynamic_model.sample(prev_state = prev_state, **data)
                new_weight = prev_weight + self.observation_model.score(new_state, observation = observation, **data)
                return new_state, new_weight
        else:
            def prop(prev_state, prev_weight, observation, **data):
                new_state = self.dynamic_model.sample(prev_state = prev_state, **data)
                new_weight = (prev_weight + self.observation_model.score(state = new_state, observation = observation, **data)
                              - self.proposal_model.log_density(state = new_state, prev_state = prev_state, observation = observation, **data)
                              + self.dynamic_model.log_density(state = new_state, prev_state = prev_state, **data))
                return new_state, new_weight
        return prop