from .base import Module
from typing import Union


class FilteringModel(Module):
    def __init__(self, dynamic_model:Module, observation_model:Module, prior_model:Module, initial_proposal_model: Union[Module, None] = None, proposal_model: Union[Module, None] = None):
        super().__init__()
        self.dynamic_model = dynamic_model
        self.observation_model = observation_model
        self.prior_model = prior_model
        self.proposal_model =  proposal_model
        self.initial_proposal_model = initial_proposal_model
        if not hasattr(self.observation_model, 'score'):
            raise AttributeError("The observation model must implement a 'score' method")
        
        if self.proposal_model is None:
            if not hasattr(self.dynamic_model, 'predict'):
                raise AttributeError("The dynamic model must implement a 'predict' method")
            if not hasattr(self.observation_model, 'score'):
                raise AttributeError("The observation model must implement a 'score' method")      
        else:
            if not hasattr(self.dynamic_model, 'log_density'):
                raise AttributeError("The dynamic model must implement a 'log_density' method")
            if not hasattr(self.proposal_model, 'log_density'):
                raise AttributeError("The proposal model must implement a 'log_density' method")
            if not hasattr(self.proposal_model, 'propose'):
                raise AttributeError("The proposal model must implement a 'propose' method")
            
        if self.initial_proposal_model is None:
            if not hasattr(self.prior_model, 'sample'):
                raise AttributeError("The observation model must implement a 'sample' method")
        else:
            if not hasattr(self.prior_model, 'log_density'):
                raise AttributeError("The prior model must implement a 'log_density' method")
            if not hasattr(self.initial_proposal_model, 'propose'):
                raise AttributeError("The initial proposal model must implement a 'propose' method")
            if not hasattr(self.initial_proposal_model, 'log_density'):
                raise AttributeError("The initial proposal model must implement a 'log_density' method")

    def get_prior_IS(self):
        if self.initial_proposal_model is None:
            def prior(n_particles, data):
                state = self.prior_model.sample(n_particles, data)
                weight = self.observation_model.score(state, data, 0)
                return state, weight
        else:
            def prior(n_particles, data):
                state = self.initial_proposal_model.propose(n_particles, data)
                weight = self.observation_model.score(state, data, 0) - self.initial_proposal_model.log_density(state, data) + self.prior_model.log_density(state, data)
                return state, weight
        return prior


    def get_prop_IS(self):
        if self.proposal_model is None:
            def prop(state, weight, data, t):
                new_state = self.dynamic_model.predict(state, data, t)
                new_weight = weight + self.observation_model.score(new_state, data, t)
                return new_state, new_weight
        else:
            def prop(state, weight, data, t):
                new_state = self.dynamic_model.predict(state, data, t)
                new_weight = weight + self.observation_model.score(new_state, data, t) - self.proposal_model.log_density(new_state, state, data, t) + self.dynamic_model.log_density(new_state, state, data, t)
                return new_state, new_weight
        return prop