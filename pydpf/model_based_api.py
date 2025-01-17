from torch.onnx.symbolic_opset11 import unsqueeze
from torch.optim import RMSprop

from .base import Module
from typing import Union
import torch

class FilteringModel(Module):
    def __init__(self, *, dynamic_model:Module, observation_model:Module, prior_model:Module, initial_proposal_model: Union[Module, None] = None, proposal_model: Union[Module, None] = None):
        super().__init__()
        self.dynamic_model = dynamic_model
        self.observation_model = observation_model
        self.prior_model = prior_model
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

class SVPFModel(Module):

    @staticmethod
    def rbf(dist, h):
        return torch.exp(-(1 / h) * torch.sum(dist ** 2, dim=-1))

    @staticmethod
    def d_rbf(dist, h):
        return (2 / h) * (dist) * SVPFModel.rbf(dist, h)

    def RMS_prop(self, x, d_x, prev_step_size):
        with torch.no_grad():
            step_size = self.alpha * prev_step_size  + (1 - self.alpha) * d_x**2
        return x + self.lr * d_x/(torch.sqrt(step_size.detach()) + 1e-8), step_size

    def SVGD(self, x, post_grad):
        N = torch.tensor(x.size(1))
        inv_N = 1/N
        with torch.no_grad():
            med = torch.median(torch.cdist(x, x).flatten(-2), dim=-1)[0]
            h = med**2/torch.log(N)
        step_size = 0
        for i in range(self.iterations):
            unsqueeze_x_1 = x.unsqueeze(1)
            unsqueeze_x_2 = x.unsqueeze(2)
            dist = unsqueeze_x_2 - unsqueeze_x_1
            stein_grad = inv_N*torch.sum(self.rbf(dist, h) * post_grad(x) + self.d_rbf(dist, h), dim=-2)
            x, step_size = self.RMS_prop(x, stein_grad, step_size)
        return x


    def set_hyper_parameters(self, lr, alpha, iterations):
        self.lr = lr
        self.alpha = alpha
        self.iterations = iterations

    def __init__(self, *, dynamic_model: Module, observation_model: Module, prior_model: Module):
        super().__init__()
        self.dynamic_model = dynamic_model
        self.observation_model = observation_model
        self.prior_model = prior_model

    def get_prior_SV(self):
        def prior(n_particles, observation, **data):
            state = self.prior_model.sample(batch_size = observation.size(0), n_particles = n_particles, **data)
            post_grad = lambda x: self.prior_model.d_log_density(x, **data) + self.observation_model.d_score(x, observation, **data)
            return self.SVGD(state, post_grad), torch.zeros((state.size(0), state.size(1)), device = state.device), torch.ones((state.size(1)), device = state.device) * torch.log(torch.tensor(n_particles))

        return prior

    def post_grad(self, x, prev_state, observation, **data):
        x_expanded = x.unsqueeze(2).expand(-1, -1, x.size(1), -1).flatten(1, 2)
        prev_state_expanded = prev_state.unsqueeze(1).expand(-1, x.size(1), -1, -1).flatten(1, 2)
        log_dynamic = self.dynamic_model.log_density(state=x_expanded, prev_state=prev_state_expanded, **data).reshape(x.size(0), x.size(1), x.size(1))
        d_log_dynamic = self.dynamic_model.d_log_density(state=x_expanded, prev_state=prev_state_expanded, **data).reshape(x.size(0), x.size(1), x.size(1))
        max_log_dynamic = (torch.max(log_dynamic, dim=-1)[0]).detach()
        log_dynamic = log_dynamic - max_log_dynamic
        dynamic = torch.exp(log_dynamic)
        dynamic_prior_grad = torch.sum(dynamic * d_log_dynamic, dim=-1)/torch.sum(dynamic, dim=-1)
        return dynamic_prior_grad + self.observation_model.d_score(x, observation, **data)


    def get_prop_SV(self):
        def prop(prev_state, prev_weight, observation, **data):
            state = self.dynamic_model.sample(prev_state = prev_state, **data)
            post_grad = lambda x: self.post_grad(x, prev_state, observation, **data)
            return self.SVGD(state, post_grad), torch.zeros((state.size(0), state.size(1)), device = state.device), torch.ones((state.size(1)), device = state.device) * torch.log(torch.tensor(state.size(1)))

        return prop



