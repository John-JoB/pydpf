import torch
import pydpf

class MazePrior(pydpf.Module):
    def __init__(self, width, height, generator):
        super().__init__()
        self.size_tensor = torch.tensor([width, height, torch.pi*2], device=generator.device)
        self.generator = generator

    def sample(self, n_particles, batch_size, **data):
        return (torch.rand((batch_size, n_particles, 3), generator=self.generator, device=self.generator.device) -0.5) * self.size_tensor[None, None, :]


class MazePriorCheat(pydpf.Module):
    def __init__(self):
        super().__init__()

    def sample(self, n_particles, batch_size, control, **data):
        return control.unsqueeze(1).expand(-1, n_particles, -1)


class MazeDynamic(pydpf.Module):
    def __init__(self, generator, cov):
        super().__init__()
        self.generator = generator
        self.dist = pydpf.MultivariateGaussian(torch.zeros(3, device=generator.device), cov, generator=generator, diagonal_cov=True)


    def deteriministic_action(self, prev_state, control):
        angle_i = prev_state[:, :, 2]
        c = torch.cos(angle_i)
        s = torch.sin(angle_i)
        rotation_matrix = torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)
        new_pos = torch.einsum('bnij, bj -> bni', rotation_matrix, control[:, :2]) + prev_state[:,:,:2]
        new_angle = prev_state[:, :, 2:3] + control[:, None, 2:3]
        return torch.concat([new_pos, new_angle], dim=-1)

    def sample(self, prev_state, control, **data):
         return self.deteriministic_action(prev_state, control) + self.dist.sample(sample_size=(prev_state.shape[0], prev_state.shape[1]))


    def log_density(self, prev_state, control, state, **data):
        return self.dist.log_density(state - self.deteriministic_action(prev_state, control))

class MazeObservation(pydpf.Module):
    def __init__(self, flow_model, encoder, decoder, state_encoder):
        super().__init__()
        self.flow_model = flow_model
        self.encoder = encoder
        self.decoder = decoder
        self.state_encoder = state_encoder

    def score(self, state, observation, **data):
        b, n , _ = state.shape
        encoded_state = self.state_encoder(state)
        return self.flow_model.log_density(observation.unsqueeze(1).expand(-1, n, -1), encoded_state)

class SimpleMazeObservation(pydpf.Module):
    def __init__(self, encoder, decoder, state_encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.state_encoder = state_encoder

    def score(self, state, observation, **data):
        b, n, _ = state.shape
        encoded_state = self.state_encoder(state)
        return -torch.log(2 - torch.cosine_similarity(encoded_state, observation.unsqueeze(1), dim=-1))

class MazeProposal(pydpf.Module):
    def __init__(self, flow_model, dynamic_model):
        super().__init__()
        self.dynamic_model = dynamic_model
        self.flow_model = flow_model

    def sample(self, prev_state, control, observation, **data):
        observation = observation.unsqueeze(1).expand(-1, prev_state.shape[1], -1)
        dynamic_evo = self.dynamic_model.sample(prev_state, control)
        output = self.flow_model.inverse(dynamic_evo, observation)[0]

        return output

    def log_density(self, prev_state, control, state, observation, **data):
        observation = observation.unsqueeze(1).expand(-1, prev_state.shape[1], -1)
        noised_action,  nf_log_det = self.flow_model.forward(state, observation)
        return self.dynamic_model.log_density(prev_state=prev_state, control=control, state=noised_action) + nf_log_det





