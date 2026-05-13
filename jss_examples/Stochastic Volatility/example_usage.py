import torch
import pydpf
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = Path("./data/example_usage.csv")

class SVDynamicModel(pydpf.Module):

    def __init__(self, alpha, sigma, device):
        super().__init__()
        self.alpha_ = torch.nn.Parameter(alpha)
        self.log_sigma = torch.nn.Parameter(torch.log(sigma))
        self.device = device

    @pydpf.constrained_parameter
    def alpha(self):
        return self.alpha_, torch.clip(self.alpha_, 1e-3, 1-1e-3)

    @pydpf.cached_property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def sample(self, prev_state, **data):
        state_size = prev_state.size()
        noise = self.sigma * torch.normal(0, 1, device=self.device, size=state_size)
        return prev_state * self.alpha + noise

class SVObservationModel(pydpf.Module):

    def __init__(self, beta, device):
        super().__init__()
        self.log_beta = torch.nn.Parameter(torch.log(beta))
        self.half_log_2pi = torch.log(torch.tensor(2*torch.pi, device = device))/2
        self.device = device

    @pydpf.cached_property
    def beta(self):
        return torch.exp(self.log_beta)

    #Note: the evaluation function for the observation model is called 'score'
    #rather than 'log_density' as there is no requirement for this to be a
    #valid Markov kernel, and frequently for DPFs it isn't
    def score(self, state, observation, **data):
        log_root_v = state + self.log_beta
        root_v = torch.exp(log_root_v)
        #Observations are independent of the particle so have one less
        #dimension than the particle dependent state, we unsqueeze
        #this dimension to broadcast over the particles.
        normalised_obs = observation.unsqueeze(1) / root_v
        return (-log_root_v - (normalised_obs**2)/2 - self.half_log_2pi).squeeze()

    def sample(self, state, **data):
        log_root_v = state + self.log_beta
        root_v = torch.exp(log_root_v)
        state_size = state.size()
        return root_v * torch.normal(0, 1, device=self.device, size=state_size)

class SVPriorModel(pydpf.Module):

    def __init__(self, dynamic_model):
        super().__init__()
        self.device = dynamic_model.device
        self.dyn_mod = dynamic_model

    @pydpf.cached_property
    def sd(self):
        return torch.sqrt(self.dyn_mod.sigma**2 / (1-self.dyn_mod.alpha**2))

    def sample(self, batch_size, n_particles, **data):
        state_size = (batch_size, n_particles, 1)
        return self.sd * torch.normal(0, 1, device=self.device, size=state_size)

def make_SSM(alpha, beta, sigma, device):
    dynamic = SVDynamicModel(alpha, sigma, device)
    observation = SVObservationModel(beta, device)
    prior = SVPriorModel(dynamic)
    return pydpf.FilteringModel(prior_model=prior,
                                dynamic_model=dynamic,
                                observation_model=observation)

if __name__ == "__main__":
    SSM = make_SSM(torch.tensor(0.91, device=device),
                   torch.tensor(0.5, device=device),
                   torch.tensor(1., device=device),
                   device)
    # data_path must have a .csv extension
    pydpf.simulate_and_save(data_path,
                            SSM=SSM,
                            time_extent=100,
                            n_trajectories=200,
                            batch_size=100,
                            device=device)
    learned_SSM = make_SSM(torch.tensor(0.6, device=device),
                           torch.tensor(0.2, device=device),
                           torch.tensor(1.5, device=device),
                           device)
    # The generator parameter is a torch RNG generator.
    # Used to track the random state if reproducibility is required.
    multinomial_base = pydpf.MultinomialResampler(generator=torch.Generator(device=device))
    soft_resampler = pydpf.SoftResampler(softness=0.7,
                                         base_resampler=multinomial_base,
                                         device=device)
    DPF = pydpf.ParticleFilter(soft_resampler, learned_SSM)
    full_dataset = pydpf.StateSpaceDataset(data_path,
                                           series_id_column="series_id",
                                           state_prefix="state",
                                           observation_prefix="observation",
                                           device=device)
    train_set, test_set = torch.utils.data.random_split(full_dataset, [0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=full_dataset.collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=full_dataset.collate)
    output_function = pydpf.MSE_Loss()
    opt = torch.optim.Adam(DPF.parameters(), lr=0.01)
    n_epochs = 50

    alpha_error = []
    beta_error = []
    sigma_error = []
    for e in range(n_epochs):
        train_loss = 0.0
        for state, observation in train_loader:
            opt.zero_grad()
            DPF.update()
            MSE = DPF(n_particles=64,
                      time_extent=100,
                      aggregation_function=output_function,
                      observation=observation,
                      ground_truth=state)
            loss = MSE.mean()
            loss.backward()
            train_loss += loss.item()
            alpha_error.append(torch.abs(learned_SSM.dynamic_model.alpha - 0.91).item())
            beta_error.append(torch.abs(learned_SSM.observation_model.beta - 0.5).item())
            sigma_error.append(torch.abs(learned_SSM.dynamic_model.sigma - 1.).item())
            opt.step()
        if e % 10 == 0:
            print(f"Epoch {e + 1}, loss: {train_loss / len(train_loader)}")

    DPF.update()
    with torch.inference_mode():
        mean_loss = 0.0
        for state, observation in test_loader:
            MSE = DPF(n_particles=64,
                      time_extent=100,
                      aggregation_function=output_function,
                      observation=observation,
                      ground_truth=state)
            mean_loss += MSE.mean().item()

    print(f"Test MSE: {mean_loss / len(test_loader)}")
    print(f"Learned alpha: {learned_SSM.dynamic_model.alpha.item()}")
    print(f"Learned beta: {learned_SSM.observation_model.beta.item()}")
    print(f"Learned sigma: {learned_SSM.dynamic_model.sigma.item()}")
    plt.plot(np.array(alpha_error))
    plt.plot(np.array(beta_error))
    plt.plot(np.array(sigma_error))
    plt.show()