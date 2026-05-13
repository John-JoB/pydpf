from pathlib import Path
import pydpf
import torch
import argparse
import os
import pandas as pd
from tqdm import tqdm
from time import time
import numpy as np
from copy import deepcopy
from math import sqrt, ceil
from typing import Tuple
import requests
import zipfile
import shutil
from torch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out, uniform_
from torch import Tensor


#======================================================================
#Script set up
#=====================================================================
c_process = ""
implemented_experiments = ["learning_proposal_parameters", "fully_specified_model"]

def process_print(string:str):
    print(f'{c_process}: {string}')



parser = argparse.ArgumentParser(description='The master script for pydpf example uses from our JSS paper.')
parser.add_argument("-e", "--experiments", action="append", default=[], help=f"Experiments to run, must be chosen from {"".join(implemented_experiments)}")
parser.add_argument("-d", "--device", action="store", default="None", help="Device to store tensors. Default: GPU if CUDA GPU is available else CPU.")
parser.add_argument("-dd", "--data_dir", action="store", default="./data/", help="Directory for data.")
parser.add_argument("-rd", "--results_dir", action="store", default="./results/", help="Directory for results.")
parser.add_argument("-b", "--max_batch_size", action="store", default=torch.inf, type=int, help="Maximum batch size to use incase any routine does not run well on the user's hardware.")


def parse_args():
    args = parser.parse_args()
    device = args.device
    processed_args = {}
    if device == "None":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(device)
    except Exception as e:
        print(f"\033[91mError no device named {device}\033[0m")
        raise e
    processed_args["device"] = device

    experiments = args.experiments
    if len(experiments) == 0:
        experiments = implemented_experiments
    for experiment in experiments:
        if not experiment in implemented_experiments:
            raise KeyError(f"Trying to run non implemented experiment {experiment}, allowed experiments are: {''.join(implemented_experiments)}.")
    processed_args["experiments"] = experiments

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        data_dir.mkdir()
    elif not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory {data_dir} does not exist.")
    processed_args["data_dir"] = data_dir

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        results_dir.mkdir()
    elif not results_dir.is_dir():
        raise NotADirectoryError(f"Data directory {results_dir} does not exist.")
    processed_args["results_dir"] = results_dir

    processed_args["max_batch_size"] = args.max_batch_size

    return processed_args

#======================================================================
#Example usage Stoch Vol
#=====================================================================
class SVDynamicModel(pydpf.Module):

    def __init__(self, alpha, sigma, device, generator):
        super().__init__()
        self.alpha_ = torch.nn.Parameter(alpha)
        self.log_sigma = torch.nn.Parameter(torch.log(sigma))
        self.device = device
        self.generator = generator

    @pydpf.constrained_parameter
    def alpha(self):
        return self.alpha_, torch.clip(self.alpha_, 1e-3, 1-1e-3)

    @pydpf.cached_property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def sample(self, prev_state, **data):
        state_size = prev_state.size()
        noise = self.sigma * torch.normal(0, 1, device=self.device, size=state_size, generator=self.generator)
        return prev_state * self.alpha + noise

class SVDynamicModelLogistic(SVDynamicModel):
    def __init__(self, alpha, sigma, device, generator):
        super().__init__(alpha, sigma, device, generator)
        self.logistic_alpha = torch.logit(alpha, eps=1e-3)

    @pydpf.cached_property
    def alpha(self):
        return torch.sigmoid(self.logistic_alpha)

class SVObservationModel(pydpf.Module):

    def __init__(self, beta, device, generator):
        super().__init__()
        self.log_beta = torch.nn.Parameter(torch.log(beta))
        self.half_log_2pi = torch.log(torch.tensor(2*torch.pi, device = device))/2
        self.device = device
        self.generator = generator

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
        return root_v * torch.normal(0, 1, device=self.device, size=state_size, generator=self.generator)

class SVPriorModel(pydpf.Module):

    def __init__(self, dynamic_model, generator):
        super().__init__()
        self.device = dynamic_model.device
        self.dyn_mod = dynamic_model
        self.generator = generator

    @pydpf.cached_property
    def sd(self):
        return torch.sqrt(self.dyn_mod.sigma**2 / (1-self.dyn_mod.alpha**2))

    def sample(self, batch_size, n_particles, **data):
        state_size = (batch_size, n_particles, 1)
        return self.sd * torch.normal(0, 1, device=self.device, size=state_size, generator=self.generator)

def make_SSM(alpha, beta, sigma, device, generator, use_logistic=False):
    if use_logistic:
        dynamic = SVDynamicModelLogistic(alpha, sigma, device, generator)
    else:
        dynamic = SVDynamicModel(alpha, sigma, device, generator)
    observation = SVObservationModel(beta, device, generator)
    prior = SVPriorModel(dynamic, generator)
    return pydpf.FilteringModel(prior_model=prior,
                                dynamic_model=dynamic,
                                observation_model=observation)

def ex_usage_setup(data_dir, **kwargs):
    data_path = data_dir / "example_usage.csv"
    if data_path.is_file():
        os.remove(data_path)

def ex_usage_run_script(device, data_dir, max_batch_size, **kwargs):
    generator = torch.Generator(device=device).manual_seed(0)
    data_path = data_dir/"example_usage.csv"
    SSM = make_SSM(torch.tensor(0.91, device=device),
                   torch.tensor(0.5, device=device),
                   torch.tensor(1., device=device),
                   device,
                   generator)
    # data_path must have a .csv extension
    pydpf.simulate_and_save(data_path,
                            SSM=SSM,
                            time_extent=100,
                            n_trajectories=200,
                            batch_size=min(100, max_batch_size),
                            device=device)
    learned_SSM = make_SSM(torch.tensor(0.6, device=device),
                           torch.tensor(0.2, device=device),
                           torch.tensor(1.5, device=device),
                           device,
                           generator)
    # The generator parameter is a torch RNG generator.
    # Used to track the random state if reproducibility is required.
    multinomial_base = pydpf.MultinomialResampler(generator=generator)
    soft_resampler = pydpf.SoftResampler(softness=0.7,
                                         base_resampler=multinomial_base,
                                         device=device)
    DPF = pydpf.ParticleFilter(soft_resampler, learned_SSM)
    full_dataset = pydpf.StateSpaceDataset(data_path,
                                           series_id_column="series_id",
                                           state_prefix="state",
                                           observation_prefix="observation",
                                           device=device)
    cpu_gen = torch.Generator().manual_seed(0)
    train_set, test_set = torch.utils.data.random_split(full_dataset, [0.5, 0.5], generator=cpu_gen)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=min(32, max_batch_size), shuffle=True, collate_fn=full_dataset.collate, generator=cpu_gen)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=min(64, max_batch_size), shuffle=False, collate_fn=full_dataset.collate)
    output_function = pydpf.MSE_Loss()
    opt = torch.optim.Adam(DPF.parameters(), lr=0.01)
    n_epochs = 50

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
            opt.step()
        if e % 10 == 0:
            process_print(f"Epoch {e + 1}, loss: {train_loss / len(train_loader)}")

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
    print()
    process_print(f"Test MSE: {mean_loss / len(test_loader)}")
    process_print(f"Learned alpha: {learned_SSM.dynamic_model.alpha.item()}")
    process_print(f"Learned beta: {learned_SSM.observation_model.beta.item()}")
    process_print(f"Learned sigma: {learned_SSM.dynamic_model.sigma.item()}")


#==========================================================
#Comparison with the Kalman filter
#==========================================================
class GaussianDynamic(pydpf.Module):
    def __new__(cls, dx:int, generator):
        device = generator.device
        dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1) - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        dynamic_offset = torch.zeros(dx, device=device)
        return pydpf.LinearGaussian(weight=dynamic_matrix, bias=dynamic_offset, cholesky_covariance=torch.eye(dx, device=device), generator=generator)

class GaussianObservation(pydpf.Module):
    def __new__(cls, dx:int, dy:int, generator):
        device = generator.device
        observation_matrix = torch.zeros((dy, dx), device=device)
        for i in range(dy):
            observation_matrix[i, i] = 1
        observation_offset = torch.zeros(dy, device=device)
        return pydpf.LinearGaussian(weight=observation_matrix, bias=observation_offset, cholesky_covariance=torch.eye(dy, device=device), generator=generator)

class GaussianPrior(pydpf.Module):
    def __new__(cls, dx:int, generator):
        device = generator.device
        return pydpf.MultivariateGaussian(torch.zeros(dx, device=device), torch.eye(dx, device=device), generator=generator)

def make_new_csv(rows, columns, dir, name):
    file_path = dir / f"{name}.csv"
    if file_path.exists():
        process_print(f"File already exists at {file_path}, skipping creation")
        return
    df = pd.DataFrame(index=pd.Index(rows, name="method"), columns=columns)
    df.to_csv(file_path)

def make_model_componets(dx, dy, generator):
    dynamic_model = GaussianDynamic(dx, generator)
    observation_model = GaussianObservation(dx, dy, generator)
    prior_model = GaussianPrior(dx, generator)
    return prior_model, dynamic_model, observation_model

def comparison_to_Kalman_setup(results_dir, device, data_dir, max_batch_size, **data):
    make_new_csv(["Kalman Filter", "PF K = 25", "PF K = 100", "PF K = 1000", "PF K = 10000"],
                 ["Time CPU (s)", "Time GPU (s)", "epsilon x", "epsilon y"],
                 results_dir,
                 "Kalman_comparison_results")
    data_path = data_dir / "LG.csv"
    if data_path.is_file:
        os.remove(data_path)
    gen_generator = torch.Generator(device=device).manual_seed(0)
    prior_model, dynamic_model, observation_model = make_model_componets(25, 1, gen_generator)
    SSM = pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)
    pydpf.simulate_and_save(data_path, SSM=SSM, time_extent=1000, n_trajectories=2000, batch_size=min(100, max_batch_size), device=device)

def fractional_diff_exp(a, b):
    frac = b-a
    return torch.abs(1 - torch.exp(frac))

def run_with_device(device, data_path, batch_size, Ks, result_path):

    cuda = device.type == "cuda"

    cuda_gen = torch.Generator(device=device).manual_seed(0)
    cpu_gen = torch.Generator().manual_seed(0)
    dataset = pydpf.StateSpaceDataset(data_path=data_path,
                                      series_id_column='series_id',
                                      state_prefix='state',
                                      observation_prefix='observation',
                                      device=device)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate, generator=cpu_gen)
    prior_model, dynamic_model, observation_model = make_model_componets(25, 1, cuda_gen)
    multinomial_resampler = pydpf.MultinomialResampler(cuda_gen)
    SSM = pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)
    PF = pydpf.ParticleFilter(resampler=multinomial_resampler, SSM=SSM)
    KalmanFilter = pydpf.KalmanFilter(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)
    aggregation_function_dict = {'Means': pydpf.FilteringMean(), 'Likelihood_factors': pydpf.LogLikelihoodFactors()}

    for K in Ks:
        if K is None:
            process_print(f"Running Kalman filter")
        else:
            process_print(f"Testing {device.type} with K={K}")
        size = 0
        state_error = []
        kalman_time = []
        pf_time = []
        likelihood_error = []
        # Time the Kalman filter without running the particle filter in the same loop as timing seems to be dependent on K.
        for state, observation in tqdm(data_loader):
            with torch.inference_mode():
                size += state.size(1)
                if cuda:
                    torch.cuda.current_stream().synchronize()
                s_time = time()
                kalman_state, kalman_cov, kalman_likelihood = KalmanFilter(observation=observation, time_extent=1000)
                if cuda:
                    torch.cuda.current_stream().synchronize()
                kalman_time.append((time() - s_time))
                if not K is None:
                    if cuda:
                        torch.cuda.current_stream().synchronize()
                    s_time = time()
                    outputs = PF(observation=observation, n_particles=K, aggregation_function=aggregation_function_dict, time_extent=1000)
                    if cuda:
                        torch.cuda.current_stream().synchronize()
                    pf_time.append((time() - s_time))
                    state_sq_error = torch.sum((outputs['Means'] - kalman_state) ** 2, dim=-1).mean()
                    state_error.append(state_sq_error.item() * state.size(1))
                    log_abs_likelihood_error = fractional_diff_exp(kalman_likelihood, outputs['Likelihood_factors'].squeeze()).mean()
                    likelihood_error.append(log_abs_likelihood_error.item() * state.size(1))

        results_df = pd.read_csv(result_path, index_col=0)
        if not K is None:
            row_label = f'PF K = {K}'
            row = list(results_df.loc[row_label])
        kalman_row = list(results_df.loc['Kalman Filter'])
        if cuda:
            if K is None:
                kalman_row[1] = sum(kalman_time[1:-1]) / (len(data_loader) - 2)
                kalman_row[2] = 0.0
                kalman_row[3] = 0.0
            else:
                # Ignore first iteration as CUDA is often slower on the first pass, ignore the last iteration incase it had a different size
                row[1] = sum(pf_time[1:-1]) / (len(data_loader) - 2)
                row[2] = sum(state_error) / size
                row[3] = sum(likelihood_error) / size
        else:
            if K is None:
                kalman_row[0] = sum(kalman_time[1:-1]) / (len(data_loader) - 2)
            else:

                row[0] = sum(pf_time[1:-1]) / (len(data_loader) - 2)

        if not K is None:
            results_df.loc[row_label] = row
        results_df.loc['Kalman Filter'] = kalman_row
        results_df.to_csv(result_path)
    print(results_df)


def comparison_to_Kalman_run_script(results_dir, device, data_dir, max_batch_size, **data):
    data_path = data_dir / "LG.csv"
    result_path = results_dir / "Kalman_comparison_results.csv"
    Ks = [None, 25, 100, 1000, 10000]
    batch_size = min(128, max_batch_size)
    cpu = torch.device("cpu")
    if device.type == "cuda":
        run_with_device(device, data_path, batch_size, Ks, result_path)
    else:
        process_print(f"\033[91mSelected device is non CUDA type {device.type} cannot run the CUDA timing experiment.\033[0m")
    run_with_device(cpu, data_path, batch_size, Ks, result_path)

#=======================================================================
#Learning proposal parameters
#=======================================================================

class GaussianOptimalProposal(pydpf.Module):
    def __init__(self, dx:int, dy:int, generator):
        super().__init__()
        device = generator.device
        covariance = torch.eye(dx, device=device)
        self.dx = dx
        self.dy = dy
        for i in range(dy):
            covariance[i,i] = .5
        self.dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1) - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(dx, device=device), cholesky_covariance=torch.sqrt(covariance), generator=generator)

    def sample(self, observation, prev_state, **data):
        sample = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1)
        mean[:,:,:self.dy] = (mean[:,:,:self.dy] + observation.unsqueeze(1))/2
        return mean + sample

    def log_density(self, observation, prev_state, state, **data):
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1)
        mean[:, :, :self.dy] = (mean[:, :, :self.dy] + observation.unsqueeze(1))/2
        sample = state - mean
        return self.dist.log_density(sample)

class GaussianLearnedProposal(pydpf.Module):
    def __init__(self, dx:int, dy:int, generator):
        super().__init__()
        device = generator.device
        cov = torch.nn.Parameter(torch.eye(dx, device=device))
        self.dx = dx
        self.dy = dy
        self.dynamic_matrix = 0.38 ** (torch.abs(torch.arange(dx, device=device).unsqueeze(1) - torch.arange(dx, device=device).unsqueeze(0)) + 1)
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(dx, device=device), cholesky_covariance=cov, generator=generator)
        self.x_weight = torch.nn.Parameter(torch.ones(dx, device=device))
        self.y_weight = torch.nn.Parameter(torch.zeros(dy, device=device))
        self.dist = pydpf.MultivariateGaussian(mean=torch.zeros(dx, device=device), cholesky_covariance=cov, generator=generator, diagonal_cov=True)

    def sample(self, observation, prev_state, **data):
        sample = self.dist.sample((prev_state.size(0), prev_state.size(1)))
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1) * self.x_weight
        mean[:,:,:self.dy] = mean[:,:,:self.dy] + observation.unsqueeze(1) * self.y_weight
        return mean + sample

    def log_density(self, observation, prev_state, state, **data):
        mean = (self.dynamic_matrix @ prev_state.unsqueeze(-1)).squeeze(-1) * self.x_weight
        mean[:, :, :self.dy] = mean[:, :, :self.dy] + observation.unsqueeze(1) * self.y_weight
        sample = state - mean
        return self.dist.log_density(sample)


def chain(*its):
    it_list = []
    for it in its:
        it_list += list(it)
    return it_list


def rotate_range(c_repeat, rel_start, rel_end, repeats, total_elements):
    range_rotation_amount = (total_elements // repeats) * c_repeat
    start = (rel_start + range_rotation_amount) % total_elements
    end = (rel_end + range_rotation_amount) % total_elements
    if end == 0:
        return range(start, total_elements)
    if start > end:
        return chain(range(start, total_elements), range(0, end))
    return range(start, end)

def fractional_diff_exp(a, b):
    frac = b-a
    return torch.abs(1 - torch.exp(frac))

def test_dpf(dpf, test_loader, KalmanFilter):
    aggregation_fun = {'ELBO': pydpf.ElBO_Loss(), 'Filtering Mean': pydpf.FilteringMean(), 'Likelihood_factors': pydpf.LogLikelihoodFactors()}
    test_ELBO = []
    epsilon_x = []
    epsilon_l = []
    dpf.update()
    total_size = 0
    with torch.inference_mode():
        for state, observation in test_loader:
            outputs = dpf(n_particles = 100, time_extent=1000, aggregation_function=aggregation_fun, observation=observation)
            test_ELBO.append(outputs['ELBO'].sum().item() * state.size(1))
            kalman_state, kalman_cov, kalman_likelihood = KalmanFilter(observation=observation, time_extent=1000)
            epsilon_x.append(torch.sum((outputs['Filtering Mean'] - kalman_state)**2, dim=-1).mean().item() * state.size(1))
            log_abs_likelihood_error = fractional_diff_exp(kalman_likelihood, outputs['Likelihood_factors'].squeeze()).mean()
            epsilon_l.append(log_abs_likelihood_error.item() * state.size(1))
            total_size += state.size(1)
    return -sum(test_ELBO)/total_size, sum(epsilon_x)/total_size, sum(epsilon_l)/total_size

def max_wass_dist(x_weight, y_weight, prop_cov, device, dx, dy):
    optimal_x_weight = torch.ones(dx, device=device)
    optimal_x_weight[:dy] = .5
    optimal_cov = torch.ones(dx, device=device)
    for i in range(dy):
        optimal_cov[i] = .5
    a = x_weight - optimal_x_weight
    b = y_weight - .5
    if torch.all(a == 0):
        mean_div = torch.zeros(dy, device=device)
    else:
        mean_div = a ** 2 / torch.sum(a ** 2)
    if torch.all(b == 0):
        y_mean_div_contr = 0
    else:
        y_mean_div_contr = b ** 2 / torch.sum(b ** 2)
    mean_div[:dy] += y_mean_div_contr
    mean_div = torch.sum(mean_div ** 2)
    cov_div = torch.sum((optimal_cov + prop_cov - 2 * torch.sqrt(optimal_cov * prop_cov)))
    return mean_div + cov_div

def proposal_ex_make_model_componets(dx, dy, generator, optimal_prop = True):
    dynamic_model = GaussianDynamic(dx, generator)
    observation_model = GaussianObservation(dx, dy, generator)
    prior_model = GaussianPrior(dx, generator)
    if optimal_prop:
        proposal_model = GaussianOptimalProposal(dx, dy, generator)
    else:
        proposal_model = GaussianLearnedProposal(dx, dy, generator)
    return prior_model, dynamic_model, observation_model, proposal_model

def proposal_ex_get_DPF(DPF_type, SSM, dim, generator, device):
    if DPF_type == 'DPF':
        return pydpf.DPF(SSM=SSM, resampling_generator=generator)
    if DPF_type == 'Soft':
        return pydpf.SoftDPF(SSM=SSM, resampling_generator=generator)
    if DPF_type == 'Stop-Gradient':
        return pydpf.StopGradientDPF(SSM=SSM, resampling_generator=generator)
    if DPF_type == 'Marginal Stop-Gradient':
        return pydpf.MarginalStopGradientDPF(SSM=SSM, resampling_generator=generator)
    if DPF_type == 'Optimal Transport':
        return pydpf.OptimalTransportDPF(SSM=SSM, regularisation=0.5)
    if DPF_type == 'Kernel':
        kernel = pydpf.KernelMixture(pydpf.MultivariateGaussian(torch.zeros(25, device=device),torch.nn.Parameter(torch.eye(25, device=device)*0.1), diagonal_cov=True, generator=generator), generator=generator)
        return pydpf.KernelDPF(SSM=SSM, kernel=kernel)
    raise ValueError('DPF_type should be one of the allowed options')

def training_loop(dpf, epochs, train_loader):
    ELBO_fun = pydpf.ElBO_Loss()
    if experiment == 'Kernel':
        opt = torch.optim.SGD([{'params': [dpf.SSM.proposal_model.x_weight], 'lr': 0.01}, {'params': [dpf.SSM.proposal_model.y_weight, dpf.SSM.proposal_model.dist.cholesky_covariance], 'lr': 0.05}, {'params': dpf.resampler.parameters(), 'lr': 0.001}], lr=.5,
                              momentum=0.9, nesterov=True)
    elif experiment == 'Optimal Transport':
        opt = torch.optim.SGD([{'params': [dpf.SSM.proposal_model.x_weight], 'lr': 0.01}, {'params': [dpf.SSM.proposal_model.y_weight, dpf.SSM.proposal_model.dist.cholesky_covariance], 'lr': 0.05}], lr=.5, momentum=0.9, nesterov=True)
    else:
        opt = torch.optim.SGD([{'params': [dpf.SSM.proposal_model.x_weight], 'lr': 0.1}, {'params': [dpf.SSM.proposal_model.y_weight, dpf.SSM.proposal_model.dist.cholesky_covariance], 'lr': 0.5}], lr=.5, momentum=0.9, nesterov=True)
    opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    best_validation_loss = torch.inf
    for epoch in tqdm(range(epochs)):
        dpf.train()
        total_size = 0
        train_loss = []
        for state, observation in train_loader:
            dpf.update()
            opt.zero_grad()
            ELBO = dpf(100, 100, ELBO_fun, observation=observation)
            loss = torch.mean(ELBO)
            loss.backward()
            train_loss.append(loss.item() * state.size(1))
            opt.step()
            total_size += state.size(1)
            opt_scheduler.step()
        train_loss = np.sum(np.array(train_loss)) / total_size

        dpf.eval()
        dpf.update()
        total_size = 0
        validation_loss = []
        with torch.inference_mode():
            for state, observation in train_loader:
                ELBO = dpf(100, 100, ELBO_fun, observation=observation)
                loss = torch.mean(ELBO)
                validation_loss.append(loss.item() * state.size(1))
                total_size += state.size(1)
            validation_loss = np.sum(np.array(validation_loss)) / total_size

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_dict = deepcopy(dpf.state_dict())
        dpf.load_state_dict(best_dict)

def proposal_learning_setup(results_dir, device, data_dir, max_batch_size, **kwargs):
    make_new_csv(["Bootstrap", "Optimal", "DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"],
                 ["e_x", "e_l", "max W2", "ELBO"],
                 results_dir,
                 "proposal_learning_results")
    data_path = data_dir / "LG.csv"
    if data_path.is_file:
        os.remove(data_path)
    gen_generator = torch.Generator(device=device).manual_seed(0)
    prior_model, dynamic_model, observation_model = make_model_componets(25, 1, gen_generator)
    SSM = pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)
    pydpf.simulate_and_save(data_path, SSM=SSM, time_extent=1000, n_trajectories=2000, batch_size=min(100, max_batch_size), device=device)

def proposal_learning_run_script(data_dir, device, max_batch_size, results_dir, **kwargs):
    data_path = data_dir / "LG.csv"
    dx = 25
    dy = 1
    n_repeats = 5
    batch_size = min(32, max_batch_size)
    result_path = results_dir / "proposal_learning_results.csv"
    experiment_list = ['Bootstrap', 'Optimal', 'DPF', 'Soft', 'Stop-Gradient', 'Marginal Stop-Gradient', 'Optimal Transport', 'Kernel']
    dataset = pydpf.StateSpaceDataset(data_path=data_path,
                                      series_id_column='series_id',
                                      state_prefix='state',
                                      observation_prefix='observation',
                                      device=device)

    for experiment in experiment_list:
        process_print(f"Running {experiment}")
        mean_wass_dist = torch.tensor(0., device=device)
        mean_epsilon_l = 0
        mean_epsilon_x = 0
        mean_ELBO = 0
        for repeat in range(n_repeats):
            print(f"Repeat {repeat + 1} of {n_repeats}:")
            cpu_gen = torch.Generator().manual_seed(10 * repeat)
            cuda_gen = torch.Generator(device=device).manual_seed(10 * repeat)
            # Cyclically permute the indices of the dataset, so that across repeats we cover whole dataset during testing.
            train_set = torch.utils.data.Subset(dataset, rotate_range(repeat, 0, 1000, n_repeats, 2000))
            validation_set = torch.utils.data.Subset(dataset, rotate_range(repeat, 1000, 1500, n_repeats, 2000))
            prior_model, dynamic_model, observation_model, proposal_model = proposal_ex_make_model_componets(dx, dy, cuda_gen, experiment == 'Optimal')
            if experiment == 'Bootstrap':
                SSM = pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)
                dpf = proposal_ex_get_DPF('DPF', SSM, dx, generator=cuda_gen, device=device)
                mean_wass_dist += max_wass_dist(torch.ones(dx, device=device), torch.zeros(dy, device=device), torch.ones(dx, device=device), device=device, dx=dx, dy=dy)

            elif experiment == 'Optimal':
                SSM = pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model, proposal_model=proposal_model)
                dpf = proposal_ex_get_DPF('DPF', SSM, dx, generator=cuda_gen, device=device)
            else:
                trained_model = pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model, proposal_model=proposal_model)
                dpf = proposal_ex_get_DPF(experiment, trained_model, dx, generator=cuda_gen, device=device)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=cpu_gen, collate_fn=dataset.collate)
                training_loop(dpf, 20, train_loader)
                cholesky_prop_cov = torch.diag(proposal_model.dist.cholesky_covariance)
                prop_cov = cholesky_prop_cov ** 2
                mean_wass_dist += max_wass_dist(proposal_model.x_weight, proposal_model.y_weight, prop_cov, device=device, dx=dx, dy=dy)

            test_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, generator=cpu_gen, collate_fn=dataset.collate)
            kalman_filter = pydpf.KalmanFilter(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)
            ELBO, e_x, e_l = test_dpf(dpf, test_loader, kalman_filter)
            mean_ELBO += ELBO
            mean_epsilon_l += e_l
            mean_epsilon_x += e_x
        mean_wass_dist = sqrt(mean_wass_dist.item() / n_repeats)
        mean_ELBO = mean_ELBO / n_repeats
        mean_epsilon_x = mean_epsilon_x / n_repeats
        mean_epsilon_l = mean_epsilon_l / n_repeats
        results_df = pd.read_csv(result_path, index_col=0)
        row = np.array([mean_epsilon_x, mean_epsilon_l, mean_wass_dist, mean_ELBO])
        results_df.loc[experiment] = row
        results_df.to_csv(result_path)


#=======================================================================
#General stochastic volatility functions
#=======================================================================

def sv_set_up(data_dir, results_dir, max_batch_size, device, **kwargs):
    rows = ["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"]
    make_new_csv(rows, ["e_x", "e_l", "time"], results_dir, "fully_specified_results")

    data_path = data_dir / f"SV.csv"
    if data_path.is_file():
        os.remove(data_path)
    data_gen_generator = torch.Generator(device=device).manual_seed(0)
    alpha = torch.tensor(0.91, device=device)
    beta = torch.tensor(0.5, device=device)
    sigma = torch.tensor(1., device=device)
    SSM = make_SSM(alpha, beta, sigma, device, generator=data_gen_generator, use_logistic=True)
    pydpf.simulate_and_save(data_path, SSM=SSM, time_extent=1000, n_trajectories=500, batch_size=min(128, max_batch_size), device=device)


def SV_get_DPF(DPF_type, SSM, generator, device):
    if DPF_type == 'DPF':
        return pydpf.DPF(SSM=SSM, resampling_generator=generator)
    if DPF_type == 'Soft':
        return pydpf.SoftDPF(SSM=SSM, resampling_generator=generator, softness=0.7)
    if DPF_type == 'Stop-Gradient':
        return pydpf.StopGradientDPF(SSM=SSM, resampling_generator=generator)
    if DPF_type == 'Marginal Stop-Gradient':
        return pydpf.MarginalStopGradientDPF(SSM=SSM, resampling_generator=generator)
    if DPF_type == 'Optimal Transport':
        return pydpf.OptimalTransportDPF(SSM=SSM, regularisation=0.5, transport_gradient_clip=1.)
    if DPF_type == 'Kernel':
        kernel = pydpf.KernelMixture(pydpf.MultivariateGaussian(torch.zeros(1, device=device),torch.eye(1, device=device)*0.1, generator=generator), generator=generator)
        return pydpf.KernelDPF(SSM=SSM, kernel=kernel)
    raise ValueError('DPF_type should be one of the allowed options')

def _get_split_amounts(split, data_length):
    split_sum = sum(split)
    s =[0]*3
    s[0] = int(split[0]*data_length/split_sum)
    s[1] = int(split[1]*data_length/split_sum)
    s[2] = data_length - s[0] - s[1]
    if s[0] < 1:
        raise ValueError(f'Trying to assign too small a fraction to the train set')
    if s[1] < 1:
        raise ValueError(f'Trying to assign too small a fraction to the validation set')
    if s[2] < 1:
        raise ValueError(f'Trying to assign too small a fraction to the test set')
    return s

def SV_full_train_loop(dpf,
          opt: torch.optim.Optimizer,
          dataset: torch.utils.data.Dataset,
          epochs: int,
          n_particles: Tuple[int, int, int],
          batch_size: Tuple[int, int, int],
          split_size: Tuple[float, float, float],
          likelihood_scaling:float = 1.,
          data_loading_generator: torch.Generator = torch.default_generator,
          gradient_regulariser = None,
          target:str = 'MSE',
          time_extent = None,
          lr_scheduler = None,
          ):

    batch_size = list(batch_size)

    aggregation_function = {'MSE': pydpf.MSE_Loss(), 'ELBO': pydpf.ElBO_Loss()}

    data_length = len(dataset)
    train_validation_test_split = _get_split_amounts(split_size, data_length)
    train_set, validation_set, test_set = torch.utils.data.random_split(dataset, train_validation_test_split, generator=data_loading_generator)
    if batch_size[0] == -1 or batch_size[0] > len(train_set):
        batch_size[0] = len(train_set)
    if batch_size[1] == -1 or batch_size[1] > len(validation_set):
        batch_size[1] = len(validation_set)
    if batch_size[2] == -1 or batch_size[2] > len(test_set):
        batch_size[2] = len(test_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size[0], shuffle=True, generator=data_loading_generator, collate_fn=dataset.collate)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size[1], shuffle=False, generator=data_loading_generator, collate_fn=dataset.collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size[2], shuffle=False, generator=data_loading_generator, collate_fn=dataset.collate)
    best_eval = torch.inf
    dpf.update()
    best_dict = deepcopy(dpf.state_dict())
    if time_extent is None:
        time_extent = dataset.observation.size(0)-1
    for epoch in range(epochs):

        train_loss = []
        total_size = 0
        dpf.train()
        for state, observation in train_loader:
            dpf.update()
            opt.zero_grad()
            loss = dpf(n_particles[0], time_extent, aggregation_function, observation=observation, ground_truth=state, gradient_regulariser = gradient_regulariser)
            loss = torch.mean(loss['ELBO'])*likelihood_scaling + (1-likelihood_scaling)*torch.mean(loss['MSE'])
            loss.backward()
            for p in dpf.parameters():
                torch.clamp_(p.grad, -1., 1.)
            train_loss.append(loss.item()*state.size(1))
            opt.step()
            total_size += state.size(1)
        train_loss = np.sum(np.array(train_loss)) / total_size
        if lr_scheduler is not None:
            lr_scheduler.step()
        dpf.update()
        dpf.eval()
        with torch.inference_mode():
            total_size = 0
            validation_MSE = []
            validation_ELBO = []
            for state, observation in validation_loader:
                loss = dpf(n_particles[1], time_extent, aggregation_function, observation=observation, ground_truth=state)
                validation_MSE.append(torch.mean(loss['MSE']).item()*state.size(1))
                validation_ELBO.append(torch.sum(loss['ELBO']).item()*state.size(1))
                total_size += state.size(1)
            validation_MSE= np.sum((np.array(validation_MSE))) / total_size
            validation_ELBO = np.sum((np.array(validation_ELBO))) / total_size

        if np.isnan(validation_MSE) or np.isnan(validation_ELBO):
            dpf.load_state_dict(best_dict)
            continue
        if target=='MSE':
            if validation_MSE < best_eval and not np.isnan(validation_MSE):
                best_eval = validation_MSE

                best_dict = deepcopy(dpf.state_dict())
        else:
            if validation_ELBO < best_eval and not np.isnan(validation_ELBO):
                best_eval = validation_ELBO
                best_dict = deepcopy(dpf.state_dict())


        process_print(f'epoch {epoch + 1}/{epochs}, train loss: {train_loss}, validation MSE: {validation_MSE}, validation ELBO: {-validation_ELBO}')
    total_size = 0
    with torch.inference_mode():
        test_MSE = []
        test_ELBO = []
        dpf.load_state_dict(best_dict)
        for state, observation in test_loader:
            loss = dpf(n_particles[1], time_extent, aggregation_function, observation=observation, ground_truth=state)
            test_MSE.append(torch.mean(loss['MSE']).item() * state.size(1))
            test_ELBO.append(torch.sum(loss['ELBO']).item() * state.size(1))
            total_size += state.size(1)
    test_MSE = np.sum((np.array(test_MSE))) / total_size
    test_ELBO = np.sum((np.array(test_ELBO))) / total_size
    print('')
    process_print(f'test MSE: {test_MSE}, test ELBO: {-test_ELBO}')
    return test_MSE, -test_ELBO


#=======================================================================
#Filtering a fully specified model -- Stochastic volatility
#=======================================================================

def full_specified_run_script(data_dir, results_dir, max_batch_size, device, **kwargs):
    data_path = data_dir / f"SV.csv"
    results_path = results_dir / f"full_specified_results.csv"
    experiments = ['DPF', 'Soft', 'Stop-Gradient', 'Marginal Stop-Gradient', 'Optimal Transport', 'Kernel']
    dataset = pydpf.StateSpaceDataset(data_path=data_path, series_id_column='series_id', state_prefix='state', observation_prefix='observation', device=device)
    alpha = torch.tensor(0.91, device=device)
    beta = torch.tensor(0.5, device=device)
    sigma = torch.tensor(1., device=device)
    batch_size = min(128, max_batch_size)
    for experiment in experiments:
        process_print(f"Testing {experiment}")
        experiment_cuda_rng = torch.Generator(device=device).manual_seed(0)
        experiment_cpu_rng = torch.Generator().manual_seed(0)
        size = 0
        pf_time = []
        MSE = []
        likelihood_error = []
        SSM = make_SSM(alpha, beta, sigma, device, generator=experiment_cuda_rng)
        dpf = SV_get_DPF(experiment, SSM, generator=experiment_cuda_rng, device=device)
        pf = pydpf.DPF(SSM=SSM, resampling_generator=experiment_cuda_rng, multinomial=True)
        aggregation_function = {'Likelihood': pydpf.LogLikelihoodFactors(), 'Filtering mean': pydpf.FilteringMean()}
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=experiment_cpu_rng, collate_fn=dataset.collate)
        for state, observation in tqdm(data_loader):
            with torch.inference_mode():
                size += state.size(1)
                true_outputs = pf(observation=observation, n_particles=10000, aggregation_function=aggregation_function, time_extent=1000)
                torch.cuda.synchronize()
                s_time = time()
                outputs = dpf(observation=observation, n_particles=100, aggregation_function=aggregation_function, time_extent=1000)
                torch.cuda.synchronize()
                pf_time.append((time() - s_time))
                MSE.append(torch.sum((true_outputs['Filtering mean'] - outputs['Filtering mean']) ** 2, dim=-1).mean().item() * state.size(1))
                likelihood_error.append(fractional_diff_exp(true_outputs['Likelihood'], outputs['Likelihood']).mean().item() * state.size(1))

        results = pd.read_csv(results_path, index_col=0)
        row = np.array([sum(MSE) / size, sum(likelihood_error) / size, sum(pf_time[1:-1]) / (len(data_loader) - 2)])
        results.loc[experiment] = row
        process_print(results)
        results.to_csv(results_path)

def SV_test_learning_params(experiment, device, data_dir, max_batch_size, alpha_only, **kwargs):
    n_repeats = 10
    alphas = np.empty(n_repeats)
    betas = np.empty(n_repeats)
    sigmas = np.empty(n_repeats)
    ELBOs = np.empty(n_repeats)
    batch_size_train = min(32, max_batch_size)
    batch_size_test = min(128, max_batch_size)
    temp_data_path = data_dir / "SV_temp.csv"
    true_alpha = torch.tensor(0.91, device=device)
    true_beta = torch.tensor(0.5, device=device)
    true_sigma = torch.tensor(1., device=device)
    for n in range(n_repeats):
        experiment_cuda_rng = torch.Generator(device).manual_seed(n*10)
        experiment_cpu_rng = torch.Generator().manual_seed(n*10)
        generation_rng = torch.Generator(device).manual_seed(n*10)
        true_SSM = make_SSM(true_alpha, true_beta, true_sigma, device, generation_rng, False)
        pydpf.simulate_and_save(temp_data_path, SSM=true_SSM, time_extent=1000, n_trajectories=500, batch_size=100, device=device, bypass_ask=True)
        alpha = torch.rand((), device=device, generator=experiment_cuda_rng)
        if alpha_only:
            beta = true_beta
            sigma = true_sigma
        else:
            beta = torch.rand((), device=device, generator=experiment_cuda_rng) * 2
            sigma = torch.rand((), device=device, generator=experiment_cuda_rng) * 5
        SSM = make_SSM(alpha, beta, sigma, device, generation_rng, False)
        dpf = SV_get_DPF(experiment, SSM, experiment_cuda_rng, device)
        if alpha_only:
            param_groups = [{'params':[alpha], 'lr':0.05}]
        else:
            param_groups = [{'params': [alpha], 'lr': 0.05}, {'params': [beta], 'lr': 0.1}, {'params': [sigma], 'lr': 0.25}]

        if experiment == 'Kernel':
            param_groups.append({'params':dpf.resampler.mixture.parameters(), 'lr':0.01})
        opt = torch.optim.SGD(param_groups)
        opt_schedule = torch.optim.lr_scheduler.ExponentialLR(opt, 0.95)
        dataset = pydpf.StateSpaceDataset(temp_data_path, state_prefix='state', device=device)
        _, ELBO = SV_full_train_loop(dpf,
                                     opt,
                                     dataset,
                                     10,
                                     (100, 100, 100),
                                     (batch_size_train, batch_size_test, batch_size_test),
                                     (0.5, 0.25, 0.25),
                                     1.,
                                     experiment_cpu_rng,
                                     target='ELBO',
                                     time_extent=100,
                                     lr_scheduler=opt_schedule)
        alphas[n] = alpha
        betas[n] = beta
        sigmas[n] = sigma
        ELBOs[n] = ELBO
        os.remove(temp_data_path)
    return alphas, betas, sigmas, ELBOs

#=======================================================================
#Unsupervised learning of a single parameter -- Stochastic volatility
#=======================================================================

def SV_single_param_set_up(results_dir, **kwargs):
    rows = ["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"]
    make_new_csv(rows, ["Forward Time (s)", "Backward Time (s)", "Gradient standard deviation", "alpha error"], results_dir, "single_parameter_results")

def sv_test_gradients(experiment, device, data_dir, max_batch_size, **kwargs):
    batch_size_test = min(128, max_batch_size)
    process_print(f"Testing Gradients of {experiment}")
    experiment_cuda_rng = torch.Generator(device).manual_seed(0)
    aggregation_function_dict = {'ELBO': pydpf.LogLikelihoodFactors()}
    test_dataset = pydpf.StateSpaceDataset(data_path=data_dir / "test_trajectory.csv", state_prefix='state', device='cuda')
    gradients = []
    size = 0
    alpha_p = torch.nn.Parameter(torch.tensor([[0.93]], dtype = torch.float32, device=device))
    alpha = torch.tensor(0.93, device=device)
    beta = torch.tensor(0.5, device=device)
    sigma = torch.tensor(1., device=device)
    SSM = make_SSM(alpha, beta, sigma, device, experiment_cuda_rng, True)
    DPF = SV_get_DPF(experiment, SSM, experiment_cuda_rng, device)
    forward_time = []
    backward_time = []
    state = test_dataset.state[:,0:1].expand((101, batch_size_test, 1)).contiguous()
    observation = test_dataset.observation[:,0:1].expand((101, batch_size_test, 1)).contiguous()
    for i in tqdm(range(2560//batch_size_test)):
        DPF.update()
        size += state.size(1)
        torch.cuda.synchronize()
        start = time()
        outputs = DPF(observation=observation, n_particles=100, ground_truth=state, aggregation_function=aggregation_function_dict, time_extent=100)
        ls = torch.mean(outputs['ELBO'], dim=0)
        loss = ls.mean()
        torch.cuda.synchronize()
        forward_time.append((time() - start))
        alpha_p.grad = None
        for i in range(len(ls)):
            ls[i].backward(retain_graph=True)
            gradients.append(alpha_p.grad.item())
            alpha_p.grad = None
        #Free the stored tensors
        torch.cuda.synchronize()
        start = time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time.append((time() - start))
    return forward_time, backward_time, gradients



def SV_single_param_run(results_dir, **kwargs):
    experiments = ["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"]
    results_path = results_dir / 'single_parameter_results.csv'
    for experiment in experiments:
        process_print(f"Testing {experiment}")
        results = pd.read_csv(results_path, index_col=0)
        process_print("Testing gradient variance")
        ft, bt, grads = sv_test_gradients(experiment, **kwargs)
        process_print("Testing learning alpha")
        alpha_list, _, _, _ = SV_test_learning_params(experiment, alpha_only=True, **kwargs)
        # Ignore first and last times because the last batch will have a different size to the rest, and cuda is often slower on the first iteration
        row = np.array([sum(ft[1:-1]) / (len(ft) - 2), sum(bt[1:-1]) / (len(bt) - 2), np.sqrt(np.var(grads)), np.mean(np.abs(alpha_list - 0.91))])
        results.loc[experiment] = row
        print(results)
        results.to_csv(results_path)


#=======================================================================
#Unsupervised learning of multiple parameters -- Stochastic volatility
#=======================================================================

def SV_multiple_param_set_up(results_dir, **kwargs):
    rows = ["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"]
    make_new_csv(rows, ["ELBO", "alpha error", "beta error", "sigma error"], results_dir, "multiple_parameter_results")

def SV_multiple_param_run(results_dir, **kwargs):
    experiments = ["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"]
    results_path = results_dir / 'multiple_parameter_results.csv'
    for experiment in experiments:
        process_print(f"Testing {experiment}")
        results = pd.read_csv(results_path, index_col=0)
        alpha_list, beta_list, sigma_list, ELBO_list = SV_test_learning_params(experiment, alpha_only=False, **kwargs)
        # Ignore first and last times because the last batch will have a different size to the rest, and cuda is often slower on the first iteration
        row = np.array([np.mean(ELBO_list), np.mean(np.abs(alpha_list - 0.91)), np.mean(np.abs(beta_list - 0.5)), np.mean(np.abs(sigma_list - 1.))])
        results.loc[experiment] = row
        print(results)
        results.to_csv(results_path)

#==============================================================================
#Deep learning -- visual localisation
#==============================================================================

def download_dataset(folder_path):
    data_url = 'https://depositonce.tu-berlin.de/bitstreams/fe02c1e0-64d9-4a92-ac4d-a8a0ef455c8f/download'
    download_path = folder_path / "raw_zip.zip"
    with requests.get(data_url, stream=True) as r:
        total = int(r.headers.get('content-length', 0))
        with open(download_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc="Downloading data") as bar:
            for chunk in r.iter_content(chunk_size=1024):
                size = f.write(chunk)
                bar.update(size)
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)
    unzipped_path = folder_path / "data"
    file_one_start = unzipped_path / "100s/nav03_test.npz"
    file_two_start = unzipped_path / "100s/nav03_train.npz"
    file_one_end = folder_path / "maze_data_raw.npz"
    file_two_end = folder_path / "maze_data_raw_2.npz"
    file_one_start.rename(file_one_end)
    file_two_start.rename(file_two_end)
    shutil.rmtree(unzipped_path)
    download_path.unlink()

def bind_angle(angle):
    bound_angle = torch.remainder(angle, 2 * torch.pi)
    return torch.where(bound_angle > torch.pi, bound_angle - 2 * torch.pi, bound_angle)


def create_actions_and_modify_state(pos):
    pos[:, 2] = pos[:, 2] * np.pi /180
    pos[:, 2] = bind_angle(pos[:, 2])
    diffs = torch.empty_like(pos)
    diffs[1:, :] = pos[1:, :] - pos[:-1, :]
    diffs[::100, :]  = 0
    angles = torch.roll(pos[:, 2], 1, 0)
    c = torch.cos(angles)
    s = torch.sin(angles)
    rotation_matrix = torch.stack([torch.stack([c, s], dim=-1), torch.stack([-s, c], dim=-1)], dim=-2)
    actions = torch.empty_like(pos)
    actions[:, :2] = (rotation_matrix @ diffs[:, :2].unsqueeze(-1)).squeeze(-1)
    actions[:, 2] = bind_angle(diffs[:, 2])
    return actions


def create_observations(obs):
    #Create randomly cropped observations and drop the depth channel.
    new_o = torch.zeros([obs.shape[0], 24, 24, 3],  device=obs.device, dtype=torch.uint8)
    rng = np.random.Generator(np.random.PCG64(seed=0))
    for i in range(obs.shape[0]):
            offsets = rng.integers(low=0, high=9, size=2)
            new_o[i] = obs[i, offsets[0]:offsets[0] + 24, offsets[1]:offsets[1] + 24, :3]
    new_o = new_o.to(dtype=torch.float16)
    new_o = torch.round(torch.clip(new_o, 0, 255)).to(dtype=torch.uint8)
    new_o = new_o.permute(0, 3, 1, 2)
    return new_o.flatten(start_dim=1)


def create_df(data, label):
    data = data.cpu().numpy()
    df = pd.DataFrame(data, columns=[f'{label}_{i+1}' for i in range(data.shape[1])])
    series_id = np.arange(2000).repeat(100)
    df['series_id'] = series_id
    return df


def dm_setup(results_dir, data_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete_raw", action="store_true", help='Delete raw data when done')
    make_new_csv(["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"],
                 ["Total time (hrs:min:s)", "Test MSE"],
                 results_dir,
                 "deep_mind_maze_results")
    make_new_csv(["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"],
                 ["Total time (hrs:min:s)", "Test MSE"],
                 results_dir,
                 "nondeterministic_deep_mind_maze_results")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_raw_path_1 = data_dir / 'maze_data_raw.npz'
    data_raw_path_2 = data_dir / 'maze_data_raw_2.npz'
    if not (data_raw_path_1.exists() and data_raw_path_2.exists()):
        download_dataset(data_dir)
    data1 = dict(np.load(data_raw_path_1, allow_pickle=True))
    data2 = dict(np.load(data_raw_path_2, allow_pickle=True))
    #The raw data is a dictionary with 2 fields, 'pose' and 'rgbd'
    #The raw data consists of 1000 trajectories of 100 time-steps concatenated end-to-end per file.
    state = torch.tensor(np.concatenate((data1['pose'], data2['pose']), axis=0), device=device, dtype=torch.float32)
    observation = torch.tensor(np.concatenate((data1['rgbd'], data2['rgbd']), axis=0), device=device, dtype=torch.uint8)
    actions = create_actions_and_modify_state(state)
    control_df = create_df(actions, 'control')
    observation = create_observations(observation)
    observation_df = create_df(observation, 'observation')
    state_df = create_df(state, 'state')
    observation_df.drop(columns=['series_id'], inplace=True)
    control_df.drop(columns=['series_id'], inplace=True)
    total_df = pd.merge(control_df, state_df, left_index=True, right_index=True)
    total_df = pd.merge(total_df, observation_df, left_index=True, right_index=True)
    total_df.to_csv(data_dir / 'maze_data.csv', index=False)

'''
The authors thank Xiongjie Chen for kindly providing the code for his paper 'Normalizing Flow-based Differentiable Particle Filters' which
we have heavily based our neural networks on.
'''

class SeedableConv2D(torch.nn.Conv2d):
    '''
    Pytorch doesn't allow Conv2D to be initialised with a random generator.
    We implement this behaviour.
    '''

    def __init__(self, *args, generator, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_reset_parameters(generator)

    def seeded_reset_parameters(self, generator) -> None:
        kaiming_uniform_(self.weight, a=sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / sqrt(fan_in)
                uniform_(self.bias, -bound, bound, generator=generator)


class SeedableLinear(torch.nn.Linear):
    '''
        Pytorch doesn't allow Linear to be initialised with a random generator.
        We implement this behaviour.
        '''

    def __init__(self, *args, generator, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_reset_parameters(generator)

    def seeded_reset_parameters(self, generator) -> None:
        kaiming_uniform_(self.weight, a=sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            uniform_(self.bias, -bound, bound, generator=generator)

class SeedableConvTranspose2D(torch.nn.ConvTranspose2d):
    '''
    Pytorch doesn't allow ConvTranspose2D to be initialised with a random generator.
    We implement this behaviour.
    '''

    def __init__(self, *args, generator, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_reset_parameters(generator)

    def seeded_reset_parameters(self, generator) -> None:
        kaiming_uniform_(self.weight, a=sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / sqrt(fan_in)
                uniform_(self.bias, -bound, bound, generator=generator)

class SeedableDropoutNd(pydpf.Module):
        __constants__ = ["p", "inplace"]
        p: float
        inplace: bool


        def __init__(self, p: float = 0.5, inplace: bool = False, generator:torch.Generator = torch.default_generator) -> None:
            super().__init__()
            if p < 0 or p > 1:
                raise ValueError(
                    f"dropout probability has to be between 0 and 1, but got {p}"
                )
            self.p = p
            if self.p < 0.0 or self.p > 1.0:
                raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
            self.inplace = inplace
            self.generator = generator

        def extra_repr(self) -> str:
            return f"p={self.p}, inplace={self.inplace}"


class SeedableDropout(SeedableDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input
        mask = (torch.rand(input.size(), device=input.device, dtype=torch.float32, generator = self.generator) > self.p).to(input.dtype)
        return (
            input.multiply_(mask) if self.inplace else input * mask
        )

class SeedableDropout2d(SeedableDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input
        if input.dim() != 4:
            raise ValueError('SeedableDropout2d only supports Batch,Channel,Height,Width tensors')
        mask = pydpf.multiple_unsqueeze((torch.rand((input.size(0), input.size(1)), device=input.device, dtype=torch.float32, generator = self.generator) > self.p).to(input.dtype), 2, -1)
        return (
            input.multiply_(mask) if self.inplace else input * mask
        )


class FCNN(pydpf.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, generator):
        super().__init__()
        self.network = torch.nn.Sequential(
            SeedableLinear(in_dim, hidden_dim, generator = generator, device=generator.device),
            torch.nn.Tanh(),
            SeedableLinear(hidden_dim, hidden_dim, generator = generator, device=generator.device),
            torch.nn.Tanh(),
            SeedableLinear(hidden_dim, out_dim, generator = generator, device=generator.device),
        )

    def forward(self, x):
        return self.network(x)


class ObservationEncoder(pydpf.Module):
    def __new__(cls, hidden_size, dropout_keep_ratio = 0.7, generator = torch.default_generator):
        encode = torch.nn.Sequential(  # input: 3*24*24
            SeedableConv2D(3, 16, kernel_size=4, stride=2, padding=1, bias=False, generator = generator, device=generator.device),  # 16*12*12
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(16),
            SeedableConv2D(16, 32, kernel_size=4, stride=2, padding=1, bias=False, generator = generator, device=generator.device),  # 32*6*6
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            SeedableConv2D(32, 64, kernel_size=4, stride=2, padding=1, bias=False, generator = generator, device=generator.device),  # 64*3*3
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(64),
            torch.nn.Flatten(),
            SeedableDropout(p=1 - dropout_keep_ratio, generator = generator),
            SeedableLinear(64 * 3 * 3, hidden_size, generator = generator, device=generator.device),
        )
        return encode


class ObservationDecoder(pydpf.Module):
    def __new__(cls, hidden_size, dropout_keep_ratio = 0.7, generator = torch.default_generator):
        decode = torch.nn.Sequential(
            SeedableLinear(hidden_size, 3 * 3 * 64, generator = generator, device=generator.device),
            torch.nn.ReLU(True),
            torch.nn.Unflatten(-1, (64, 3, 3)),  # -1 means the last dim, (64, 3, 3)
            SeedableConvTranspose2D(64, 32, kernel_size=4, padding=1, stride=2, bias=False, generator = generator, device=generator.device),  # (32, 6,6)
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            SeedableConvTranspose2D(32, 16, kernel_size=4, padding=1, stride=2, bias=False, generator = generator, device=generator.device),  # (16, 12,12)
            SeedableDropout2d(p=1 - dropout_keep_ratio, generator=generator),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(16),
            SeedableConvTranspose2D(16, 3, kernel_size=4, padding=1, stride=2, bias=False, generator = generator, device=generator.device),  # (3, 24, 24)
            SeedableDropout2d(p=1 - dropout_keep_ratio, generator=generator),
            torch.nn.BatchNorm2d(3),
            torch.nn.Sigmoid()
        )
        return decode

class StateEncoder(pydpf.Module):
    def __new__(cls, hidden_size, dropout_keep_ratio = 0.7, generator = torch.default_generator):
        particle_encode = torch.nn.Sequential(
            SeedableLinear(4, 16, generator = generator, device=generator.device),
            torch.nn.ReLU(True),
            SeedableLinear(16, 32, generator = generator, device=generator.device),
            torch.nn.ReLU(True),
            SeedableLinear(32, 64, generator = generator, device=generator.device),
            torch.nn.ReLU(True),
            SeedableDropout(p=1 - dropout_keep_ratio, generator=generator),
            SeedableLinear(64, hidden_size, generator = generator, device=generator.device),
        )
        return particle_encode

class RealNVP_cond(pydpf.Module):

    def __init__(self, dim, hidden_dim=8, base_network=FCNN, condition_on_dim=None, generator = torch.default_generator, zero_i = False):
        super().__init__()
        self.dim = dim
        self.condition_on_dim = condition_on_dim
        self.t1 = base_network(dim // 2 + self.condition_on_dim, ceil(dim / 2), hidden_dim, generator)
        self.t2 = base_network(ceil(dim / 2) + self.condition_on_dim, dim // 2, hidden_dim, generator)
        self.generator = generator
        if zero_i:
            self.zero_initialization()

    def zero_initialization(self, std=0.01):
        for layer in self.t1.network:
            if layer.__class__.__name__ == 'SeedableLinear':
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                layer.bias.data.fill_(0)
        for layer in self.t2.network:
            if layer.__class__.__name__ == 'SeedableLinear':
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                layer.bias.data.fill_(0)

    def forward(self, x, condition_on):
        lower, upper = x[..., :self.dim // 2], x[..., self.dim // 2:]
        lower_extended = torch.cat([lower, condition_on], dim=-1)
        t1_transformed = self.t1(lower_extended)
        upper = t1_transformed + upper
        upper_extended = torch.cat([upper, condition_on], dim=-1)
        t2_transformed = self.t2(upper_extended)
        lower = t2_transformed + lower
        z = torch.cat([lower, upper], dim=-1)
        return z, 0

    def inverse(self, z, condition_on):
        lower, upper = z[..., :self.dim // 2], z[..., self.dim // 2:]

        upper_extended = torch.cat([upper, condition_on], dim=-1)
        t2_transformed = self.t2(upper_extended)
        lower = lower - t2_transformed
        lower_extended = torch.cat([lower, condition_on], dim=-1)
        t1_transformed = self.t1(lower_extended)
        upper = upper - t1_transformed
        x = torch.cat([lower, upper], dim=-1)
        log_det = 0
        return x, log_det


class NormalizingFlowModel_cond(pydpf.Module):

    def __init__(self, prior, flows, device='cuda:0'):
        super().__init__()
        self.prior = prior
        self.device = device
        self.flows = torch.nn.ModuleList(flows).to(self.device)

    def forward(self, x, condition_on):
        b, m, d = x.shape
        log_det = torch.zeros((b, m)).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x, condition_on)
            log_det += ld
        return x, log_det

    def inverse(self, z, condition_on):
        b, m, d = z.shape
        log_det = torch.zeros((b, m)).to(self.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, condition_on)
            log_det += ld
        x = z
        return x, log_det

    def log_density(self, x, condition_on):
        z, log_det = self.forward(x, condition_on)
        prior_prob = self.prior.log_density(z)

        return prior_prob + log_det

    def sample(self, sample_size, condition_on):
        z = self.prior.sample(sample_size, device = self.device)
        return self.inverse(z, condition_on)[0]

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
    def __init__(self, flow_model, encoder, decoder, state_encoder, device=torch.device('cpu')):
        super().__init__()
        self.flow_model = flow_model
        self.encoder = encoder
        self.scaling_tensor = torch.tensor([[[1., 1., torch.pi]]], device=device)
        self.decoder = decoder
        self.state_encoder = state_encoder

    def score(self, state, observation, t, **data):
        b, n , _ = state.shape
        c = torch.cos(state[:, :, 2:3])
        s = torch.sin(state[:, :, 2:3])
        encoded_state = self.state_encoder(torch.concat([state[:, :, :2], c, s], dim=-1))
        return self.flow_model.log_density(observation.unsqueeze(1).expand(-1, n, -1), encoded_state)

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


def train(dpf,
          opt: torch.optim.Optimizer,
          dataset: torch.utils.data.Dataset,
          epochs: int,
          n_particles: Tuple[int, int, int],
          batch_size: Tuple[int, int, int],
          split_size: Tuple[float, float, float],
          scalings:Tuple[float,float,float] = (1.,1.,1.),
          data_loading_generator: torch.Generator = torch.default_generator,
          gradient_regulariser = None,
          target:str = 'MSE',
          time_extent = None,
          lr_scheduler = None,
          pre_train_epochs=0,
          device = torch.device('cuda:0'),
          state_scaling = 1000.
          ):

    batch_size = list(batch_size)
    position_scaling = torch.tensor([[[state_scaling, state_scaling]]], device=device)

    aggregation_function = {'Mean Pose': pydpf.FilteringMean(lambda state: torch.concat([state[..., :2], torch.sin(state[..., 2:3]), torch.cos(state[..., 2:3])], dim=-1)), 'ELBO': pydpf.ElBO_Loss()}
    validation_aggregation_function = {'Mean Pose': pydpf.FilteringMean(lambda state: torch.concat([state[..., :2], bind_angle(state[..., 2:3])], dim=-1)), 'ELBO': pydpf.ElBO_Loss()}
    data_length = len(dataset)
    train_validation_test_split = _get_split_amounts(split_size, data_length)
    train_set, validation_set, test_set = torch.utils.data.random_split(dataset, train_validation_test_split, generator=data_loading_generator)
    if batch_size[0] == -1 or batch_size[0] > len(train_set):
        batch_size[0] = len(train_set)
    if batch_size[1] == -1 or batch_size[1] > len(validation_set):
        batch_size[1] = len(validation_set)
    if batch_size[2] == -1 or batch_size[2] > len(test_set):
        batch_size[2] = len(test_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size[0], shuffle=True, generator=data_loading_generator, collate_fn=dataset.collate)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size[1], shuffle=False, generator=data_loading_generator, collate_fn=dataset.collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size[2], shuffle=False, generator=data_loading_generator, collate_fn=dataset.collate)
    best_eval = torch.inf
    best_dict = None
    encoder = dpf.SSM.observation_model.encoder
    decoder = dpf.SSM.observation_model.decoder
    state_encoder = dpf.SSM.observation_model.state_encoder

    if time_extent is None:
        time_extent = dataset.observation.size(0)-1

    for epoch in range(pre_train_epochs):
        dpf.train()
        train_loss = 0
        total_size = 0
        for i, (state, observation, control) in enumerate(train_loader):
            dpf.update()
            opt.zero_grad()
            observation = observation.to(device).reshape(observation.size(0)*observation.size(1), 3, 24, 24)
            encoded_obs = encoder(observation)
            decoded_obs = decoder(encoded_obs)
            AE_loss = torch.mean((decoded_obs - observation)**2)*scalings[2]
            AE_loss.backward()
            opt.step()
            train_loss += AE_loss.item() * observation.size(1)
            total_size += observation.size(1)
        process_print(f'Auto_encoder training loss = {train_loss / total_size}')

    start_time = time()

    for epoch in range(epochs):

        train_loss = []
        total_size = 0
        dpf.train()
        for state, observation, control in train_loader:
            dpf.update()
            opt.zero_grad()
            batch_size_now = observation.size(1)
            #print(observation.size())
            #observation = einops.rearrange(observation, 't b (c h w) -> (t b) c h w', c=3, h=24, w=24)
            observation = observation.to(device).reshape(observation.size(0)*observation.size(1), 3, 24, 24)
            #print(observation.size())
            state = state.to(device)
            control = control.to(device)
            encoded_obs = encoder(observation)
            decoded_obs = decoder(encoded_obs)
            AE_loss = torch.mean((decoded_obs - observation) ** 2)
            outputs = dpf(n_particles[0], time_extent, aggregation_function, observation=encoded_obs.reshape(100, batch_size_now, encoded_obs.size(1)).contiguous(), ground_truth=state, control=control, gradient_regulariser = gradient_regulariser)
            #print(torch.sqrt(torch.mean(torch.sum(((outputs['Mean Pose'][-1,:,:2] - state[-1,:,:2])*position_scaling)**2, dim=-1))).item())

            cos_loss = torch.mean((outputs['Mean Pose'][:,:,3] - torch.cos(state[:, :, 2]))**2)
            sin_loss = torch.mean((outputs['Mean Pose'][:,:,2] - torch.sin(state[:, :, 2]))**2)
            angle_loss = cos_loss + sin_loss
            elbo_loss = outputs['ELBO'].mean()
            #print(torch.sum((outputs['Mean Pose'][:,:,:2] - state[:outputs['Mean Pose'].size(0),:,:2])**2, dim=-1))
            position_loss = torch.mean(torch.sum((outputs['Mean Pose'][:,:,:2] - state[:,:,:2])**2, dim=-1))
            #print(position_loss)
            loss = scalings[0]*position_loss + scalings[1]*angle_loss + AE_loss*scalings[2] #+ elbo_loss
            #print(outputs['Mean Pose'][:, 0])
            loss.backward()
            train_loss.append(loss.item()*state.size(1))
            opt.step()
            total_size += state.size(1)
        if lr_scheduler is not None:
            lr_scheduler.step()
        train_loss = np.sum(np.array(train_loss)) / total_size
        dpf.update()
        dpf.eval()
        with torch.inference_mode():
            total_size = 0
            validation_Pos_MSE = []
            validation_Angle_MSE = []
            for state, observation, control in validation_loader:
                batch_size_now = observation.size(1)
                observation = observation.to(device).reshape(observation.size(0)*observation.size(1), 3, 24, 24)

                state = state.to(device)
                control = control.to(device)
                encoded_obs = encoder(observation)
                outputs = dpf(n_particles[1], time_extent, validation_aggregation_function, observation=encoded_obs.reshape(100, batch_size_now, encoded_obs.size(1)).contiguous(), ground_truth=state, control=control)
                validation_Pos_MSE.append(torch.mean(torch.sum(((outputs['Mean Pose'][-1,:,:2] - state[-1,:,:2])*position_scaling)**2, dim=-1)).item()*state.size(1))
                validation_Angle_MSE.append(torch.mean(bind_angle(bind_angle( outputs['Mean Pose'][:,:,2]) - bind_angle(state[:,:,2]))**2).item()*state.size(1))
                #print(torch.sum(((outputs['Mean Pose'][:,:,:2] - state[:,:,:2])*position_scaling)**2, dim=-1))
                total_size += state.size(1)
            validation_Pos_MSE= np.sum((np.array(validation_Pos_MSE))) / total_size
            validation_Angle_MSE = np.sum((np.array(validation_Angle_MSE))) / total_size
            if validation_Pos_MSE < best_eval:
                best_eval = validation_Pos_MSE
                best_dict = deepcopy(dpf.state_dict())



        process_print(f'epoch {epoch + 1}/{epochs}, train loss: {train_loss}, validation position RMSE: {np.sqrt(validation_Pos_MSE)}, validation angle RMSE: {np.sqrt(validation_Angle_MSE)}')
    total_size = 0
    dpf.load_state_dict(best_dict)

    with torch.inference_mode():
        test_Pos_MSE = []
        test_angle_MSE = []


        for state, observation, control in test_loader:
            batch_size_now = observation.size(1)
            observation = observation.to(device).reshape(observation.size(0)*observation.size(1), 3, 24, 24)
            state = state.to(device)
            control = control.to(device)
            encoded_obs = encoder(observation)
            outputs = dpf(n_particles[1], time_extent, validation_aggregation_function, observation=encoded_obs.reshape(100, batch_size_now, encoded_obs.size(1)).contiguous(), ground_truth=state, control =control)
            test_Pos_MSE.append(torch.mean(torch.sum(((outputs['Mean Pose'][-1,:,:2] - state[-1,:,:2])*position_scaling)**2, dim=-1)).item() * state.size(1))
            test_angle_MSE.append(torch.mean(bind_angle(bind_angle( outputs['Mean Pose'][:,:,2]) - bind_angle(state[:outputs['Mean Pose'].size(0),:,2]))**2).item()*state.size(1))
            total_size += state.size(1)
    test_Pos_MSE = np.sum((np.array(test_Pos_MSE))) / total_size
    test_angle_MSE = np.sum((np.array(test_angle_MSE))) / total_size
    print('')
    process_print(f'test position RMSE: {np.sqrt(test_Pos_MSE)}, test angle RMSE: {np.sqrt(test_angle_MSE)}')
    process_print(f'Final time = {time() - start_time}')
    return test_Pos_MSE, test_angle_MSE

def dm_get_DPF(DPF_type, SSM, cuda_gen):
    if DPF_type == 'DPF':
        return pydpf.DPF(SSM=SSM, resampling_generator=cuda_gen)
    if DPF_type == 'Soft':
        return pydpf.SoftDPF(SSM=SSM, resampling_generator=cuda_gen)
    if DPF_type == 'Stop-Gradient':
        return pydpf.StopGradientDPF(SSM=SSM, resampling_generator=cuda_gen)
    if DPF_type == 'Marginal Stop-Gradient':
        return pydpf.MarginalStopGradientDPF(SSM=SSM, resampling_generator=cuda_gen)
    if DPF_type == 'Optimal Transport':
        return pydpf.OptimalTransportDPF(SSM=SSM, regularisation=1.)
    if DPF_type == 'Kernel':
        Gaussian_kernel = pydpf.StandardGaussian(3, cuda_gen, learn_mean=False, learn_cov=True)
        kernel_mixture = pydpf.KernelMixture(kernel=Gaussian_kernel, generator=cuda_gen)
        return pydpf.KernelDPF(SSM=SSM, kernel=kernel_mixture)
    raise ValueError('DPF_type should be one of the allowed options')

def normalise_obs(observation, **data):
    return (observation - torch.mean(observation))/torch.std(observation)

def transform_control(control, **data):
    output = control/torch.tensor([[[1000., 1000., 1.]]], device=control.device)
    return output


def flatten_gens(list_of_gens):
    return [item for gen in list_of_gens for item in gen]


def is_in_it(item, it):
    return any(id(item) == id(item_) for item_ in it)


def get_SSM(generator, device):
    """
    Build the model from components
    """
    observation_encoding_size = 128
    state_encoding_size = 64
    encoder = ObservationEncoder(observation_encoding_size, generator=generator, dropout_keep_ratio=0.3)
    decoder = ObservationDecoder(observation_encoding_size, generator=generator, dropout_keep_ratio=0.3)
    state_encoder = StateEncoder(state_encoding_size, generator=generator, dropout_keep_ratio=0.6)
    observation_partial_flows = [RealNVP_cond(dim=observation_encoding_size, hidden_dim=observation_encoding_size, condition_on_dim=state_encoding_size, generator=generator, zero_i=True),
                                 RealNVP_cond(dim=observation_encoding_size, hidden_dim=observation_encoding_size, condition_on_dim=state_encoding_size, generator=generator, zero_i=True)]
    flow_cov = torch.nn.Parameter(torch.eye(observation_encoding_size, device=device) * 1, requires_grad=False)
    observation_flow = NormalizingFlowModel_cond(pydpf.MultivariateGaussian(torch.zeros(observation_encoding_size, device=device), cholesky_covariance=flow_cov, diagonal_cov=True, generator=generator), observation_partial_flows,
                                                                    device)
    observation_model = MazeObservation(observation_flow, encoder, decoder, state_encoder, device=device)
    dynamic_cov = torch.diag(torch.tensor([30 / 1000, 30 / 1000, 0.1], device=device))
    dynamic_model = MazeDynamic(generator, dynamic_cov)
    prior_model = MazePrior(2, 1.3, generator)
    encoder_parameters = flatten_gens([encoder.parameters(), state_encoder.parameters(), decoder.parameters()])
    flow_parameters = flatten_gens([observation_flow.parameters(), prior_model.parameters()])
    SSM = pydpf.FilteringModel(dynamic_model=dynamic_model, prior_model=prior_model, observation_model=observation_model)
    process_print(f'observation encoder has {sum(p.numel() for p in encoder.parameters())} parameters')
    print(f'state encoder has {sum(p.numel() for p in state_encoder.parameters())} parameters')
    print(f'decoder has {sum(p.numel() for p in decoder.parameters())} parameters')
    print(f'observation flow has {sum(p.numel() for p in observation_flow.parameters())} parameters')
    return SSM, encoder_parameters, flow_parameters, [flow_cov]



def select_experiment(experiment):
    match experiment:
        case "example_usage":
            return ex_usage_setup, ex_usage_run_script
        case "comparison_to_Kalman":
            return comparison_to_Kalman_setup, comparison_to_Kalman_run_script
        case "learning_proposal_parameters":
            return proposal_learning_setup, proposal_learning_run_script
        case "fully_specified_model":
            return sv_set_up, full_specified_run_script
        case "single_parameter_learning":
            return SV_single_param_set_up, SV_single_param_run

if __name__ == "__main__":
    args = parse_args()
    for i, experiment in enumerate(args["experiments"]):
        set_up, run = select_experiment(experiment)
        print("=======================================================================")
        print(f"Running experiment {i+1} of {len(args['experiments'])}: {experiment}")
        print("=======================================================================")
        c_process = experiment
        process_print("Setting up")
        set_up(**args)
        process_print("Running")
        run(**args)
    print("Done")