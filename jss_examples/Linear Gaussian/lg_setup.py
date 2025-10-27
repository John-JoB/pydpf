import torch
import pathlib
import pydpf
import argparse
import lg_model
import pandas as pd

def make_new_csv(rows, columns, dir, name):
    file_path = dir / f"{name}.csv"
    if file_path.exists():
        print(f"File already exists at {file_path}, skipping creation")
        return
    df = pd.DataFrame(index=pd.Index(rows, name="method"), columns=columns)
    df.to_csv(file_path)

def make_model_componets(dx, dy, generator):
    dynamic_model = lg_model.GaussianDynamic(dx, generator)
    observation_model = lg_model.GaussianObservation(dx, dy, generator)
    prior_model = lg_model.GaussianPrior(dx, generator)
    return prior_model, dynamic_model, observation_model

if __name__ == "__main__":
    data_folder = pathlib.Path('.').parent.absolute().joinpath(f'data/')
    if not data_folder.exists():
        data_folder.mkdir()
    results_folder = pathlib.Path('.').parent.absolute().joinpath(f'results/')
    if not results_folder.exists():
        results_folder.mkdir()
    make_new_csv(["Kalman Filter", "PF K = 25", "PF K = 100", "PF K = 1000", "PF K = 10000"],
                 ["Time CPU (s)", "Time GPU (s)", "epsilon x", "epsilon y"],
                 results_folder,
                 "Kalman_comparison_results")
    make_new_csv(["Bootstrap", "Optimal", "DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"],
                 ["e_x", "e_l", "max W2", "ELBO"],
                 results_folder,
                 "proposal_learning_results")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", action="store", type=int, default = 25)
    parser.add_argument("--dy", action="store", type=int, default=1)
    parser.add_argument("--batch_size", action="store", type=int, default=128, help="The size of the batches in which data is generated in parallel")
    args = parser.parse_args()
    data_path = pathlib.Path('.').parent.absolute().joinpath(f'data/dx={args.dx}-dy={args.dy}.csv')
    gen_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen_generator = torch.Generator(device=gen_device).manual_seed(0)
    prior_model, dynamic_model, observation_model = make_model_componets(args.dx, args.dy, gen_generator)
    SSM = pydpf.FilteringModel(prior_model=prior_model, dynamic_model=dynamic_model, observation_model=observation_model)
    pydpf.simulate_and_save(data_path, SSM=SSM, time_extent=1000, n_trajectories=2000, batch_size=100, device=gen_device)