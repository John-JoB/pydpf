import argparse
import torch
from sv_model import make_SSM
import pathlib
import pydpf
import pandas as pd

def make_new_csv(rows, columns, dir, name):
    file_path = dir / f"{name}.csv"
    if file_path.exists():
        print(f"File already exists at {file_path}, skipping creation")
        return
    df = pd.DataFrame(index=pd.Index(rows, name="method"), columns=columns)
    df.to_csv(file_path)

if __name__ == "__main__":
    data_folder = pathlib.Path('.').parent.absolute().joinpath(f'data/')
    if not data_folder.exists():
        data_folder.mkdir()
    results_folder = pathlib.Path('.').parent.absolute().joinpath(f'results/')
    if not results_folder.exists():
        results_folder.mkdir()
    rows = ["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"]
    make_new_csv(rows, ["e_x", "e_l", "time"], results_folder, "fully_specified_results")
    make_new_csv(rows, ["Forward Time (s)", "Backward Time (s)", "Gradient standard deviation", "alpha error"], results_folder, "single_parameter_results")
    make_new_csv(rows, ["ELBO", "alpha error", "beta error","sigma error"], results_folder, "multiple_parameter_results")
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.91, help="Generation alpha")
    parser.add_argument("--beta", type=float, default=0.5, help="Generation beta")
    parser.add_argument("--sigma",type=float, default=1., help="Generation sigma")
    parser.add_argument("--batch_size",type=int, default=128, help="Generation sigma")
    args = parser.parse_args()
    data_path = data_folder / f"alpha={args.alpha}-beta={args.beta}-sigma={args.sigma}.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_gen_generator = torch.Generator(device=device).manual_seed(0)
    alpha = torch.tensor([[args.alpha]], device=device)
    beta = torch.tensor([args.beta], device=device)
    sigma = torch.tensor([[args.sigma]], device=device)
    SSM = make_SSM(sigma, alpha, beta, device, generator=data_gen_generator)
    pydpf.simulate_and_save(data_path, SSM=SSM, time_extent=1000, n_trajectories=500, batch_size=args.batch_size, device=device)