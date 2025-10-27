import torch
import numpy as np
import pathlib
import pandas as pd
import requests
from tqdm import tqdm
import zipfile
import argparse
import shutil


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




def make_new_csv(rows, columns, dir, name):
    file_path = dir / f"{name}.csv"
    if file_path.exists():
        print(f"File already exists at {file_path}, skipping creation")
        return
    df = pd.DataFrame(index=pd.Index(rows, name="method"), columns=columns)
    df.to_csv(file_path)

def bind_angle(angle):
    out = torch.where(angle > torch.pi, angle - 2*torch.pi, angle)
    return torch.where(out < -torch.pi, out + 2*torch.pi, out)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete_raw", action="store_true", help='Delete raw data when done')
    args = parser.parse_args()
    results_folder = pathlib.Path('.').parent.absolute().joinpath(f'results/')
    if not results_folder.exists():
        results_folder.mkdir()
    make_new_csv(["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"],
                 ["Total time (hrs:min:s)", "Test MSE"],
                 results_folder,
                 "deep_mind_maze_results")
    make_new_csv(["DPF", "Soft", "Stop-Gradient", "Marginal Stop-Gradient", "Optimal Transport", "Kernel"],
                 ["Total time (hrs:min:s)", "Test MSE"],
                 results_folder,
                 "nondeterministic_deep_mind_maze_results")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_folder_path = pathlib.Path('.').parent.absolute().joinpath('data/')
    data_folder_path.mkdir(exist_ok=True)
    data_raw_path_1 = data_folder_path / 'maze_data_raw.npz'
    data_raw_path_2 = data_folder_path / 'maze_data_raw_2.npz'
    if not (data_raw_path_1.exists() and data_raw_path_2.exists()):
        download_dataset(data_folder_path)
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
    total_df.to_csv(data_folder_path.joinpath('maze_data.csv'), index=False)
    if args.delete_raw:
        data_raw_path_1.unlink()
        data_raw_path_2.unlink()