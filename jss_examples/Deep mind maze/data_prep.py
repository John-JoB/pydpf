import torch
import numpy as np
import pathlib
import pandas as pd


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


def create_observations(obs, generator):
    new_o = torch.zeros([obs.shape[0], 24, 24, 3],  device=obs.device, dtype=torch.uint8)
    for i in range(obs.shape[0]):
            offsets = np.random.random_integers(0, 8, 2)
            new_o[i] = obs[i, offsets[0]:offsets[0] + 24, offsets[1]:offsets[1] + 24, :3]
    new_o = new_o.to(dtype=torch.float16)
    random = torch.normal(0.0, 20.0, new_o.shape, dtype=torch.float16, generator=generator, device=obs.device)
    new_o = torch.round(torch.clip(new_o + random, 0, 255)).to(dtype=torch.uint8)
    new_o = new_o.permute(0, 3, 1, 2)
    return new_o.flatten(start_dim=1)


def create_df(data, label):
    data = data.cpu().numpy()
    df = pd.DataFrame(data, columns=[f'{label}_{i+1}' for i in range(data.shape[1])])
    series_id = np.arange(2000).repeat(100)
    df['series_id'] = series_id
    return df


def prepare_data(device):
    data_folder_path = pathlib.Path('.').parent.absolute().joinpath('data/')
    data1 = dict(np.load(data_folder_path.joinpath('maze_data_raw.npz'), allow_pickle=True))
    data2 = dict(np.load(data_folder_path.joinpath('maze_data_raw_2.npz'), allow_pickle=True))
    state = torch.tensor(np.concatenate((data1['pose'], data2['pose']), axis=0), device=device, dtype=torch.float32)
    observation = torch.tensor(np.concatenate((data1['rgbd'], data2['rgbd']), axis=0), device=device, dtype=torch.uint8)
    actions = create_actions_and_modify_state(state)
    control_df = create_df(actions, 'control')
    observation = create_observations(observation, torch.Generator(device=device).manual_seed(0))
    observation_df = create_df(observation, 'observation')
    state_df = create_df(state, 'state')
    observation_df.drop(columns=['series_id'], inplace=True)
    control_df.drop(columns=['series_id'], inplace=True)
    total_df = pd.merge(control_df, state_df, left_index=True, right_index=True)
    total_df = pd.merge(total_df, observation_df, left_index=True, right_index=True)
    total_df.to_csv(data_folder_path.joinpath('maze_data.csv'), index=False)




