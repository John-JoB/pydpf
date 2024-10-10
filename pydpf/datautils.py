import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Union, Callable, Tuple
import os
import numpy as np
import shutil
from math import ceil
from joblib import Parallel, delayed
from csv import QUOTE_NONNUMERIC



def series_to_tensor(series: pd.Series, device: Union[str, torch.device] = torch.device('cpu')):
    #This is really awkward, but I timed it, and it's not prohibitively slow if only done infrequently.
    output = np.vstack((series.apply(lambda entry: np.fromstring(entry[1:-1], dtype=np.float32, sep=','))).to_numpy())
    return torch.from_numpy(output).to(device=device)

def tensor_to_series(tensor: torch.Tensor):
    output = tensor.cpu().numpy()
    output = pd.Series(list(output))
    output = output.apply(lambda entry: np.array2string(entry, floatmode='unique', separator=','))
    return output

class StateSpaceDataset(Dataset):
    '''
        Dataset class for state-observation data.
        Latent state of the system stored in the state Tensor.
        All other data including non-discrete time (if applicable) should be treated as observations.

        Dimensions are Discrete Time - Batch - Data

        At the moment I only give functionality to load entire data set into RAM/VRAM.
        Might give the option to load lazily in the future, if required.
    '''

    def __init__(self, dir_path: str, device: Union[str, torch.device] = torch.device('cpu'), processes: int = -1) -> None:
        self.device = device

        def read_helper(file_: str):
            nonlocal dir_path
            file_data = pd.read_csv(os.path.join(dir_path, file_))
            state_ = series_to_tensor(file_data['state'], device=self.device)
            observation_ = series_to_tensor(file_data['observation'], device=self.device)
            return state_, observation_

        read_output = Parallel(n_jobs=processes)(delayed(read_helper)(file) for file in os.listdir(dir_path) if file.endswith('.csv'))
        read_output = list(zip(*read_output))
        state = torch.stack(read_output[0])
        observation = torch.stack(read_output[1])
        self.state = torch.einsum('ijk -> jik', state)
        self.observation = torch.einsum('ijk -> jik', observation)

    def __len__(self):
        return self.state.size(1)

    def __getitem__(self, idx):
        return self.state[:, idx], self.observation[:, idx]

    @staticmethod
    def collate(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        #By default, the batch is the first dimension.
        #Pass this function to collate_fn when defining a dataloader to make it the second.
        collated_batch = torch.utils.data.default_collate(batch)
        return torch.einsum('ijk -> jik', collated_batch[0]), torch.einsum('ijk -> jik', collated_batch[1])


def simulate_to_folder(dir_path: str,
                       prior: Callable[[Tuple[int]], torch.tensor],
                       Markov_kernel: Callable[[torch.tensor, int], torch.tensor],
                       observation_model: Callable[[torch.tensor, int], torch.tensor],
                       time_extent: int,
                       n_trajectories: int,
                       batch_size: int,
                       device: Union[str, torch.device] = torch.device('cpu'),
                       processes: int = -1):

    if os.path.exists(dir_path):
        print(f'Warning - folder already exists at {dir_path}, continuing could overwrite its data')
        response = input('Continue? (y/n) ')
        if response != 'Y' and response != 'y':
            print('Halting')
            return
    else:
        os.mkdir(dir_path)

    def write_helper(trajectory_index: int, absolute_index :int, dir_path_: str, state_: torch.Tensor, observation_: torch.Tensor) -> None:
        traj_state = tensor_to_series(state_[:, trajectory_index])
        traj_observation = tensor_to_series(observation_[:, trajectory_index])
        df = pd.concat([traj_state.rename('state'), traj_observation.rename('observation')], axis=1)
        df.to_csv(os.path.join(dir_path_, f'{trajectory_index + absolute_index}.csv'), index=True, quoting=QUOTE_NONNUMERIC, mode='w')
        return None




    n_batches = ceil(n_trajectories / batch_size)

    with torch.inference_mode():
        for batch in range(n_batches):
            print(f'Generating batch {batch + 1}/{n_batches}', end = '\r')
            if batch == (n_trajectories // batch_size):
                temp = prior((n_trajectories - batch*batch_size,))
            else:
                temp = prior((batch_size,))
            state = torch.empty(size=(time_extent+1, temp.size(0), temp.size(1)), dtype=torch.float32, device=device)
            state[0] = temp
            temp = observation_model(state[0], 0)
            observation = torch.empty(size=(time_extent+1, temp.size(0), temp.size(1)), device=device)
            observation[0] = temp
            for t in range(time_extent):
                state[t+1] = Markov_kernel(state[t], t+1)
                observation[t+1] = observation_model(state[t+1], t+1)
            print(f'Saving batch     {batch + 1}/{n_batches}', end='\r')
            Parallel(n_jobs=processes)(delayed(lambda traj_index_: write_helper(traj_index_, batch_size * batch, dir_path, state, observation))(traj_index) for traj_index in range(observation.size(1)))
    print('Done                  \n')


