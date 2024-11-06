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

    def normalise_dims(self, normalise_state: bool, scale_dims: str = 'all', individual_timesteps: bool = True, dims: Union[Tuple[int], None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalise the data to have mean zero and standard deviation one.

        This function normalises the data inplace and returns the offset and scale. Such that the original data can be reclaimed by original_data = normalised_data * scale + offset.

        This function can be applied to either the state or observations, this is controlled by the parameter normalise_state.

        There are various methods to control the scaling, determined by the value of scale_dims:
            - 'all': scale each dimension independently, such that every dimension have standard deviation 1.
            - 'max': scale each dimension by the same factor, such that the maximum of the standard deviations is 1.
            - 'min': scale each dimension by the same factor, such that the minimum of the standard deviations is 1.
            - 'norm': scale each dimension by the same factor, such that the standard deviation of the vector norm of the data is 1.

        The parameter individual_timesteps controls whether to apply the same normalisation across time-steps, or to calculate a separate mean and standard deviation per time-step.

        The normalisation doesn't have to be across all data dimensions, one can specify a tuple of dimensions to include to the parameter dims. Or set dims=None to use all dimensions.

        Parameters
        ----------
        normalise_state: bool
            When True, normalise the state. When False, normalise the observations.
        scale_dims: str
            The method to scale over dimensions. See above for options and details.
        individual_timesteps: bool, default=True
            When true, the scaling and offset is calculated per-time-step, when false the scaling and offset are set to be the same for each time-step (in most cases this should be True).
        dims: Tuple[int] or None, default=None
            The dimensions to normalise.

        Returns
        -------
        offset: torch.Tensor
            The per-element offset.
        scaling: torch.Tensor
            The per-element scaling.

        """
        with torch.no_grad():
            if not scale_dims in ['all', 'max', 'min', 'norm']:
                raise ValueError('scale_dims must be one of "all", "max", "min" or "norm"')
            if normalise_state:
                data = self.state
            else:
                data = self.observation
            data_size = data.size(-1)
            data = data.transpose(0, -1)
            if dims is None:
                mask = [True for _ in range(data_size)]
            else:
                mask = [False for _ in range(data_size)]
                for d in dims:
                    if d < 0:
                        d = data_size + d
                    if d >= data_size or d < 0:
                        raise IndexError('Dimension out of bounds')
                    mask[d] = True

            if individual_timesteps:
                reduction_dims = (2,)
            else:
                reduction_dims = (1, 2)

            masked_data = data[mask]

            means = torch.mean(masked_data, dim=reduction_dims, keepdim=True)
            if scale_dims == 'all':
                std = torch.std(masked_data, dim=reduction_dims, keepdim=True)
            if scale_dims == 'max':
                std = torch.amax(torch.std(masked_data, dim=reduction_dims, keepdim=True), dim=0, keepdim=True)
            if scale_dims == 'min':
                std = torch.amin(torch.std(masked_data, dim=reduction_dims, keepdim=True), dim=0, keepdim=True)
            if scale_dims == 'norm':
                std = torch.std(torch.linalg.vector_norm(masked_data, dim=0, keepdim=True), dim=reduction_dims, keepdim=True)

            means = means.expand(masked_data.size())
            std = std.expand(masked_data.size())
            data[mask] = (masked_data - means) / std
            if normalise_state:
                self.state = data.transpose(0, -1)
            else:
                self.observation = data.transpose(0, -1)
            return means.transpose(0, -1), std.transpose(0, -1)

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








