import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Union, Callable, Tuple
import os
import numpy as np
from torch import Tensor
from math import ceil
from joblib import Parallel, delayed
from .deserialisation import load_data_csv
from pathlib import Path
from itertools import chain



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
    def __init__(self,
                 data_path: Union[Path,str],
                 *,
                 series_id_column="series_id",
                 state_prefix=None,
                 observation_prefix="observation_",
                 time_column=None,
                 control_prefix=None,
                 device = torch.device('cpu')
             ):
        self.device = device
        self.data = load_data_csv(data_path, series_id_column = series_id_column, state_prefix = state_prefix, observation_prefix = observation_prefix, time_column = time_column, control_prefix = control_prefix)
        self.data['tensor'] = torch.from_numpy(self.data['tensor']).to(device=self.device, dtype=torch.float32)
        self.data_order = []
        self.observation = self.data['tensor'][:, :, self.data['indices']['observation']]
        if state_prefix is not None:
            self.state = self.data['tensor'][:, :, self.data['indices']['state']]
            self.data_order.append('state')
        self.data_order.append('observation')
        if time_column is not None:
            self.time = self.data['tensor'][:, :, self.data['indices']['time']].squeeze()
            self.data_order.append('time')
        if control_prefix is not None:
            self.control = self.data['tensor'][:, :, self.data['indices']['control']]
            self.data_order.append('control')
        self.metadata_exists = False
        try:
            self.series_metadata = torch.from_numpy(self.data['metadata']).to(device=self.device, dtype=torch.float32)
            self.metadata_exists = True
        except KeyError:
            pass



    def __len__(self):
        return self.data['tensor'].size(0)

    def __getitem__(self, idx):
        if self.metadata_exists:
            return self.data['tensor'][idx], self.data['metadata'][idx]
        return self.data['tensor'][idx]

    def collate(self, batch) -> Tuple[torch.Tensor, ...]:
        #By default, the batch is the first dimension.
        #Pass this function to collate_fn when defining a dataloader to make it the second.
        #collated_batch = torch.utils.data.default_collate(batch)
        if self.metadata_exists:
            batch = tuple(zip(*batch))
            collated_data = torch.stack(batch[0], dim=0).transpose(0, 1)
            collated_metadata = torch.stack(batch[1], dim=0)
            return *(collated_data[:, :, self.data['indices'][data_category]].contiguous() for data_category in self.data_order), collated_metadata
        else:
            collated_batch = torch.stack(batch, dim=0).transpose(0, 1)
            return *(collated_batch[:, :, self.data['indices'][data_category]].squeeze().contiguous() for data_category in self.data_order),

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
                self.state.data = data.transpose(0, -1)
            else:
                self.observation.data = data.transpose(0, -1)
            return means.transpose(0, -1), std.transpose(0, -1)

def _get_time_data(data: dict, t: int) -> dict:
    time_dict = {k:v[t] for k, v in data.items() if k != 'series_metadata'}
    try:
        time_dict['series_metadata'] = data['series_metadata']
    except KeyError:
        pass
    return time_dict


def _format_to_save(state, observation, control, time):
    data_list = [state.cpu().numpy(), observation.cpu().numpy()]
    columns_list = [[f'state_{i + 1}' for i in range(state.size(-1))], [f'observation_{i + 1}' for i in range(observation.size(-1))]]
    if control is not None:
        data_list.append(control.cpu().numpy())
        columns_list.append([f'control_{i + 1}' for i in range(state.size(-1))])
    if time is not None:
        data_list.append(time.unsqueeze(-1).cpu().numpy())
        columns_list.append(['time'])
    return np.concatenate(data_list, axis=-1), chain.from_iterable(columns_list)

def _save_directory_csv(path:Path, start_index, state, observation, control, time, n_processes = -1):

    data, columns_list = _format_to_save(state, observation, control, time)
    def write_help(series_index):
        df = pd.DataFrame(data[series_index - start_index])
        df.columns = columns_list
        df.to_csv(path / f'trajectory_{series_index + 1}.csv' ,  index=False)
    Parallel(n_jobs=n_processes)(delayed(write_help)(series_index)
                                 for series_index in range(start_index, start_index + state.size(0))
                                 )

def _save_file_csv(path:Path, state, observation, control, time, n_processes = -1):

    data, columns_list = _format_to_save(state, observation, control, time)
    def make_traj_frame(series_index):
        df = pd.DataFrame(data[series_index])
        df.columns = columns_list
        df['series_index'] = series_index + 1
        return df
    df_list = list(Parallel(n_jobs=n_processes)(delayed(make_traj_frame)(series_index)
                                                for series_index in range(len(data))
                                                ))
    total_df = pd.concat(df_list, axis=0)
    total_df.to_csv(path, index=False)


def simulate_and_save(data_path: Union[Path, str],
                       prior: Callable,
                       Markov_kernel: Callable,
                       observation_model: Callable,
                       time_extent: int,
                       n_trajectories: int,
                       batch_size: int,
                       device: Union[str, torch.device] = torch.device('cpu'),
                       control: Tensor = None,
                       time:Tensor = None,
                       n_processes = -1):
    if isinstance(data_path, str):
        data_path = Path(data_path)
    if data_path.suffix == '.csv':
        state_list = []
        observation_list = []
        if data_path.is_file():
            print(f'Warning - folder already exists at {data_path}, continuing could overwrite its data')
            response = input('Continue? (y/n) ')
            if response != 'Y' and response != 'y':
                print('Halting')
                return
            os.remove(data_path)
    else:
        if data_path.is_dir():
            print(f'Warning - folder already exists at {data_path}, continuing could overwrite its data')
            response = input('Continue? (y/n) ')
            if response != 'Y' and response != 'y':
                print('Halting')
                return
        else:
            os.mkdir(data_path)

    data_dict = {}

    n_batches = ceil(n_trajectories / batch_size)

    with torch.inference_mode():
        for batch in range(n_batches):
            print(f'Generating batch {batch + 1}/{n_batches}', end = '\r')
            if batch == (n_trajectories // batch_size):
                if control is not None:
                    batch_control = control[batch * batch_size:]
                    data_dict['control'] = batch_control[:, 0]
                if time is not None:
                    batch_time = time[batch * batch_size:]
                    data_dict['time'] = batch_time[: 0]
                temp = prior((n_trajectories - batch*batch_size,), **data_dict)
            else:
                if control is not None:
                    batch_control = control[batch * batch_size : (batch + 1) * batch_size]
                    data_dict['control'] = batch_control[:, 0]
                if time is not None:
                    batch_time = time[batch * batch_size : (batch + 1) * batch_size]
                    data_dict['time'] = batch_time[: 0]
                temp = prior((batch_size,), **data_dict)
            state = torch.empty(size=(temp.size(0), time_extent+1, temp.size(1)), dtype=torch.float32, device=device)
            state[:, 0] = temp
            temp = observation_model(state[:, 0], **data_dict)
            observation = torch.empty(size=(temp.size(0), time_extent+1, temp.size(1)), device=device)
            observation[:, 0] = temp
            for t in range(time_extent):
                if control is not None:
                    data_dict['control'] = batch_control[:, t]
                if time is not None:
                    data_dict['time'] = batch_time[:, t]
                state[:, t+1] = Markov_kernel(state[:, t], **data_dict)
                observation[:, t+1] = observation_model(state[:, t+1], **data_dict)
            if data_path.suffix == '.csv':
                state_list.append(state)
                observation_list.append(observation)
            else:
                _save_directory_csv(data_path, batch_size*batch, state, observation, control, time, n_processes)
        if data_path.suffix == '.csv':
            state = torch.cat(state_list, dim=0)
            observation = torch.cat(observation_list, dim=0)
            _save_file_csv(data_path, state, observation, control, time)
    print('Done                  \n')








