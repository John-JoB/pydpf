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
from .model_based_api import FilteringModel
from pathlib import Path
from itertools import chain

class StateSpaceDataset(Dataset):
    '''
        Dataset class for state-observation data.
        Latent state of the system stored in the state Tensor.

        Dimensions are Discrete Time - Batch - Data

        When used as called from a dataloader you must use the custom collate function
        Data will always be returned in the order 'state' - 'observation' - 'time' - 'control' - 'metadata'

        At the moment I only give functionality to load entire data set into RAM/VRAM.
        Might give the option to load lazily in the future, if required.
    '''

    @property
    def state(self):
        if 'state' in self.data_order:
            return self.data['tensor'][:, :, self.data['indices']['state']].permute(1, 0, 2).contiguous()
        raise AttributeError('No state data available')

    @property
    def observation(self):
        if 'observation' in self.data_order:
            return self.data['tensor'][:, :, self.data['indices']['observation']].permute(1, 0, 2).contiguous()
        raise AttributeError('No state data available')

    @property
    def time(self):
        if 'time' in self.data_order:
            return self.data['tensor'][:, :, self.data['indices']['time']].squeeze(-1).permute(1, 0).contiguous()
        raise AttributeError('No time data available')

    @property
    def control(self):
        if 'control' in self.data_order:
            return self.data['tensor'][:, :, self.data['indices']['control']].permute(1, 0, 2).contiguous()
        raise AttributeError('No control data available')

    @property
    def metadata(self):
        if self.metadata:
            return self.data['metadata']
        raise AttributeError('No metadata data available')

    def __init__(self,
                 data_path: Union[Path,str],
                 *,
                 series_id_column="series_id",
                 state_prefix=None,
                 observation_prefix="observation",
                 time_column=None,
                 control_prefix=None,
                 device = torch.device('cpu')
             ):
        self.device = device
        self.data = load_data_csv(data_path, series_id_column = series_id_column, state_prefix = state_prefix, observation_prefix = observation_prefix, time_column = time_column, control_prefix = control_prefix)
        self.data['tensor'] = torch.from_numpy(self.data['tensor']).to(device=self.device, dtype=torch.float32)
        self.data_order = []
        if state_prefix is not None:
            self.data_order.append('state')
        self.data_order.append('observation')
        if time_column is not None:
            self.data_order.append('time')
        if control_prefix is not None:
            self.data_order.append('control')
        self.metadata_exists = False
        try:
            self.series_metadata = torch.from_numpy(self.data['metadata']).to(device=self.device, dtype=torch.float32)
            self.metadata_exists = True
        except KeyError:
            pass

    @property
    def observation_dimension(self):
        return self.observation.shape[-1]

    @property
    def state_dimension(self):
        return self.state.shape[-1]

    @property
    def control_dimension(self):
        return self.control.shape[-1]

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
            return (*(collated_data[:, :, self.data['indices'][data_category]].squeeze(-1).contiguous() if data_category == "time" else collated_data[:, :, self.data['indices'][data_category]].contiguous() for data_category in self.data_order),
                    collated_metadata)
        else:
            collated_batch = torch.stack(batch, dim=0).transpose(0, 1)
            return *(collated_batch[:, :, self.data['indices'][data_category]].squeeze(-1).contiguous() if data_category == "time" else collated_batch[:, :, self.data['indices'][data_category]].contiguous() for data_category in self.data_order),

    def normalise_dims(self, normalised_series:str = 'observation', scale_dims: str = 'all', individual_timesteps: bool = False, dims: Union[Tuple[int], None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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

            data = self.data['tensor'][:, :, self.data['indices'][normalised_series]].clone()
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
            self.apply(lambda **data_dict : data.transpose(0,1), modified_series=normalised_series)
            return means.transpose(0, -1).transpose(0,1).contiguous(), std.transpose(0, -1).transpose(0,1).contiguous()

    def apply(self, f, modified_series:str = 'observation'):
        """
        Apply a function across all trajectories

        Returns
        -------

        """
        with torch.no_grad():
            true_order = ['state', 'observation', 'time', 'control', 'metadata']
            if not modified_series in true_order:
                raise ValueError('modified_series must be one of "state", "observation", "control", "time", or "metadata"')

            partitioned_data = {data_category: self.data['tensor'][:, :, self.data['indices'][data_category]].transpose(0,1).contiguous() for data_category in self.data_order}
            if self.metadata_exists:
                partitioned_data['metadata'] = self.data['metadata']
            new_series = f(**partitioned_data).transpose(0,1)
            if modified_series in self.data_order:
                inverse_index = [i for i in range(self.data['tensor'].size(-1)) if (i not in self.data['indices'][modified_series])]
                new_data = self.data['tensor'][:, :, inverse_index]
                if new_series.dim() == 2:
                    new_series = new_series.unsqueeze(-1)
                start_index = new_data.size(-1)
                self.data['tensor'] = torch.cat((new_data, new_series), dim=-1)
                for series in self.data_order:
                    if self.data['indices'][series][0] > self.data['indices'][modified_series][0]:
                        self.data['indices'][series] = range(self.data['indices'][series][0]-new_series.size(-1), self.data['indices'][series][-1] + 1 - new_series.size(-1))
                self.data['indices'][modified_series] = range(start_index, self.data['tensor'].size(-1))
                return
            self.data_order = [series for series in true_order if (series in self.data_order or series == modified_series)]
            start_index = self.data['tensor'].size(-1)
            self.data['tensor'] = torch.cat((self.data['tensor'], new_series), dim=-1)
            self.data['indices'][modified_series] = range(start_index, self.data['tensor'].size(-1))






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
    return np.concatenate(data_list, axis=-1), list(chain.from_iterable(columns_list))

def _save_directory_csv(path:Path, start_index, state, observation, control, time, n_processes = -1):

    data, columns_list = _format_to_save(state, observation, control, time)
    def write_help(series_id):
        df = pd.DataFrame(data[series_id - start_index])
        df.columns = columns_list
        df.to_csv(path / f'trajectory_{series_id + 1}.csv' ,  index=False)
    Parallel(n_jobs=n_processes)(delayed(write_help)(series_id)
                                 for series_id in range(start_index, start_index + state.size(0))
                                 )

def _save_file_csv(path:Path, state, observation, control, time, n_processes = -1):
    data, columns_list = _format_to_save(state, observation, control, time)
    def make_traj_frame(series_id):
        df = pd.DataFrame(data[series_id])
        df.columns = columns_list

        df['series_id'] = series_id + 1
        return df
    #df_list = list(Parallel(n_jobs=n_processes)(delayed(make_traj_frame)(series_id)
                                               # for series_id in range(len(data))
                                                #))
    df_list = [make_traj_frame(series_id) for series_id in range(len(data))]
    total_df = pd.concat(df_list, axis=0)
    total_df.to_csv(path, index=False)


def simulate_and_save(data_path: Union[Path, str],
                    *,
                    SSM: FilteringModel = None,
                    prior: Callable = None,
                    Markov_kernel: Callable = None,
                    observation_model: Callable = None,
                    time_extent: int,
                    n_trajectories: int,
                    batch_size: int,
                    device: Union[str, torch.device] = torch.device('cpu'),
                    control: Tensor = None,
                    time:Tensor = None,
                    n_processes = -1,
                    by_pass_ask = False):

    if SSM is not None:
        prior = lambda _batch_size, **_data_dict:  torch.squeeze(SSM.prior_model.sample(_batch_size, 1, **_data_dict), 1)
        observation_model = lambda _state, **_data_dict: SSM.observation_model.sample(state=_state, **_data_dict)
        Markov_kernel = lambda _prev_state, **_data_dict: SSM.dynamic_model.sample(prev_state=_prev_state, **_data_dict)
    if isinstance(data_path, str):
        data_path = Path(data_path)
    if data_path.suffix == '.csv':
        state_list = []
        observation_list = []
        if data_path.is_file():
            if not by_pass_ask:
                print(f'Warning - file already exists at {data_path}, continuing could overwrite its data')
                response = input('Continue? (y/n) ')
                if response != 'Y' and response != 'y':
                    print('Halting')
                    return
            os.remove(data_path)
    else:
        if data_path.is_dir() and not by_pass_ask:
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
                print()
                temp = prior(n_trajectories - batch*batch_size, **data_dict)
            else:
                if control is not None:
                    batch_control = control[batch * batch_size : (batch + 1) * batch_size]
                    data_dict['control'] = batch_control[:, 0]
                if time is not None:
                    batch_time = time[batch * batch_size : (batch + 1) * batch_size]
                    data_dict['time'] = batch_time[: 0]
                temp = prior(batch_size, **data_dict)
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








