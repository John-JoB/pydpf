import pydpf
import torch
from typing import Tuple, Union
import numpy as np
from copy import deepcopy

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


def train(dpf,
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
          lr_scheduler = None
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
    best_dict = None
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
        if target=='MSE':
            if validation_MSE < best_eval:
                best_eval = validation_MSE
                best_dict = deepcopy(dpf.state_dict())
        else:
            if validation_ELBO < best_eval:
                best_eval = validation_ELBO
                best_dict = deepcopy(dpf.state_dict())


        print(f'epoch {epoch + 1}/{epochs}, train loss: {train_loss}, validation MSE: {validation_MSE}, validation ELBO: {validation_ELBO}')
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
    print(f'test MSE: {test_MSE}, test ELBO: {test_ELBO}')