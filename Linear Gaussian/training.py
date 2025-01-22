import torch
from torchviz import make_dot
import pydpf
from pathlib import Path
from typing import Union, Tuple
import numpy as np
from copy import deepcopy
import gc

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

def test(pf,
          data: pydpf.StateSpaceDataset,
          n_particles: int,
          batch_size: int,
          loss_function: pydpf.Module,
          data_loading_generator: torch.Generator = torch.default_generator
          ):
    print('   ')
    with torch.inference_mode():

        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, generator=data_loading_generator, collate_fn=data.collate)
        losses = []
        total_size = 0

        for state, observation in loader:
            pf.update()
            loss = pf(n_particles, observation.size(0) - 1, loss_function, observation=observation, ground_truth=state, gradient_regulariser = None)
            loss = loss.mean()
            losses.append(loss.item()*state.size(1))
            total_size += state.size(1)
        train_loss = np.sum(np.array(losses))/total_size
        print(f"loss = {train_loss}")



def train(dpf,
          opt: torch.optim.Optimizer,
          data_set: Union[Path, str],
          epochs: int,
          n_particles: Tuple[int, int, int],
          batch_size: Tuple[int, int, int],
          split_size: Tuple[float, float, float],
          loss_function: pydpf.Module,
          metric: pydpf.Module = None,
          data_loading_generator: torch.Generator = torch.default_generator,
          gradient_regulariser = None
          ):

    batch_size = list(batch_size)
    if metric is None:
        metric = loss_function

    data = data_set
    data_length = len(data)
    train_validation_test_split = _get_split_amounts(split_size, data_length)
    train_set, validation_set, test_set = torch.utils.data.random_split(data, train_validation_test_split, generator=data_loading_generator)
    if batch_size[0] == -1 or batch_size[0] > len(train_set):
        batch_size[0] = len(train_set)
    if batch_size[1] == -1 or batch_size[1] > len(validation_set):
        batch_size[1] = len(validation_set)
    if batch_size[2] == -1 or batch_size[2] > len(test_set):
        batch_size[2] = len(test_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size[0], shuffle=True, generator=data_loading_generator, collate_fn=data.collate)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size[1], shuffle=False, generator=data_loading_generator, collate_fn=data.collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size[2], shuffle=False, generator=data_loading_generator, collate_fn=data.collate)
    best_eval = torch.inf
    best_dict = None
    for epoch in range(epochs):
        train_loss = []
        total_size = 0
        dpf.train()
        for state, observation in train_loader:
            dpf.update()
            opt.zero_grad()
            loss = dpf(n_particles[0], observation.size(0) - 1, loss_function, observation=observation, ground_truth=state, gradient_regulariser = gradient_regulariser)
            loss = loss.mean()
            loss.backward()
            train_loss.append(loss.item()*state.size(1))
            opt.step()
            total_size += state.size(1)
        train_loss = np.sum(np.array(train_loss)) / total_size

        dpf.update()
        dpf.eval()
        with torch.inference_mode():
            total_size = 0
            validation_loss = []
            for state, observation in validation_loader:
                loss = dpf(n_particles[1], observation.size(0) - 1, metric, observation=observation, ground_truth=state)
                loss = loss.mean()
                validation_loss.append(loss.item()*state.size(1))
                total_size += state.size(1)
            validation_loss = np.sum((np.array(validation_loss)))
        validation_loss = np.sum(np.array(validation_loss)) / total_size
        if validation_loss < best_eval:
            best_eval = validation_loss
            best_dict = deepcopy(dpf.state_dict())

        print(f'epoch {epoch + 1}/{epochs}, train loss: {train_loss}, validation loss: {validation_loss}')
    total_size = 0
    with torch.inference_mode():
        test_loss = []
        dpf.load_state_dict(best_dict)
        for state, observation in test_loader:
            loss = dpf(n_particles[1], observation.size(0) - 1, metric, observation=observation, ground_truth=state)
            loss = loss.mean()
            test_loss.append(loss.item()*state.size(1))
            total_size += state.size(1)
    test_loss = np.sum(np.array(test_loss)) / total_size
    print('')
    print(f'Test loss: {test_loss}')