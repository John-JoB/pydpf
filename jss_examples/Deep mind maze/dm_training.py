
import torch
import pydpf
import numpy as np
from typing import Tuple
from copy import deepcopy
import time

def bind_angle(angle):
    bound_angle = torch.remainder(angle, 2 * torch.pi)
    return torch.where(bound_angle > torch.pi, bound_angle - 2 * torch.pi, bound_angle)


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
          scalings:Tuple[float,float,float] = (1.,1.,1.),
          data_loading_generator: torch.Generator = torch.default_generator,
          gradient_regulariser = None,
          target:str = 'MSE',
          time_extent = None,
          lr_scheduler = None,
          pre_train_epochs=0,
          device = torch.device('cuda:0'),
          state_scaling = 1000.
          ):

    batch_size = list(batch_size)
    position_scaling = torch.tensor([[[state_scaling, state_scaling]]], device=device)

    aggregation_function = {'Mean Pose': pydpf.FilteringMean(lambda state: torch.concat([state[..., :2], torch.sin(state[..., 2:3]), torch.cos(state[..., 2:3])], dim=-1)), 'ELBO': pydpf.ElBO_Loss()}
    validation_aggregation_function = {'Mean Pose': pydpf.FilteringMean(lambda state: torch.concat([state[..., :2], bind_angle(state[..., 2:3])], dim=-1)), 'ELBO': pydpf.ElBO_Loss()}
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
    encoder = dpf.SSM.observation_model.encoder
    decoder = dpf.SSM.observation_model.decoder
    state_encoder = dpf.SSM.observation_model.state_encoder

    if time_extent is None:
        time_extent = dataset.observation.size(0)-1

    for epoch in range(pre_train_epochs):
        dpf.train()
        train_loss = 0
        total_size = 0
        for i, (state, observation, control) in enumerate(train_loader):
            dpf.update()
            opt.zero_grad()
            observation = observation.to(device).reshape(observation.size(0)*observation.size(1), 3, 24, 24)
            encoded_obs = encoder(observation)
            decoded_obs = decoder(encoded_obs)
            AE_loss = torch.mean((decoded_obs - observation)**2)*scalings[2]
            AE_loss.backward()
            opt.step()
            train_loss += AE_loss.item() * observation.size(1)
            total_size += observation.size(1)
        print(f'Auto_encoder training loss = {train_loss / total_size}')

    start_time = time.time()




    for epoch in range(epochs):

        train_loss = []
        total_size = 0
        dpf.train()
        for state, observation, control in train_loader:
            dpf.update()
            opt.zero_grad()
            batch_size_now = observation.size(1)
            #print(observation.size())
            #observation = einops.rearrange(observation, 't b (c h w) -> (t b) c h w', c=3, h=24, w=24)
            observation = observation.to(device).reshape(observation.size(0)*observation.size(1), 3, 24, 24)
            #print(observation.size())
            state = state.to(device)
            control = control.to(device)
            encoded_obs = encoder(observation)
            decoded_obs = decoder(encoded_obs)
            AE_loss = torch.mean((decoded_obs - observation) ** 2)
            outputs = dpf(n_particles[0], time_extent, aggregation_function, observation=encoded_obs.reshape(100, batch_size_now, encoded_obs.size(1)).contiguous(), ground_truth=state, control=control, gradient_regulariser = gradient_regulariser)
            #print(torch.sqrt(torch.mean(torch.sum(((outputs['Mean Pose'][-1,:,:2] - state[-1,:,:2])*position_scaling)**2, dim=-1))).item())

            cos_loss = torch.mean((outputs['Mean Pose'][:,:,3] - torch.cos(state[:, :, 2]))**2)
            sin_loss = torch.mean((outputs['Mean Pose'][:,:,2] - torch.sin(state[:, :, 2]))**2)
            angle_loss = cos_loss + sin_loss
            elbo_loss = outputs['ELBO'].mean()
            #print(torch.sum((outputs['Mean Pose'][:,:,:2] - state[:outputs['Mean Pose'].size(0),:,:2])**2, dim=-1))
            position_loss = torch.mean(torch.sum((outputs['Mean Pose'][:,:,:2] - state[:,:,:2])**2, dim=-1))
            #print(position_loss)
            loss = scalings[0]*position_loss + scalings[1]*angle_loss + AE_loss*scalings[2] #+ elbo_loss
            #print(outputs['Mean Pose'][:, 0])
            loss.backward()
            train_loss.append(loss.item()*state.size(1))
            opt.step()
            total_size += state.size(1)
        if lr_scheduler is not None:
            lr_scheduler.step()
        train_loss = np.sum(np.array(train_loss)) / total_size
        dpf.update()
        dpf.eval()
        with torch.inference_mode():
            total_size = 0
            validation_Pos_MSE = []
            validation_Angle_MSE = []
            for state, observation, control in validation_loader:
                batch_size_now = observation.size(1)
                observation = observation.to(device).reshape(observation.size(0)*observation.size(1), 3, 24, 24)

                state = state.to(device)
                control = control.to(device)
                encoded_obs = encoder(observation)
                outputs = dpf(n_particles[1], time_extent, validation_aggregation_function, observation=encoded_obs.reshape(100, batch_size_now, encoded_obs.size(1)).contiguous(), ground_truth=state, control=control)
                validation_Pos_MSE.append(torch.mean(torch.sum(((outputs['Mean Pose'][-1,:,:2] - state[-1,:,:2])*position_scaling)**2, dim=-1)).item()*state.size(1))
                validation_Angle_MSE.append(torch.mean(bind_angle(bind_angle( outputs['Mean Pose'][:,:,2]) - bind_angle(state[:,:,2]))**2).item()*state.size(1))
                #print(torch.sum(((outputs['Mean Pose'][:,:,:2] - state[:,:,:2])*position_scaling)**2, dim=-1))
                total_size += state.size(1)
            validation_Pos_MSE= np.sum((np.array(validation_Pos_MSE))) / total_size
            validation_Angle_MSE = np.sum((np.array(validation_Angle_MSE))) / total_size
            if validation_Pos_MSE < best_eval:
                best_eval = validation_Pos_MSE
                best_dict = deepcopy(dpf.state_dict())



        print(f'epoch {epoch + 1}/{epochs}, train loss: {train_loss}, validation position RMSE: {np.sqrt(validation_Pos_MSE)}, validation angle RMSE: {np.sqrt(validation_Angle_MSE)}')
    total_size = 0
    dpf.load_state_dict(best_dict)



    with torch.inference_mode():
        test_Pos_MSE = []
        test_angle_MSE = []


        for state, observation, control in test_loader:
            batch_size_now = observation.size(1)
            observation = observation.to(device).reshape(observation.size(0)*observation.size(1), 3, 24, 24)
            state = state.to(device)
            control = control.to(device)
            encoded_obs = encoder(observation)
            outputs = dpf(n_particles[1], time_extent, validation_aggregation_function, observation=encoded_obs.reshape(100, batch_size_now, encoded_obs.size(1)).contiguous(), ground_truth=state, control =control)
            test_Pos_MSE.append(torch.mean(torch.sum(((outputs['Mean Pose'][-1,:,:2] - state[-1,:,:2])*position_scaling)**2, dim=-1)).item() * state.size(1))
            test_angle_MSE.append(torch.mean(bind_angle(bind_angle( outputs['Mean Pose'][:,:,2]) - bind_angle(state[:outputs['Mean Pose'].size(0),:,2]))**2).item()*state.size(1))
            total_size += state.size(1)
    test_Pos_MSE = np.sum((np.array(test_Pos_MSE))) / total_size
    test_angle_MSE = np.sum((np.array(test_angle_MSE))) / total_size
    print('')
    print(f'test position RMSE: {np.sqrt(test_Pos_MSE)}, test angle RMSE: {np.sqrt(test_angle_MSE)}')
    print(f'Final time = {time.time() - start_time}')
    return test_Pos_MSE, test_angle_MSE