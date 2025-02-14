import torch
from utils.quaternion import quaternion_to_cont6d, qrot, qinv

import numpy as np

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def feats2joints(features, num_joints=22, no_rot=False):
    if num_joints == 22:
        mean = torch.Tensor(np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')).to(features)
        std = torch.Tensor(np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')).to(features)
    elif num_joints == 21:
        mean = torch.Tensor(np.load('./checkpoints/kit/Comp_v6_KLD005/meta/mean.npy')).to(features)
        std = torch.Tensor(np.load('./checkpoints/kit/Comp_v6_KLD005/meta/std.npy')).to(features)
    
    if no_rot:
        mean = torch.cat((mean[:(4+(num_joints-1)*3)], mean[(4+(num_joints-1)*9):]))
        std = torch.cat((std[:(4+(num_joints-1)*3)], std[(4+(num_joints-1)*9):]))
    features = features * std + mean
    return recover_from_ric(features, num_joints)


def remove_padding(tensors, lengths):
    return [
        tensor[:tensor_length]
        for tensor, tensor_length in zip(tensors, lengths)
    ]
