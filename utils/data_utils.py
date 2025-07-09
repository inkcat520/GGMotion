#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import pickle
import random
from utils import kinematics
import torch.nn.functional as F


def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4
    Args
      R: 3x3 rotation matrix
    Returns
      q: 1x4 quaternion
    """
    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x)
    return R


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
    Args
      q: 1x4 quaternion
    Returns
      r: 1x3 exponential map
    Raises
      ValueError if the l2 norm of the quaternion is not close to 1
    """
    if (np.abs(np.linalg.norm(q) - 1) > 1e-3):
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot):
    """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
    Args
      normalizedData: nxd matrix with normalized data
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      origData: data originally used to
    """
    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    if one_hot:
        origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
    else:
        origData[:, dimensions_to_use] = normalizedData

    # potentially ineficient, but only done once per experiment
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
    """
    Converts the output of the neural network to a format that is more easy to
    manipulate for, e.g. conversion to other format or visualization
    Args
      poses: The output from the TF model. A list with (seq_length) entries,
      each with a (batch_size, dim) output
    Returns
      poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
      batch is an n-by-d sequence of poses.
    """
    seq_len = len(poses)
    if seq_len == 0:
        return []

    batch_size, dim = poses[0].shape

    poses_out = np.concatenate(poses)
    poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    poses_out = np.transpose(poses_out, [1, 0, 2])

    poses_out_list = []
    for i in range(poses_out.shape[0]):
        poses_out_list.append(
            unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

    return poses_out_list


def normalize_data(data, data_mean, data_std, dim_to_use, actions, one_hot):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and
    dividing by the standard deviation
    Args
      data: nx99 matrix with data to normalize
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dim_to_use: vector with dimensions used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      data_out: the passed data matrix, but normalized
    """
    data_out = {}
    nactions = len(actions)

    if not one_hot:
        # No one-hot encoding... no need to do anything special
        for key in data.keys():
            data_out[key] = np.divide((data[key] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]

    else:
        # TODO hard-coding 99 dimensions for un-normalized human poses
        for key in data.keys():
            data_out[key] = np.divide((data[key][:, 0:99] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]
            data_out[key] = np.hstack((data_out[key], data[key][:, -nactions:]))

    return data_out


def normalization_stats(completeData):
    """
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33
    Args
      completeData: nx99 matrix with data to normalize
    Returns
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      dimensions_to_use: vector with dimensions used by the model
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def rotmat2euler_torch(R):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above
    :param R:N*3*3
    :return: N*3
    """
    n = R.data.shape[0]
    eul = torch.zeros(n, 3).float().cuda()
    idx_spec1 = (R[:, 0, 2] == 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = torch.zeros(len(idx_spec1), 3).float().cuda()
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = torch.zeros(len(idx_spec2), 3).float().cuda()
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = torch.zeros(len(idx_remain), 3).float().cuda()
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
        eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))
        eul[idx_remain, :] = eul_remain

    return eul


def rotmat2quat_torch(R):
    """
    Converts a rotation matrix to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N * 3 * 3
    :return: N * 4
    """
    rotdiff = R - R.transpose(1, 2)
    r = torch.zeros_like(rotdiff[:, 0])
    r[:, 0] = -rotdiff[:, 1, 2]
    r[:, 1] = rotdiff[:, 0, 2]
    r[:, 2] = -rotdiff[:, 0, 1]
    r_norm = torch.norm(r, dim=1)
    sintheta = r_norm / 2
    r0 = torch.div(r, r_norm.unsqueeze(1).repeat(1, 3) + 0.00000001)
    t1 = R[:, 0, 0]
    t2 = R[:, 1, 1]
    t3 = R[:, 2, 2]
    costheta = (t1 + t2 + t3 - 1) / 2
    theta = torch.atan2(sintheta, costheta)
    q = torch.zeros(R.shape[0], 4).float().cuda()
    q[:, 0] = torch.cos(theta / 2)
    q[:, 1:] = torch.mul(r0, torch.sin(theta / 2).unsqueeze(1).repeat(1, 3))

    return q


def expmap2quat_torch(exp):
    """
    Converts expmap to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N*3
    :return: N*4
    """
    theta = torch.norm(exp, p=2, dim=1).unsqueeze(1)
    v = torch.div(exp, theta.repeat(1, 3) + 0.0000001)
    sinhalf = torch.sin(theta / 2)
    coshalf = torch.cos(theta / 2)
    q1 = torch.mul(v, sinhalf.repeat(1, 3))
    q = torch.cat((coshalf, q1), dim=1)
    return q


def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = torch.eye(3, 3).repeat(n, 1, 1).float().cuda() + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R


def expmap2xyz_torch(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*32*3
    """
    parent, offset, rotInd, expmapInd = kinematics._some_variables()
    xyz = kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def expmap2xyz_torch_cmu(expmap):
    parent, offset, rotInd, expmapInd = kinematics._some_variables_cmu()
    xyz = kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def rot_matrix(r):
    theta = torch.norm(r, p=2, dim=-1)
    d = F.normalize(r, p=2, dim=-1, eps=1e-6)
    x, y, z = torch.unbind(d, dim=-1)
    cos, sin = torch.cos(theta), torch.sin(theta)
    ret = torch.stack((
        cos + (1 - cos) * x * x,
        (1 - cos) * x * y - sin * z,
        (1 - cos) * x * z + sin * y,
        (1 - cos) * x * y + sin * z,
        cos + (1 - cos) * y * y,
        (1 - cos) * y * z - sin * x,
        (1 - cos) * x * z - sin * y,
        (1 - cos) * y * z + sin * x,
        cos + (1 - cos) * z * z,
    ), dim=-1)

    return ret.unflatten(dim=-1, sizes=(3, 3))


def gaussian_kernel(size=5, sigma=1.0):
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()
    return kernel


def gaussian_smooth(data, kernel_size=3, sigma=0.5):
    N, T = data.shape
    kernel = gaussian_kernel(kernel_size, sigma)
    padding = kernel_size // 2
    smoothed_data = np.zeros_like(data)

    for i in range(N):
        padded = np.pad(data[i], (padding, padding), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        smoothed_data[i] = smoothed

    return smoothed_data


def dct_matrix(N, device='cuda'):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    dct_m = torch.tensor(dct_m, dtype=torch.float32, device=device)
    idct_m = torch.tensor(idct_m, dtype=torch.float32, device=device)
    return dct_m, idct_m


def parent_ind(edges):
    root = edges[0, 0]
    leaf = edges[-1, -1]
    parent = [[root, leaf]]
    ind = len(edges) - 1
    while ind > 0:
        if edges[-1, 0] != edges[ind - 1, 0]:
            break
        else:
            parent.append([root, edges[ind - 1, -1]])

        ind -= 1

    return parent


def edge_attr(edges, nodes, hops, sym=True):
    nodes = nodes + 1
    edges = edges + 1
    matrix = np.zeros((nodes, nodes), dtype=int)

    if sym:
        for edge in edges:
            matrix[edge[0], edge[1]] = 1
            matrix[edge[1], edge[0]] = 1
    else:
        for edge in edges:
            matrix[edge[0], edge[1]] = 1

    power_list = []
    if hops < 1:
        hops = edges.shape[0]
    for k in range(hops):
        power = np.linalg.matrix_power(matrix, k + 1)
        power_list.append(power)

    edge_list = []
    for i in range(1, nodes):
        for j in range(1, nodes):
            if i != j:
                for k, power in enumerate(power_list):
                    if power[i, j] > 0:
                        edge_list.append([i - 1, j - 1, k + 1])
                        break

    edges = np.array(edge_list, dtype=int)
    return edges


def group_attr(groups):
    group_dic = {}
    for key in groups:
        group_list = []
        for i in range(len(groups[key])):
            nop = 0
            pre = groups[key][i, 0]
            for j in range(i, -1, -1):
                if pre == groups[key][j - 1, 1]:
                    nop += 1
                    pre = groups[key][j - 1, 0]

            group_list.append(np.append(groups[key][i], nop))

        group_dic[key] = np.array(group_list)

    return group_dic


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def read_data(filename):
    seqs = []
    lines = open(filename).readlines()
    for line in lines:
        if "," in line:
            line = line.strip().split(',')
        else:
            line = line.strip().split(' ')
        if len(line) > 0:
            seqs.append(np.array([np.float32(x) for x in line]))

    seqs = np.array(seqs)
    return seqs

def load_data_3d(path_to_dataset, subjects, actions, sample_rate, input_n, output_n, test_manner):
    """
    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/src/data_utils.py#L216
    """

    seq_len = input_n + output_n
    sampled_seq = []
    complete_seq = []
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]

            if subj != 5:
                for subact in [1, 2]:

                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)
                    action_sequence = read_data(filename)
                    n, d = action_sequence.shape
                    even_list = range(0, n, sample_rate)
                    num_frames = len(even_list)
                    the_sequence = np.array(action_sequence[even_list, :])
                    the_seq = torch.from_numpy(the_sequence).float().cuda()
                    # remove global rotation and translation
                    the_seq[:, 0:6] = 0
                    p3d = expmap2xyz_torch(the_seq)
                    the_sequence = p3d.view(num_frames, -1).cpu().data.numpy()

                    fs = np.arange(0, num_frames - seq_len + 1)
                    fs_sel = fs
                    for i in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + i + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = the_sequence[fs_sel, :]
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)

            else:
                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, 1)
                action_sequence = read_data(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)

                num_frames1 = len(even_list)
                the_sequence1 = np.array(action_sequence[even_list, :])
                the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                the_seq1[:, 0:6] = 0
                p3d1 = expmap2xyz_torch(the_seq1)
                the_sequence1 = p3d1.view(num_frames1, -1).cpu().data.numpy()

                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, 2)
                action_sequence = read_data(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)

                num_frames2 = len(even_list)
                the_sequence2 = np.array(action_sequence[even_list, :])
                the_seq2 = torch.from_numpy(the_sequence2).float().cuda()
                the_seq2[:, 0:6] = 0
                p3d2 = expmap2xyz_torch(the_seq2)
                the_sequence2 = p3d2.view(num_frames2, -1).cpu().data.numpy()

                if test_manner == "8":
                    # 随机取 8 个
                    fs_sel1, fs_sel2 = find_sample_indices(num_frames1, num_frames2, seq_len, 8, input_n)
                elif test_manner == "256":
                    # 随机取 256 个
                    fs_sel1, fs_sel2 = find_sample_indices(num_frames1, num_frames2, seq_len, 256, input_n)
                else:
                    # 全部数据用来测试
                    fs_sel1 = [np.arange(i, i + seq_len) for i in range(num_frames1 - seq_len + 1)]
                    fs_sel2 = [np.arange(i, i + seq_len) for i in range(num_frames2 - seq_len + 1)]

                seq_sel1 = the_sequence1[fs_sel1, :]
                seq_sel2 = the_sequence2[fs_sel2, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel1
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = the_sequence1
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel1), axis=0)
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence1, axis=0)
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)

    # ignore constant joints and joints at same position with other joints
    joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)

    arm_l = np.array([[9, 12], [12, 13], [13, 14], [14, 15], [14, 16]])
    arm_r = np.array([[9, 17], [17, 18], [18, 19], [19, 20], [19, 21]])
    head = np.array([[9, 10], [10, 11]])
    spine = np.array([[8, 8], [8, 9]])
    leg_l = np.array([[8, 0], [0, 1], [1, 2], [2, 3]])
    leg_r = np.array([[8, 4], [4, 5], [5, 6], [6, 7]])

    group = {"spine": spine, "head": head, "arm_l": arm_l, "arm_r": arm_r, "leg_l": leg_l, "leg_r": leg_r}

    return sampled_seq, dimensions_to_ignore, dimensions_to_use, group


def load_data_cmu_3d(path_to_dataset, actions, sample_rate, input_n, output_n, test_manner="all"):
    seq_len = input_n + output_n
    sampled_seq = []
    complete_seq = []
    subj = os.path.basename(os.path.normpath(path_to_dataset))
    for action_idx in np.arange(len(actions)):
        action = actions[action_idx]
        path = os.path.join(path_to_dataset, str(action))
        count = 1
        for file in os.listdir(path):
            if file.endswith(".txt"):
                count = count + 1
        for sub_acts in np.arange(1, count):

            print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, sub_acts))
            filename = '{}/{}/{}_{}.txt'.format(path_to_dataset, action, action, sub_acts)
            action_sequence = read_data(filename)
            n, d = action_sequence.shape
            exptmps = torch.from_numpy(action_sequence).float().cuda()
            xyz = expmap2xyz_torch_cmu(exptmps)
            xyz = xyz.view(-1, 38 * 3)
            xyz = xyz.cpu().data.numpy()
            action_sequence = xyz

            even_list = range(0, n, sample_rate)
            the_sequence = np.array(action_sequence[even_list, :])
            num_frames = len(the_sequence)

            if 'train' in path_to_dataset:
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()

            else:
                # 水滴测试
                # 测试集 随机挑选 8
                if test_manner == "8":
                    fs_sel = find_sample_indices_cmu(num_frames, seq_len, 8, input_n)
                # 测试集 随机挑选 256
                elif test_manner == "256":
                    fs_sel = find_sample_indices_cmu(num_frames, seq_len, 256, input_n)
                # 全部数据用来测试
                else:
                    fs_sel = [np.arange(i, i + seq_len) for i in range(num_frames - seq_len + 1)]

            seq_sel = the_sequence[fs_sel, :]
            if len(sampled_seq) == 0:
                sampled_seq = seq_sel
                complete_seq = the_sequence
            else:
                sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                complete_seq = np.append(complete_seq, the_sequence, axis=0)

    joint_to_ignore = np.array([0, 1, 2, 7, 8, 13, 16, 20, 29, 24, 27, 33, 36])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)

    arm_l = np.array([[9, 13], [13, 14], [14, 15], [15, 18], [15, 16], [16, 17]])
    arm_r = np.array([[9, 19], [19, 20], [20, 21], [21, 24], [21, 22], [22, 23]])
    head = np.array([[9, 10], [10, 11], [11, 12]])
    spine = np.array([[8, 8], [8, 9]])
    leg_l = np.array([[8, 0], [0, 1], [1, 2], [2, 3]])
    leg_r = np.array([[8, 4], [4, 5], [5, 6], [6, 7]])

    group = {"spine": spine, "head": head, "arm_l": arm_l, "arm_r": arm_r, "leg_l": leg_l, "leg_r": leg_r}

    return sampled_seq, dimensions_to_ignore, dimensions_to_use, group


def find_sample_indices(frame_num1, frame_num2, seq_len, seq_num, input_n):
    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, seq_num//2):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2


def find_sample_indices_cmu(frame_num, seq_len, seq_num, input_n):
    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    source_seq_len = 50
    target_seq_len = 25
    total_frames = source_seq_len + target_seq_len
    idxo = None
    for _ in np.arange(0, seq_num):
        idx = rng.randint(0, frame_num - total_frames)
        idxs = np.arange(idx + source_seq_len - input_n, idx + source_seq_len - input_n + seq_len)
        if idxo is None:
            idxo = idxs
        else:
            idxo = np.vstack((idxo, idxs))

    return idxo


def load_data_3dpw_3d(path_to_dataset, input_n, output_n):
    seq_len = input_n + output_n
    sampled_seq = []
    complete_seq = []
    subj = os.path.basename(os.path.normpath(path_to_dataset))
    for file in os.listdir(path_to_dataset):
        path = os.path.join(path_to_dataset, file)
        if path.endswith(".pkl"):
            act = file[:-7]
            sub_acts = file[-5]
            print("Reading subject {0}, action {1}, subaction {2}".format(subj, act, sub_acts))
            with open(path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                joint_pos = data['jointPositions']
                for i in range(len(joint_pos)):
                    the_sequence = joint_pos[i].astype(np.float32)
                    the_sequence = the_sequence - the_sequence[:, 0:3].repeat(24, axis=0).reshape(-1, 72)
                    num_frames = the_sequence.shape[0]
                    fs = np.arange(0, num_frames - seq_len + 1)
                    fs_sel = fs
                    for j in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + j + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = the_sequence[fs_sel, :]
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)

    sampled_seq = sampled_seq * 1000
    joint_to_ignore = np.array([0])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)

    arm_l = np.array([[8, 13], [13, 16], [16, 18], [18, 20], [20, 22]])
    arm_r = np.array([[8, 12], [12, 15], [15, 17], [17, 19], [19, 21]])
    head = np.array([[8, 11], [11, 14]])
    spine = np.array([[2, 2], [2, 5], [5, 8]])
    leg_l = np.array([[2, 1], [1, 4], [4, 7], [7, 10]])
    leg_r = np.array([[2, 0], [0, 3], [3, 6], [6, 9]])

    group = {"spine": spine, "head": head, "arm_l": arm_l, "arm_r": arm_r, "leg_l": leg_l, "leg_r": leg_r}

    return sampled_seq, dimensions_to_ignore, dimensions_to_use, group
