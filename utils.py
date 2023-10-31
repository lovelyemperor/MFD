from datetime import datetime
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mpl_toolkits.mplot3d as p3d
import math
import os
import torch.optim as optim
from scipy.special import binom
import scipy.io as io
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal, chi2
import scipy.io as io
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Scatter
# from pod_loss import *
import sys
import conf.config as conf
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except Exception:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)

def Check(input, target):
    ret = input * target
    print('input: ', input.pow(2).sum(dim=1))
    print('target: ', target.pow(2).sum(dim=1))
    ret = torch.sum(ret, dim=1)
    print('ret: ', ret)
    m = ret.clone().detach()
    # print('m: ',m)
    a = m.cpu().numpy()
    # print('a:',a)
    b = np.linspace(-1, 1, num=400)
    # print('b:', b)
    plt.hist(a, b, histtype='bar', rwidth=0.8)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('/home/data/ZXW/Data/CIFAR10/分布的图片1.png')
    # plt.show()
    ret = 1 - ret
    ret = ret ** 2
    ret = torch.mean(ret)
    return ret.cuda()

# def blockDecorelation(output):
#     # flag = 0
#     # values = torch.cat([torch.zeros(5), torch.ones(12)])
#     # indices = torch.randperm(len(values))
#     # random_values = values[indices]
#     # # k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17 = random_values
#     # loss = 0
#     # for i in range(17):
#     #     if i == 0:
#     #         # k = k1
#     #         loss = NewDecorrelationBetweenDim1(output[i], 1)
#         # elif i == 1:
#         #     k == k2
#         # elif i == 2:
#         #     k == k3
#         # elif i == 3:
#         #     k == k4
#         # elif i == 4:
#         #     k == k5
#         # elif i == 5:
#         #     k == k6
#         # elif i == 6:
#         #     k == k7
#         # elif i == 7:
#         #     k == k8
#         # elif i == 8:
#         #     k == k9
#         # elif i == 9:
#         #     k == k10
#         # elif i == 10:
#         #     k == k11
#         # elif i == 11:
#         #     k == k12
#         # elif i == 12:
#         #     k == k13
#         # elif i == 13:
#         #     k == k14
#         # elif i == 14:
#         #     k == k15
#         # elif i == 15:
#         #     k == k16
#         # elif i == 16:
#         #     k == k17
#     # k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17 = random_values
#     loss = 0
#     for i in range(17):
#         if i == 0:
#             # k = k1
#             loss += NewDecorrelationBetweenDim1(output[i], 1)
#         else:
#             for j in range(3):
#                 loss += NewDecorrelationBetweenDim1(output[i][j], 1)
#     return loss
def blockDecorelation(output):
    loss = 0
    for i in range(17):
        if i == 0:
            loss += NewDecorrelationBetweenDim1(output[i], 1)
        else:
            for j in range(3):
                loss += NewDecorrelationBetweenDim1(output[i][j], 1)
    # for i in [0, 3, 7, 13, 16]:
    #     if flag == 0:
    #         loss = NewDecorrelationBetweenDim1(output[i], 1)
    #     else:
        # flag = 1
    return loss

def vgg_blockDecorelation(output):
    # flag = 0
    loss = 0
    for i in range(13):
        # if 0 < i < 7:
        #     loss += NewDecorrelationBetweenDim1(output[i], 1)
        # else:
        loss += NewDecorrelationBetweenDim1(output[i], 1)
    return loss

def mobile_v2_blockDecorelation(output):
    # flag = 0
    loss = 0
    for i in range(19):
        if i == 0 or i == 18:
            loss += NewDecorrelationBetweenDim1(output[i], 1)
        elif i == 1:
            for k in range(1):
                # loss += NewDecorrelationBetweenDim1(output[i][k], 1)
                loss += 0
        else:
            for j in range(1):
                loss += NewDecorrelationBetweenDim1(output[i][j], 1)
    return loss

def CorrectMinLoss(input, target):
    # noise = torch.randn((label.data == 0).sum().item(), target.shape[1])
    # noise = noise - torch.mean(noise)
    # target[label.data == 0] = target[label.data == 0] + (0.1**0.5)*noise.cuda()
    ret = input * target
    # ret = torch.mean(ret)
    # ret = (1 - ret).pow(2)
    ret_sum = torch.sum(ret, dim=1)
    ret = 1 - ret_sum

    ret = ret ** 2
    ret = torch.mean(ret)
    return ret
# def CorrectMinLoss(input, target):
#     ret = input * target
#     ret = torch.sum(ret, dim=1)
#     ret = 1 - ret + 0.1
#     ret = ret.pow(2)
#     ret = torch.mean(ret)
#     return ret

def ErrorMinLoss(input, all_target, ret_sum):
    # noise = torch.randn(all_target.shape[0], all_target.shape[1], all_target.shape[2])
    # noise = noise - torch.mean(noise)
    # all_target = all_target + (0.01**0.5)*noise.cuda()
    all_input = input.unsqueeze(1)
    # print(all_input.shape)
    all_input = all_input.repeat(1, 10, 1)
    # print(all_input.shape)
    ret = all_input * all_target
    ret = torch.sum(ret, dim=2)
    # print(ret.shape)
    ret = ret + 1 / 9
    ret = ret ** 2
    ret_sum = ret_sum + 1/9
    ret_sum = ret_sum ** 2
    ret = torch.sum(ret, dim=1)
    ret = (ret - ret_sum)/9
    ret = torch.mean(ret)
    # print(ret.shape)
    return ret

# def ErrorMinLoss(input, all_error_target):
#     all_ret = torch.tensor(0).cuda()
#     for i in range(99):
#         error_target = all_error_target[:, i, :]
#         ret = input * error_target
#         ret = torch.sum(ret, dim=1)
#         ret = ret + 1 / 99
#         ret = ret ** 2
#         ret = torch.mean(ret)
#         all_ret = all_ret + ret
#     return all_ret / 99
# def ErrorMinLoss(input, all_error_target):
#     all_ret = torch.tensor(0).cuda()
#     for i in range(99):
#         error_target = all_error_target[i]
#         ret = input * error_target
#         ret = torch.sum(ret, dim=1)
#         ret = ret + 1/99
#         ret = ret.pow(2)
#         ret = torch.mean(ret)
#         all_ret = all_ret + ret
#     return all_ret/99

class CConvarianceLoss(nn.Module):
    def __init__(self, map_PEDCC):
        super(CConvarianceLoss, self).__init__()
        self.map_PEDCC = map_PEDCC
        return
    def forward(self, feature, label):
        average_feature = self.map_PEDCC[label.long().data].float().cuda()
        # print('average_feature:', average_feature.requires_grad)
        feature = feature - average_feature
        # covariance100_loss = torch.sum(feature)
        # print('feature:', feature.requires_grad)
        new_covariance100 = 1 / (feature.shape[0] - 1) * torch.mm(feature.T, feature).float()
        # print('covariance100:', covariance100.shape)
        # covariance100_loss = torch.sum(new_covariance100)
        # print('new_covariance100:', new_covariance100.requires_grad)
        # covariance100 = 1/3 * covariance100 + 2/3 * new_covariance100
        covariance100 = new_covariance100
        # covariance100_loss = torch.sum(covariance100)
        # # print('covariance100:', covariance100.requires_grad)
        covariance100_loss = torch.sum(pow(covariance100, 2)) - torch.sum(pow(torch.diagonal(covariance100), 2))
        # print(covariance100.shape[0])
        # covariance100_loss = covariance100_loss / (covariance100.shape[0] - 1)
        # covariance100_loss = torch.norm(covariance100)
        # print('covariance100_loss:', covariance100_loss.requires_grad)
        return covariance100_loss, covariance100

def CovarianceLoss(covariance100, feature, label, map_PEDCC):
    # feature = torch.mm(feature, mapT_PEDCC)
    # feature = feature[:, -99:]
    # print('feature:', feature.shape)
    # for i in range(covariance100.shape[0]):
    #     each_feature = feature[label.data == i]
    #     if each_feature.shape[0] > 1:
    #         average_feature = torch.mean(each_feature, dim=0)
    #         each_feature = each_feature - average_feature
    #         each_covariance = 1 / (each_feature.shape[0] - 1) * torch.mm(each_feature.T, each_feature).float()
    #         covariance100[i] = (500 - each_feature.shape[0]) / 500 * covariance100[i] + each_feature.shape[0] / 500 * each_covariance

    average_feature = map_PEDCC[label.long().data].float().cuda()
    # print('average_feature:', average_feature.requires_gard)
    feature = feature - average_feature
    print('feature:', feature.requires_gard)
    new_covariance100 = 1 / (feature.shape[0] - 1) * torch.mm(feature.T, feature).float()
    print('new_covariance100:', new_covariance100.requires_gard)
    covariance100 = 0.99 * covariance100 + 0.01 * new_covariance100
    print('covariance100:', covariance100.requires_gard)
    covariance100_loss = torch.norm(covariance100)
    print('covariance100_loss:', covariance100_loss.requires_gard)
    return covariance100_loss

def MinCovarianceLoss(covariance100, triangle100):
    ret = torch.sum(pow(covariance100, 2)) - torch.sum(pow(triangle100, 2))
    # ret = torch.tensor(0).cuda()- torch.sum(torch.diagonal(covariance100[1]))
    # for i in range(covariance100.shape[0]):
    #     ret = ret + (torch.sum(covariance100[i]) - torch.sum(torch.diagonal(covariance100[i]))) / 2
    # - torch.sum(pow(torch.diagonal(covariance100[0]),2))
    # ret = torch.sum(covariance100[0] ** 2) + torch.sum(covariance100[1]**2) + torch.sum(covariance100[2]**2) + torch.sum(covariance100[3]**2) + torch.sum(covariance100[4]**2) +
    #       torch.sum(covariance100[5]**2) + torch.sum(covariance100[6]**2) + torch.sum(covariance100[7]**2) + torch.sum(covariance100[8]**2) + torch.sum(covariance100[9]**2) + \
    #       torch.sum(covariance100[10]**2)
    # ret = ret + torch.sum(covariance100[11]**2)
    #
    # ret = ret + torch.sum(covariance100[12]**2)
    # ret = ret + torch.sum(covariance100[13]**2)
    # ret = ret + torch.sum(covariance100[14]**2)
    # ret = ret + torch.sum(covariance100[15]**2)
    # ret = ret + torch.sum(covariance100[16]**2)
    # ret = ret + torch.sum(covariance100[17]**2)
    # ret = ret + torch.sum(covariance100[18]**2)
    # ret = ret + torch.sum(covariance100[19]**2)
    # ret = ret + torch.sum(covariance100[20]**2)
    # ret = ret + torch.sum(covariance100[21]**2)
    # ret = ret + torch.sum(covariance100[22]**2) - torch.sum(torch.diagonal(covariance100[22]))
    # ret = ret + torch.sum(covariance100[23]**2) - torch.sum(torch.diagonal(covariance100[23]))
    # ret = ret + torch.sum(covariance100[24]**2) - torch.sum(torch.diagonal(covariance100[24]))
    # ret = ret + torch.sum(covariance100[25]**2) - torch.sum(torch.diagonal(covariance100[25]))
    # ret = ret + torch.sum(covariance100[26]**2) - torch.sum(torch.diagonal(covariance100[26]))
    # ret = ret + torch.sum(covariance100[27]**2) - torch.sum(torch.diagonal(covariance100[27]))
    # ret = ret + torch.sum(covariance100[28]**2) - torch.sum(torch.diagonal(covariance100[28]))
    # ret = ret + torch.sum(covariance100[29]**2) - torch.sum(torch.diagonal(covariance100[29]))
    # ret = ret + torch.sum(covariance100[30]**2) - torch.sum(torch.diagonal(covariance100[30]))
    # ret = ret + torch.sum(covariance100[31]**2) - torch.sum(torch.diagonal(covariance100[31]))
    # ret = ret + torch.sum(covariance100[32]**2) - torch.sum(torch.diagonal(covariance100[32]))
    # ret = ret + torch.sum(covariance100[33]**2) - torch.sum(torch.diagonal(covariance100[33]))
    # ret = ret + torch.sum(covariance100[34]**2) - torch.sum(torch.diagonal(covariance100[34]))
    # ret = ret + torch.sum(covariance100[35]**2) - torch.sum(torch.diagonal(covariance100[35]))
    # ret = ret + torch.sum(covariance100[36]**2) - torch.sum(torch.diagonal(covariance100[36]))
    # ret = ret + torch.sum(covariance100[37]**2) - torch.sum(torch.diagonal(covariance100[37]))
    # ret = ret + torch.sum(covariance100[38]**2) - torch.sum(torch.diagonal(covariance100[38]))
    # ret = ret + torch.sum(covariance100[39]**2) - torch.sum(torch.diagonal(covariance100[39]))
    # ret = ret + torch.sum(covariance100[40]**2) - torch.sum(torch.diagonal(covariance100[40]))
    # ret = ret + torch.sum(covariance100[41]**2) - torch.sum(torch.diagonal(covariance100[41]))
    # ret = ret + torch.sum(covariance100[42]**2) - torch.sum(torch.diagonal(covariance100[42]))
    # ret = ret + torch.sum(covariance100[43]**2) - torch.sum(torch.diagonal(covariance100[43]))
    # ret = ret + torch.sum(covariance100[44]**2) - torch.sum(torch.diagonal(covariance100[44]))
    # ret = ret + torch.sum(covariance100[45]**2) - torch.sum(torch.diagonal(covariance100[45]))
    # ret = ret + torch.sum(covariance100[46]**2) - torch.sum(torch.diagonal(covariance100[46]))
    # ret = ret + torch.sum(covariance100[47]**2) - torch.sum(torch.diagonal(covariance100[47]))
    # ret = ret + torch.sum(covariance100[48]**2) - torch.sum(torch.diagonal(covariance100[48]))
    # ret = ret + torch.sum(covariance100[49]**2) - torch.sum(torch.diagonal(covariance100[49]))
    # ret = ret + torch.sum(covariance100[50]**2) - torch.sum(torch.diagonal(covariance100[50]))
    # ret = ret + torch.sum(covariance100[51]**2) - torch.sum(torch.diagonal(covariance100[51]))
    # ret = ret + torch.sum(covariance100[52]**2) - torch.sum(torch.diagonal(covariance100[52]))
    # ret = ret + torch.sum(covariance100[53]**2) - torch.sum(torch.diagonal(covariance100[53]))
    # ret = ret + torch.sum(covariance100[54]**2) - torch.sum(torch.diagonal(covariance100[54]))
    # ret = ret + torch.sum(covariance100[55]**2) - torch.sum(torch.diagonal(covariance100[55]))
    # ret = ret + torch.sum(covariance100[56]**2) - torch.sum(torch.diagonal(covariance100[56]))
    # ret = ret + torch.sum(covariance100[57]**2) - torch.sum(torch.diagonal(covariance100[57]))
    # ret = ret + torch.sum(covariance100[58]**2) - torch.sum(torch.diagonal(covariance100[58]))
    # ret = ret + torch.sum(covariance100[59]**2) - torch.sum(torch.diagonal(covariance100[59]))
    # ret = ret + torch.sum(covariance100[60]**2) - torch.sum(torch.diagonal(covariance100[60]))
    # ret = ret + torch.sum(covariance100[61]**2) - torch.sum(torch.diagonal(covariance100[61]))
    # ret = ret + torch.sum(covariance100[62]**2) - torch.sum(torch.diagonal(covariance100[62]))
    # ret = ret + torch.sum(covariance100[63]**2) - torch.sum(torch.diagonal(covariance100[63]))
    # ret = ret + torch.sum(covariance100[64]**2) - torch.sum(torch.diagonal(covariance100[64]))
    # ret = ret + torch.sum(covariance100[65]**2) - torch.sum(torch.diagonal(covariance100[65]))
    # ret = ret + torch.sum(covariance100[66]**2) - torch.sum(torch.diagonal(covariance100[66]))
    # ret = ret + torch.sum(covariance100[67]**2) - torch.sum(torch.diagonal(covariance100[67]))
    # ret = ret + torch.sum(covariance100[68]**2) - torch.sum(torch.diagonal(covariance100[68]))
    # ret = ret + torch.sum(covariance100[69]**2) - torch.sum(torch.diagonal(covariance100[69]))
    # ret = ret + torch.sum(covariance100[70]**2) - torch.sum(torch.diagonal(covariance100[70]))
    # ret = ret + torch.sum(covariance100[71]**2) - torch.sum(torch.diagonal(covariance100[71]))
    # ret = ret + torch.sum(covariance100[72]**2) - torch.sum(torch.diagonal(covariance100[72]))
    # ret = ret + torch.sum(covariance100[73]**2) - torch.sum(torch.diagonal(covariance100[73]))
    # ret = ret + torch.sum(covariance100[74]**2) - torch.sum(torch.diagonal(covariance100[74]))
    # ret = ret + torch.sum(covariance100[75]**2) - torch.sum(torch.diagonal(covariance100[75]))
    # ret = ret + torch.sum(covariance100[76]**2) - torch.sum(torch.diagonal(covariance100[76]))
    # ret = ret + torch.sum(covariance100[77]**2) - torch.sum(torch.diagonal(covariance100[77]))
    # ret = ret + torch.sum(covariance100[78]**2) - torch.sum(torch.diagonal(covariance100[78]))
    # ret = ret + torch.sum(covariance100[79]**2) - torch.sum(torch.diagonal(covariance100[79]))
    # ret = ret + torch.sum(covariance100[80]**2) - torch.sum(torch.diagonal(covariance100[80]))
    # ret = ret + torch.sum(covariance100[81]**2) - torch.sum(torch.diagonal(covariance100[81]))
    # ret = ret + torch.sum(covariance100[82]**2) - torch.sum(torch.diagonal(covariance100[82]))
    # ret = ret + torch.sum(covariance100[83]**2) - torch.sum(torch.diagonal(covariance100[83]))
    # ret = ret + torch.sum(covariance100[84]**2) - torch.sum(torch.diagonal(covariance100[84]))
    # ret = ret + torch.sum(covariance100[85]**2) - torch.sum(torch.diagonal(covariance100[85]))
    # ret = ret + torch.sum(covariance100[86]**2) - torch.sum(torch.diagonal(covariance100[86]))
    # ret = ret + torch.sum(covariance100[87]**2) - torch.sum(torch.diagonal(covariance100[87]))
    # ret = ret + torch.sum(covariance100[88]**2) - torch.sum(torch.diagonal(covariance100[88]))
    # ret = ret + torch.sum(covariance100[89]**2) - torch.sum(torch.diagonal(covariance100[89]))
    # ret = ret + torch.sum(covariance100[90]**2) - torch.sum(torch.diagonal(covariance100[90]))
    # ret = ret + torch.sum(covariance100[91]**2) - torch.sum(torch.diagonal(covariance100[91]))
    # ret = ret + torch.sum(covariance100[92]**2) - torch.sum(torch.diagonal(covariance100[92]))
    # ret = ret + torch.sum(covariance100[93]**2) - torch.sum(torch.diagonal(covariance100[93]))
    # ret = ret + torch.sum(covariance100[94]**2) - torch.sum(torch.diagonal(covariance100[94]))
    # ret = ret + torch.sum(covariance100[95]**2) - torch.sum(torch.diagonal(covariance100[95]))
    # ret = ret + torch.sum(covariance100[96]**2) - torch.sum(torch.diagonal(covariance100[96]))
    # ret = ret + torch.sum(covariance100[97]**2) - torch.sum(torch.diagonal(covariance100[97]))
    # ret = ret + torch.sum(covariance100[98]**2) - torch.sum(torch.diagonal(covariance100[98]))
    # ret = ret + torch.sum(covariance100[99]**2) - torch.sum(torch.diagonal(covariance100[99]))
    print('ret:', ret)
    return ret / 200

def NewLoss(input, target):
    ret = pow((input - target), 2)
    ret = torch.sum(ret, dim=1)
    ret = torch.mean(ret)
    return ret
def FoursqrtLoss(input, target):
    ret = input * target
    ret = torch.sum(ret,dim=1)
    ret = torch.acos(ret)
    # ret = ret ** 2
    ret = torch.mean(ret)
    return ret
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

def get_acc_top5(output, label):
    total = output.shape[0]
    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    correct_k = correct[:5].view(-1).float().sum(0)
    return correct_k / total
def top5_accuracy(output, label):
    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.reshape(1, -1).expand_as(pred))
    correct_k = correct[:5].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.item() / label.size(0)

def train(net, train_data, valid_data, num_epochs, criterion, modelname=None):
    LR = 0.1
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        if epoch in [0, 30, 60, 90]:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)

            # forward
            output = net(im)[0]
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()

                output = net(im)[0]
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc1: %f, Valid Loss: %f, Valid Acc: %f, LR: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open('./FM.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        # if valid_acc / len(valid_data)>0.77:
        #     break
    if modelname:
        torch.save(net, modelname)

def train_imagenet(net, train_data, valid_data, num_epochs, criterion, modelname=None):
    LR = 0.1
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        if epoch in [0, 20, 40, 60]:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_acc = 0
        train_acc_top5 = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)

            # forward
            output = net(im, label)[0]
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_acc += get_acc(output, label)
            train_acc_top5 += get_acc_top5(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            valid_acc_top5 = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()

                output = net(im)[0]
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
                valid_acc_top5 += get_acc_top5(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Train top5: %f, Valid Loss: %f, Valid Acc: %f, Valid top5: %f, LR: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), train_acc_top5 / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), valid_acc_top5 / len(valid_data), LR))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open('./resnet18-imagenet-log', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        # if train_acc / len(train_data)>0.9995:
        #     break
    if modelname:
        torch.save(net, modelname)

def train_img_soft_mse(net, train_data, valid_data, num_epochs, criterion, criterion1, modelname=None):
    LR = 0.1
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    prev_time = datetime.now()
    map_dict = read_pkl()
    for epoch in range(num_epochs):
        if epoch in [0, 20, 40, 60]:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_acc = 0
        train_acc_top5 = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                tensor_empty = torch.Tensor([]).cuda()
                for label_index in label:
                    tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()), 0)

                label_mse_tensor = tensor_empty.view(-1, 512)
                label_mse_tensor = label_mse_tensor.cuda()

            # forward
            output = net(im)[0]
            loss1 = criterion(output, label)
            loss2 = criterion1(net(im)[1], label_mse_tensor)
            loss = loss1 + loss2
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_acc += get_acc(output, label)
            train_acc_top5 += get_acc_top5(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            valid_acc_top5 = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()

                output = net(im)[0]
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
                valid_acc_top5 += get_acc_top5(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Train top5: %f, Valid Loss: %f, Valid Acc: %f, Valid top5: %f,"
                " LR: %f, Train Loss1: %f, Train Loss2: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), train_acc_top5 / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), valid_acc_top5 / len(valid_data), LR, train_loss1 / len(train_data), train_loss2 / len(train_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open('./resnet18-imagenet-log', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()

        # if train_acc / len(train_data) > 0.9995:
        #     break
    if modelname:
        torch.save(net, modelname)

def train_soft_mse(net, train_data, valid_data, num_epochs, criterion, criterion1, modelname=None):   #  AMSoftmax_PEDCCloss
    LR = 0.1
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    prev_time = datetime.now()
    map_dict = read_pkl()
    for epoch in range(num_epochs):
        if epoch in [0, 30, 60, 90]:  # update the lr, optimizer updated synchronize
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_acc = 0
        net = net.train()
        for im, label in tqdm(train_data):
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                tensor_empty = torch.Tensor([]).cuda()
                for label_index in label:                              # according to the label info we extract the label info in PEDCC, then concatenate them
                    tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()), 0)

                label_mse_tensor = tensor_empty.view(-1, 100)          # orignate the PEDCC label info
                label_mse_tensor = label_mse_tensor.cuda()

            # forward
            output = net(im)     # now the output is a vector, whose length is the number of classes
            loss1 = criterion(output, label)    # loss1 is amsoftmaxloss
            # loss2 = criterion1(net(im)[2], label_mse_tensor)   # loss2 is mseloss calculating  the distance between the feature and PEDCC center
            loss2 = 0
            loss = loss1
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)  # time calculating
        m, s = divmod(remainder, 60)                                 # time calculating
        time_str = "Time %02d:%02d:%02d" % (h, m, s)                 # define the time format
        if valid_data is not None:                                   # calculate the valid loss
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()

                output = net(im)
                loss = criterion(output, label)                       # Only amsoftmax loss is considered here
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR, train_loss1 / len(train_data), train_loss2 / len(train_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open('./wideped100.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        # if train_acc / len(train_data) > 0.9995:
        #     break
    if modelname:
        torch.save(net, modelname)


def train_soft_mse_zl_traditional(net, train_data, valid_data, num_epochs,criterion1, modelname=None, top_acc=0.60):

    if torch.cuda.is_available():                                      # Initialize the network
        # net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    # net = net.to(device)
    LR = 0.1
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.RMSprop(net.parameters(), lr=0.1, alpha=0.9)
    # optimizer = optim.Adagrad(net.parameters(), lr=0.1)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

    prev_time = datetime.now()
    map_dict = read_pkl()                                              # PEDCC access here
    for epoch in range(num_epochs):
        flag = 0
        if epoch in [0, 30, 60, 90]:
        # if epoch in [0, 60, 90, 120, 150, 180]:
        # if epoch in [0, 40, 60, 80]:
            if epoch != 0:
                LR *= 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR
                # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.98, weight_decay=5e-4)
                # optimizer = optim.RMSprop(net.parameters(), lr=LR, alpha=0.9)
                # optimizer = optim.Adagrad(net.parameters(), lr=LR)
        # if epoch >= 10:
        #     LR = 0.1 * math.pow(0.96,epoch)
        #     optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

        train_loss = 0
        train_loss1 = 0

        train_acc = 0
        net = net.train()
        for im, label in tqdm(train_data):
            if torch.cuda.is_available():
                label = label
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)



            # forward
            output = net(im)[0]                                    # the output to do classification
            # if flag==0:
            #     temp = output.view(-1)
            #     temp = temp.cpu().detach().numpy()
            #     pylab.hist(temp,bins=50,normed=1)
            #     pylab.savefig('./NOBN_epoch'+str(epoch))
            #     pylab.close()
            #     flag=1



            loss1 = criterion1(output, label)
            # print(loss1)
            # if flag==0:
            #     output.register_hook(print_grad)
            flag = 1
            loss = loss1
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            train_loss += loss.data
            train_loss1 += loss1.item()


            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    label = label
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)[0]
                loss = criterion1(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, Train Loss1: %f, LR:%f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), train_loss1 / len(train_data), LR))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        # (MIN+MO_subclass)+log(MIN+MO)
        # f = open('./ImageNet/res34/baseline.txt', 'a+')
        # f = open('./cifar100/lunwenres18/bn+wnorm+mos1t2*0-01.txt', 'a+')
        # f = open('./oldNorm/tiny/369/baseline+bn.txt', 'a+')
        # f = open('./cifarRes18_AM300-5_3.txt', 'a+')
        f = open('./tinywide_Baseline_4.txt', 'a+')
        # f = open('./faceRes50_AM100-5_BB.txt', 'a+')
        # f = open('./cifar/baseline+bn_2.txt', 'a+')
        # # f = open('./tiny/0-75bn+oneMinAi.txt', 'a+')
        # f = open('./tiny_resnet50_BN0-8_3.txt', 'a+')
        # # f = open('./cifar100/resnet18/norm/sqrt(2*MIN+MO)_1.txt', 'a+')
        # # f = open('./newNorm/tiny/Min+MoLoss/tiny_zbnsqrt512_sqrt(sum(zj+subc-1)^2+3*(1-zi)^2)_wnorm.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        # for i in range(len(grad_list)):
        #     np.save('./2/'+str(i)+'.npy',grad_list[i])
        # print(valid_acc / len(valid_data))

        # if (valid_acc / len(valid_data)) > top_acc:
        #     top_acc = (valid_acc / len(valid_data))
        #     print("超过了")
        #     torch.save(net, str(epoch) + 'TOPACC_epoch.pkl')
        if epoch % 20 == 0:
            torch.save(net, modelname+str(epoch)+'_epoch.pkl')

def draw(history):
    np.savetxt('./out_acc/out_softmax+FCN+cifar100.txt', history['acc_val'], fmt="%.4f")
    epochs = range(1, len(history['loss_train']) + 1)
    plt.plot(epochs, history['loss_train'], 'blue', label='Training loss')
    plt.plot(epochs, history['loss1_train'], 'green', label='Training loss1')
    plt.plot(epochs, history['loss2_train'], 'yellow', label='Training loss2')
    plt.plot(epochs, history['loss_val'], 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.imsave('E:/acc_and_loss/Training and Validation loss.jpg')
    plt.savefig('./Training and Validation loss.jpg')
    plt.figure()
    epochs = range(1, len(history['acc_train']) + 1)
    plt.plot(epochs, history['acc_train'], 'b', label='Training accuracy')
    plt.plot(epochs, history['acc_val'], 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy of Softmax Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.figure()

    # epochs = range(1, len(history['acc_train']) + 1)
    # plt.plot(epochs, history['ratio_train'], 'b', label='Training ratio')
    # plt.plot(epochs, history['ratio_val'], 'r', label='Validation ratio')
    # plt.title('Training and validation ratio')
    # plt.xlabel('Epochs')
    # plt.ylabel('ratio')
    # plt.legend()
    # plt.figure()
    #plt.imsave('E:/acc_and_loss/Training and validation acc.jpg')
    # plt.savefig('./Training and validation acc.jpg')
    # foo_fig = plt.gcf()
    # foo_fig.savefig('./Softmax_Loss_TrainingandValidationloss.eps', format='eps', dpi=1000)
    plt.show()

def drawHistogram(data, cata):
    # print(data)
    data = data.cpu().detach().numpy()
    np.savetxt('/home/data/ZXW/Data/CIFAR10/kafangData.txt', data)
    # f = open('/home/data/ZXW/Data/CIFAR10/kafangData.txt', 'a+')
    # f.write(data + '\n')
    # f.close()
    # plt.hist(data.cpu().detach().numpy(), 500, color='blue', alpha=0.5)
    # plt.title('Kafang %d Histogram' % cata)
    # plt.xlabel('value')
    # plt.ylabel('frequency')
    b = np.linspace(0, 1, num=1000)#1.5, num=12000
    plt.hist(data, b, histtype='bar', rwidth=0.8)
    # plt.legend()
    plt.title('Kafang %d Histogram' % cata)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.legend()
    str = ('/home/data/ZXW/Data/CIFAR100/Kafang分布的图片%d.png' % cata)
    plt.savefig(str)
    plt.show()

def drawTestHistogram(output2_1, label, map_PEDCC, mapgT_PEDCC):
    tensor_empty1 = map_PEDCC[label.long()].float().cuda()
    label_mse_tensor = tensor_empty1.view(label.shape[0], -1)  # (batchSize, dimension)
    label_mse_tensor = label_mse_tensor.cuda()

    res = GetDiffsqrt(output2_1, label_mse_tensor)
    epoch_str = ("Test Average%d: %f, Covirance%d: %f, " %
                 (label[0].data, torch.mean(res), label[0].data, torch.var(res, unbiased=False)))
    print(epoch_str)
    # if label[0].data < 10:
    #     drawHistogram(res, label[0].data)
        # print('res.shape:', res.shape)

def GetDiffsqrt(Y_Feature, label_mse_tensor):
    res = (Y_Feature - label_mse_tensor) ** 2
    res = torch.sum(res, dim=1)
    return res

def vGet100FreedomDegree(v_label, v_Feature, map_PEDCC, mapgT_PEDCC, epoch):
    # freedom_degree0 = torch.Tensor([]).cuda()
    # freedom_degree1 = torch.Tensor([]).cuda()
    # freedom_degree2 = torch.Tensor([]).cuda()
    # freedom_degree3 = torch.Tensor([]).cuda()
    # freedom_degree4 = torch.Tensor([]).cuda()
    # freedom_degree5 = torch.Tensor([]).cuda()
    # freedom_degree6 = torch.Tensor([]).cuda()
    # freedom_degree7 = torch.Tensor([]).cuda()
    # freedom_degree8 = torch.Tensor([]).cuda()
    # freedom_degree9 = torch.Tensor([]).cuda()
    freedom_degree100 = torch.zeros([100, 100], dtype=torch.float32).cuda()


    tensor_empty1 = map_PEDCC[v_label.long()].float().cuda()
    label_mse_tensor = tensor_empty1.view(v_label.shape[0], -1)  # (batchSize, dimension)
    label_mse_tensor = label_mse_tensor.cuda()

    v_Feature = torch.mm(v_Feature, mapgT_PEDCC)
    v_Feature = v_Feature[:, -99:]
    label_mse_tensor = torch.mm(label_mse_tensor, mapgT_PEDCC)
    label_mse_tensor = label_mse_tensor[:, -99:]

    diff_sqrt = GetDiffsqrt(v_Feature, label_mse_tensor)
    index_label = torch.zeros([100]).cuda()
    for i in range(diff_sqrt.shape[0]):
        freedom_degree100[v_label[i].long(), index_label[v_label[i].long()].long()] = diff_sqrt[i].float()
        index_label[v_label[i].long()] = index_label[v_label[i].long()] + 1
        # if Y_label[i] == 0.:
        #     freedom_degree0 = torch.cat((freedom_degree0, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 1.:
        #     freedom_degree1 = torch.cat((freedom_degree1, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 2.:
        #     freedom_degree2 = torch.cat((freedom_degree2, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 3.:
        #     freedom_degree3 = torch.cat((freedom_degree3, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 4.:
        #     freedom_degree4 = torch.cat((freedom_degree4, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 5.:
        #     freedom_degree5 = torch.cat((freedom_degree5, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 6.:
        #     freedom_degree6 = torch.cat((freedom_degree6, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 7.:
        #     freedom_degree7 = torch.cat((freedom_degree7, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 8.:
        #     freedom_degree8 = torch.cat((freedom_degree8, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 9.:
        #     freedom_degree9 = torch.cat((freedom_degree9, diff_sqrt[i].float().view(1).cuda()), 0)
        # else:
        #     pass
        #     print('label is not in range(0~9)!!!')
    if epoch == 149:
        drawHistogram(freedom_degree100[0], 0)
        drawHistogram(freedom_degree100[9], 9)
        drawHistogram(freedom_degree100[19], 19)
        drawHistogram(freedom_degree100[29], 29)
        drawHistogram(freedom_degree100[39], 39)
        drawHistogram(freedom_degree100[49], 49)
        drawHistogram(freedom_degree100[59], 59)
        drawHistogram(freedom_degree100[69], 69)
        drawHistogram(freedom_degree100[79], 79)
        drawHistogram(freedom_degree100[89], 89)

    freedom_degree = torch.mean(freedom_degree100, dim=1)
    # freedom_degree = torch.Tensor([]).cuda()
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree0).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree1).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree2).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree3).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree4).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree5).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree6).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree7).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree8).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree9).float().view(1).cuda()), 0)
    for i in range(freedom_degree100.shape[0]):
        epoch_str = ("Test Average%d: %f, Covirance%d: %f, " %
                    (i, torch.mean(freedom_degree100[i]), i, torch.var(freedom_degree100[i], unbiased=False)))
        print(epoch_str)
    # epoch_str0 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree0), torch.var(freedom_degree0, unbiased=False)))
    # epoch_str1 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree1), torch.var(freedom_degree1, unbiased=False)))
    # epoch_str2 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree2), torch.var(freedom_degree2, unbiased=False)))
    # epoch_str3 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree3), torch.var(freedom_degree3, unbiased=False)))
    # epoch_str4 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree4), torch.var(freedom_degree4, unbiased=False)))
    # epoch_str5 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree5), torch.var(freedom_degree5, unbiased=False)))
    # epoch_str6 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree6), torch.var(freedom_degree6, unbiased=False)))
    # epoch_str7 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree7), torch.var(freedom_degree7, unbiased=False)))
    # epoch_str8 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree8), torch.var(freedom_degree8, unbiased=False)))
    # epoch_str9 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree9), torch.var(freedom_degree9, unbiased=False)))
    # print(epoch_str0)
    # print(epoch_str1)
    # print(epoch_str2)
    # print(epoch_str3)
    # print(epoch_str4)
    # print(epoch_str5)
    # print(epoch_str6)
    # print(epoch_str7)
    # print(epoch_str8)
    # print(epoch_str9)

    return freedom_degree

def Get100FreedomDegree(Y_label, Y_Feature, map_PEDCC, mapgT_PEDCC, epoch):
    # freedom_degree0 = torch.Tensor([]).cuda()
    # freedom_degree1 = torch.Tensor([]).cuda()
    # freedom_degree2 = torch.Tensor([]).cuda()
    # freedom_degree3 = torch.Tensor([]).cuda()
    # freedom_degree4 = torch.Tensor([]).cuda()
    # freedom_degree5 = torch.Tensor([]).cuda()
    # freedom_degree6 = torch.Tensor([]).cuda()
    # freedom_degree7 = torch.Tensor([]).cuda()
    # freedom_degree8 = torch.Tensor([]).cuda()
    # freedom_degree9 = torch.Tensor([]).cuda()
    freedom_degree100 = torch.zeros([100, 500], dtype=torch.float32).cuda()


    tensor_empty1 = map_PEDCC[Y_label.long()].float().cuda()
    label_mse_tensor = tensor_empty1.view(Y_label.shape[0], -1)  # (batchSize, dimension)
    label_mse_tensor = label_mse_tensor.cuda()

    Y_Feature = torch.mm(Y_Feature, mapgT_PEDCC)
    Y_Feature = Y_Feature[:, -99:]
    label_mse_tensor = torch.mm(label_mse_tensor, mapgT_PEDCC)
    label_mse_tensor = label_mse_tensor[:, -99:]

    diff_sqrt = GetDiffsqrt(Y_Feature, label_mse_tensor)
    index_label = torch.zeros([100]).cuda()
    for i in range(diff_sqrt.shape[0]):
        freedom_degree100[Y_label[i].long(), index_label[Y_label[i].long()].long()] = diff_sqrt[i].float().cuda()
        index_label[Y_label[i].long()] = index_label[Y_label[i].long()] + 1
        # if Y_label[i] == 0.:
        #     freedom_degree0 = torch.cat((freedom_degree0, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 1.:
        #     freedom_degree1 = torch.cat((freedom_degree1, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 2.:
        #     freedom_degree2 = torch.cat((freedom_degree2, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 3.:
        #     freedom_degree3 = torch.cat((freedom_degree3, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 4.:
        #     freedom_degree4 = torch.cat((freedom_degree4, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 5.:
        #     freedom_degree5 = torch.cat((freedom_degree5, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 6.:
        #     freedom_degree6 = torch.cat((freedom_degree6, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 7.:
        #     freedom_degree7 = torch.cat((freedom_degree7, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 8.:
        #     freedom_degree8 = torch.cat((freedom_degree8, diff_sqrt[i].float().view(1).cuda()), 0)
        # elif Y_label[i] == 9.:
        #     freedom_degree9 = torch.cat((freedom_degree9, diff_sqrt[i].float().view(1).cuda()), 0)
        # else:
        #     pass
        #     print('label is not in range(0~9)!!!')
    if epoch == 149:
        drawHistogram(freedom_degree100[0], 0)
        drawHistogram(freedom_degree100[9], 9)
        drawHistogram(freedom_degree100[19], 19)
        drawHistogram(freedom_degree100[29], 29)
        drawHistogram(freedom_degree100[39], 39)
        drawHistogram(freedom_degree100[49], 49)
        drawHistogram(freedom_degree100[59], 59)
        drawHistogram(freedom_degree100[69], 69)
        drawHistogram(freedom_degree100[79], 79)
        drawHistogram(freedom_degree100[89], 89)

    freedom_degree = torch.mean(freedom_degree100, dim=1)
    # freedom_degree = torch.Tensor([]).cuda()
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree0).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree1).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree2).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree3).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree4).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree5).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree6).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree7).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree8).float().view(1).cuda()), 0)
    # freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree9).float().view(1).cuda()), 0)
    # for i in range(freedom_degree100.shape[0]):
    #     epoch_str = ("Train Average%d: %f, Covirance%d: %f, " %
    #                 (i, torch.mean(freedom_degree100[i]), i, torch.var(freedom_degree100[i], unbiased=False)))
    #     print(epoch_str)
    # epoch_str0 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree0), torch.var(freedom_degree0, unbiased=False)))
    # epoch_str1 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree1), torch.var(freedom_degree1, unbiased=False)))
    # epoch_str2 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree2), torch.var(freedom_degree2, unbiased=False)))
    # epoch_str3 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree3), torch.var(freedom_degree3, unbiased=False)))
    # epoch_str4 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree4), torch.var(freedom_degree4, unbiased=False)))
    # epoch_str5 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree5), torch.var(freedom_degree5, unbiased=False)))
    # epoch_str6 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree6), torch.var(freedom_degree6, unbiased=False)))
    # epoch_str7 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree7), torch.var(freedom_degree7, unbiased=False)))
    # epoch_str8 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree8), torch.var(freedom_degree8, unbiased=False)))
    # epoch_str9 = ("Average0: %f, Covirance0: %f, " %
    #              (torch.mean(freedom_degree9), torch.var(freedom_degree9, unbiased=False)))
    # print(epoch_str0)
    # print(epoch_str1)
    # print(epoch_str2)
    # print(epoch_str3)
    # print(epoch_str4)
    # print(epoch_str5)
    # print(epoch_str6)
    # print(epoch_str7)
    # print(epoch_str8)
    # print(epoch_str9)

    return freedom_degree

def GetFreedomDegree(Y_label, Y_Feature, map_PEDCC, mapgT_PEDCC, epoch):
    freedom_degree0 = torch.Tensor([]).cuda()
    freedom_degree1 = torch.Tensor([]).cuda()
    freedom_degree2 = torch.Tensor([]).cuda()
    freedom_degree3 = torch.Tensor([]).cuda()
    freedom_degree4 = torch.Tensor([]).cuda()
    freedom_degree5 = torch.Tensor([]).cuda()
    freedom_degree6 = torch.Tensor([]).cuda()
    freedom_degree7 = torch.Tensor([]).cuda()
    freedom_degree8 = torch.Tensor([]).cuda()
    freedom_degree9 = torch.Tensor([]).cuda()


    tensor_empty1 = map_PEDCC[Y_label.long()].float().cuda()
    label_mse_tensor = tensor_empty1.view(Y_label.shape[0], -1)  # (batchSize, dimension)
    label_mse_tensor = label_mse_tensor.cuda()

    Y_Feature = torch.mm(Y_Feature, mapgT_PEDCC)
    Y_Feature = Y_Feature[:, -9:]
    label_mse_tensor = torch.mm(label_mse_tensor, mapgT_PEDCC)
    label_mse_tensor = label_mse_tensor[:, -9:]

    diff_sqrt = GetDiffsqrt(Y_Feature, label_mse_tensor)
    for i in range(diff_sqrt.shape[0]):
        if Y_label[i] == 0.:
            freedom_degree0 = torch.cat((freedom_degree0, diff_sqrt[i].float().view(1).cuda()), 0)
        elif Y_label[i] == 1.:
            freedom_degree1 = torch.cat((freedom_degree1, diff_sqrt[i].float().view(1).cuda()), 0)
        elif Y_label[i] == 2.:
            freedom_degree2 = torch.cat((freedom_degree2, diff_sqrt[i].float().view(1).cuda()), 0)
        elif Y_label[i] == 3.:
            freedom_degree3 = torch.cat((freedom_degree3, diff_sqrt[i].float().view(1).cuda()), 0)
        elif Y_label[i] == 4.:
            freedom_degree4 = torch.cat((freedom_degree4, diff_sqrt[i].float().view(1).cuda()), 0)
        elif Y_label[i] == 5.:
            freedom_degree5 = torch.cat((freedom_degree5, diff_sqrt[i].float().view(1).cuda()), 0)
        elif Y_label[i] == 6.:
            freedom_degree6 = torch.cat((freedom_degree6, diff_sqrt[i].float().view(1).cuda()), 0)
        elif Y_label[i] == 7.:
            freedom_degree7 = torch.cat((freedom_degree7, diff_sqrt[i].float().view(1).cuda()), 0)
        elif Y_label[i] == 8.:
            freedom_degree8 = torch.cat((freedom_degree8, diff_sqrt[i].float().view(1).cuda()), 0)
        elif Y_label[i] == 9.:
            freedom_degree9 = torch.cat((freedom_degree9, diff_sqrt[i].float().view(1).cuda()), 0)
        else:
            pass
            print('label is not in range(0~9)!!!')
    if epoch == 99:
        drawHistogram(freedom_degree0, 0)
        drawHistogram(freedom_degree1, 1)
        drawHistogram(freedom_degree2, 2)
        drawHistogram(freedom_degree3, 3)
        drawHistogram(freedom_degree4, 4)
        drawHistogram(freedom_degree5, 5)
        drawHistogram(freedom_degree6, 6)
        drawHistogram(freedom_degree7, 7)
        drawHistogram(freedom_degree8, 8)
        drawHistogram(freedom_degree9, 9)

    freedom_degree = torch.Tensor([]).cuda()
    freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree0).float().view(1).cuda()), 0)
    freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree1).float().view(1).cuda()), 0)
    freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree2).float().view(1).cuda()), 0)
    freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree3).float().view(1).cuda()), 0)
    freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree4).float().view(1).cuda()), 0)
    freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree5).float().view(1).cuda()), 0)
    freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree6).float().view(1).cuda()), 0)
    freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree7).float().view(1).cuda()), 0)
    freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree8).float().view(1).cuda()), 0)
    freedom_degree = torch.cat((freedom_degree, torch.mean(freedom_degree9).float().view(1).cuda()), 0)


    epoch_str0 = ("Average0: %f, Covirance0: %f, " %
                 (torch.mean(freedom_degree0), torch.var(freedom_degree0, unbiased=False)))
    epoch_str1 = ("Average0: %f, Covirance0: %f, " %
                 (torch.mean(freedom_degree1), torch.var(freedom_degree1, unbiased=False)))
    epoch_str2 = ("Average0: %f, Covirance0: %f, " %
                 (torch.mean(freedom_degree2), torch.var(freedom_degree2, unbiased=False)))
    epoch_str3 = ("Average0: %f, Covirance0: %f, " %
                 (torch.mean(freedom_degree3), torch.var(freedom_degree3, unbiased=False)))
    epoch_str4 = ("Average0: %f, Covirance0: %f, " %
                 (torch.mean(freedom_degree4), torch.var(freedom_degree4, unbiased=False)))
    epoch_str5 = ("Average0: %f, Covirance0: %f, " %
                 (torch.mean(freedom_degree5), torch.var(freedom_degree5, unbiased=False)))
    epoch_str6 = ("Average0: %f, Covirance0: %f, " %
                 (torch.mean(freedom_degree6), torch.var(freedom_degree6, unbiased=False)))
    epoch_str7 = ("Average0: %f, Covirance0: %f, " %
                 (torch.mean(freedom_degree7), torch.var(freedom_degree7, unbiased=False)))
    epoch_str8 = ("Average0: %f, Covirance0: %f, " %
                 (torch.mean(freedom_degree8), torch.var(freedom_degree8, unbiased=False)))
    epoch_str9 = ("Average0: %f, Covirance0: %f, " %
                 (torch.mean(freedom_degree9), torch.var(freedom_degree9, unbiased=False)))
    print(epoch_str0)
    print(epoch_str1)
    print(epoch_str2)
    print(epoch_str3)
    print(epoch_str4)
    print(epoch_str5)
    print(epoch_str6)
    print(epoch_str7)
    print(epoch_str8)
    print(epoch_str9)

    return freedom_degree

def Obtain_Covariance100(Y_label, Y_Feature, map_PEDCC):
    # covariance100 = torch.zeros([100, 99, 99], dtype=torch.float32).cuda()
    # covariance100 = torch.Tensor([]).cuda()
    average_feature = map_PEDCC[Y_label.long().data].float().cuda()
    Y_Feature = Y_Feature - average_feature
    covariance100 = 1 / (Y_Feature.shape[0] - 1) * torch.mm(Y_Feature.T, Y_Feature).float()


    # for i in range(100):
    #     each_feature = Y_Feature[Y_label.data == i]
    #     average_feature = torch.mean(each_feature, dim=0)
    #     each_feature = each_feature - average_feature
    #     each_covariance = 1 / (each_feature.shape[0] - 1) * torch.mm(each_feature.T, each_feature).float()
    #     covariance100 = torch.cat((covariance100, each_covariance.cuda()), 0)
    # print('covariance100:', covariance100.shape)
    # covariance100 = covariance100.view(100, each_covariance.shape[0], each_covariance.shape[1])
    # print('covariance10011:', covariance100.shape)
    return covariance100

def Obtain_AverageCovariance(Y_label, Y_Feature, mapgT_PEDCC):
    class0 = torch.Tensor([]).cuda()
    class1 = torch.Tensor([]).cuda()
    class2 = torch.Tensor([]).cuda()
    class3 = torch.Tensor([]).cuda()
    class4 = torch.Tensor([]).cuda()
    class5 = torch.Tensor([]).cuda()
    class6 = torch.Tensor([]).cuda()
    class7 = torch.Tensor([]).cuda()
    class8 = torch.Tensor([]).cuda()
    class9 = torch.Tensor([]).cuda()

    for i in range(Y_label.shape[0]):
        if Y_label[i] == 0.:
            class0 = torch.cat((class0, Y_Feature[i].float().cuda()), 0)
        elif Y_label[i] == 1.:
            class1 = torch.cat((class1, Y_Feature[i].float().cuda()), 0)
        elif Y_label[i] == 2.:
            class2 = torch.cat((class2, Y_Feature[i].float().cuda()), 0)
        elif Y_label[i] == 3.:
            class3 = torch.cat((class3, Y_Feature[i].float().cuda()), 0)
        elif Y_label[i] == 4.:
            class4 = torch.cat((class4, Y_Feature[i].float().cuda()), 0)
        elif Y_label[i] == 5.:
            class5 = torch.cat((class5, Y_Feature[i].float().cuda()), 0)
        elif Y_label[i] == 6.:
            class6 = torch.cat((class6, Y_Feature[i].float().cuda()), 0)
        elif Y_label[i] == 7.:
            class7 = torch.cat((class7, Y_Feature[i].float().cuda()), 0)
        elif Y_label[i] == 8.:
            class8 = torch.cat((class8, Y_Feature[i].float().cuda()), 0)
        elif Y_label[i] == 9.:
            class9 = torch.cat((class9, Y_Feature[i].float().cuda()), 0)
        else:
            pass
            print('label is not in range(0~9)!!!')
    # net = net.eval()
    # for im, label in tqdm(train_data):
    #     if torch.cuda.is_available():
    #         # label = label+80
    #         im = im.cuda()  # (bs, 3, h, w)
    #         # label = label.cuda()  # (bs, h, w)
    #         output2 = net(im)[1]
    #         for i in range(label.shape[0]):
    #             if label[i] == 0:
    #                 class0 = torch.cat((class0, output2[i].float().cuda()), 0)
    #             elif label[i] == 1:
    #                 class1 = torch.cat((class1, output2[i].float().cuda()), 0)
    #             elif label[i] == 2:
    #                 class2 = torch.cat((class2, output2[i].float().cuda()), 0)
    #             elif label[i] == 3:
    #                 class3 = torch.cat((class3, output2[i].float().cuda()), 0)
    #             elif label[i] == 4:
    #                 class4 = torch.cat((class4, output2[i].float().cuda()), 0)
    #             elif label[i] == 5:
    #                 class5 = torch.cat((class5, output2[i].float().cuda()), 0)
    #             elif label[i] == 6:
    #                 class6 = torch.cat((class6, output2[i].float().cuda()), 0)
    #             elif label[i] == 7:
    #                 class7 = torch.cat((class7, output2[i].float().cuda()), 0)
    #             elif label[i] == 8:
    #                 class8 = torch.cat((class8, output2[i].float().cuda()), 0)
    #             elif label[i] == 9:
    #                 class9 = torch.cat((class9, output2[i].float().cuda()), 0)
    #             else:
    #                 pass
    #                 print('label is not in range(0~9)!!!')

    class0 = torch.mm(class0.view(-1, Y_Feature.shape[1]), mapgT_PEDCC)
    class1 = torch.mm(class1.view(-1, Y_Feature.shape[1]), mapgT_PEDCC)
    class2 = torch.mm(class2.view(-1, Y_Feature.shape[1]), mapgT_PEDCC)
    class3 = torch.mm(class3.view(-1, Y_Feature.shape[1]), mapgT_PEDCC)
    class4 = torch.mm(class4.view(-1, Y_Feature.shape[1]), mapgT_PEDCC)
    class5 = torch.mm(class5.view(-1, Y_Feature.shape[1]), mapgT_PEDCC)
    class6 = torch.mm(class6.view(-1, Y_Feature.shape[1]), mapgT_PEDCC)
    class7 = torch.mm(class7.view(-1, Y_Feature.shape[1]), mapgT_PEDCC)
    class8 = torch.mm(class8.view(-1, Y_Feature.shape[1]), mapgT_PEDCC)
    class9 = torch.mm(class9.view(-1, Y_Feature.shape[1]), mapgT_PEDCC)

    class0 = class0[:, -9:]
    class1 = class1[:, -9:]
    class2 = class2[:, -9:]
    class3 = class3[:, -9:]
    class4 = class4[:, -9:]
    class5 = class5[:, -9:]
    class6 = class6[:, -9:]
    class7 = class7[:, -9:]
    class8 = class8[:, -9:]
    class9 = class9[:, -9:]


    AverageOfClass = torch.Tensor([]).cuda()
    CovarianceOfClass = torch.Tensor([]).cuda()

    AverageOfClass = torch.cat((AverageOfClass, torch.mean(class0, 0).float().cuda()), 0)
    AverageOfClass = torch.cat((AverageOfClass, torch.mean(class1, 0).float().cuda()), 0)
    AverageOfClass = torch.cat((AverageOfClass, torch.mean(class2, 0).float().cuda()), 0)
    AverageOfClass = torch.cat((AverageOfClass, torch.mean(class3, 0).float().cuda()), 0)
    AverageOfClass = torch.cat((AverageOfClass, torch.mean(class4, 0).float().cuda()), 0)
    AverageOfClass = torch.cat((AverageOfClass, torch.mean(class5, 0).float().cuda()), 0)
    AverageOfClass = torch.cat((AverageOfClass, torch.mean(class6, 0).float().cuda()), 0)
    AverageOfClass = torch.cat((AverageOfClass, torch.mean(class7, 0).float().cuda()), 0)
    AverageOfClass = torch.cat((AverageOfClass, torch.mean(class8, 0).float().cuda()), 0)
    AverageOfClass = torch.cat((AverageOfClass, torch.mean(class9, 0).float().cuda()), 0)
    AverageOfClass = AverageOfClass.view(10, -1)

    class0 = class0 - AverageOfClass[0]
    class1 = class1 - AverageOfClass[1]
    class2 = class2 - AverageOfClass[2]
    class3 = class3 - AverageOfClass[3]
    class4 = class4 - AverageOfClass[4]
    class5 = class5 - AverageOfClass[5]
    class6 = class6 - AverageOfClass[6]
    class7 = class7 - AverageOfClass[7]
    class8 = class8 - AverageOfClass[8]
    class9 = class9 - AverageOfClass[9]

    class0 = 1 / (class0.shape[0] - 1) * torch.mm(class0.T, class0).float()
    class1 = 1 / (class1.shape[0] - 1) * torch.mm(class1.T, class1).float()
    class2 = 1 / (class2.shape[0] - 1) * torch.mm(class2.T, class2).float()
    class3 = 1 / (class3.shape[0] - 1) * torch.mm(class3.T, class3).float()
    class4 = 1 / (class4.shape[0] - 1) * torch.mm(class4.T, class4).float()
    class5 = 1 / (class5.shape[0] - 1) * torch.mm(class5.T, class5).float()
    class6 = 1 / (class6.shape[0] - 1) * torch.mm(class6.T, class6).float()
    class7 = 1 / (class7.shape[0] - 1) * torch.mm(class7.T, class7).float()
    class8 = 1 / (class8.shape[0] - 1) * torch.mm(class8.T, class8).float()
    class9 = 1 / (class9.shape[0] - 1) * torch.mm(class9.T, class9).float()


    CovarianceOfClass = torch.cat((CovarianceOfClass, class0.cuda()), 0)
    CovarianceOfClass = torch.cat((CovarianceOfClass, class1.cuda()), 0)
    CovarianceOfClass = torch.cat((CovarianceOfClass, class2.cuda()), 0)
    CovarianceOfClass = torch.cat((CovarianceOfClass, class3.cuda()), 0)
    CovarianceOfClass = torch.cat((CovarianceOfClass, class4.cuda()), 0)
    CovarianceOfClass = torch.cat((CovarianceOfClass, class5.cuda()), 0)
    CovarianceOfClass = torch.cat((CovarianceOfClass, class6.cuda()), 0)
    CovarianceOfClass = torch.cat((CovarianceOfClass, class7.cuda()), 0)
    CovarianceOfClass = torch.cat((CovarianceOfClass, class8.cuda()), 0)
    CovarianceOfClass = torch.cat((CovarianceOfClass, class9.cuda()), 0)
    CovarianceOfClass = CovarianceOfClass.view(10, 9, -1)

    return AverageOfClass, CovarianceOfClass

def Predict_Test(output2, AverageOfClass, CovarianceOfClass):
    # Denterminant_class = torch.Tensor([]).cuda()
    # Inverse_class = torch.Tensor([]).cuda()
    # # print(torch.det(CovarianceOfClass[0]).view(1))
    #
    # Denterminant_class = torch.cat((Denterminant_class, torch.det(CovarianceOfClass[0].float()).view(1).cuda()), 0)
    # Denterminant_class = torch.cat((Denterminant_class, torch.det(CovarianceOfClass[1].float()).view(1).cuda()), 0)
    # Denterminant_class = torch.cat((Denterminant_class, torch.det(CovarianceOfClass[2].float()).view(1).cuda()), 0)
    # Denterminant_class = torch.cat((Denterminant_class, torch.det(CovarianceOfClass[3].float()).view(1).cuda()), 0)
    # Denterminant_class = torch.cat((Denterminant_class, torch.det(CovarianceOfClass[4].float()).view(1).cuda()), 0)
    # Denterminant_class = torch.cat((Denterminant_class, torch.det(CovarianceOfClass[5].float()).view(1).cuda()), 0)
    # Denterminant_class = torch.cat((Denterminant_class, torch.det(CovarianceOfClass[6].float()).view(1).cuda()), 0)
    # Denterminant_class = torch.cat((Denterminant_class, torch.det(CovarianceOfClass[7].float()).view(1).cuda()), 0)
    # Denterminant_class = torch.cat((Denterminant_class, torch.det(CovarianceOfClass[8].float()).view(1).cuda()), 0)
    # Denterminant_class = torch.cat((Denterminant_class, torch.det(CovarianceOfClass[9].float()).view(1).cuda()), 0)
    # # print(Denterminant_class)
    #
    # Inverse_class = torch.cat((Inverse_class, torch.inverse(CovarianceOfClass[0]).cuda()), 0)
    # Inverse_class = torch.cat((Inverse_class, torch.inverse(CovarianceOfClass[1]).cuda()), 0)
    # Inverse_class = torch.cat((Inverse_class, torch.inverse(CovarianceOfClass[2]).cuda()), 0)
    # Inverse_class = torch.cat((Inverse_class, torch.inverse(CovarianceOfClass[3]).cuda()), 0)
    # Inverse_class = torch.cat((Inverse_class, torch.inverse(CovarianceOfClass[4]).cuda()), 0)
    # Inverse_class = torch.cat((Inverse_class, torch.inverse(CovarianceOfClass[5]).cuda()), 0)
    # Inverse_class = torch.cat((Inverse_class, torch.inverse(CovarianceOfClass[6]).cuda()), 0)
    # Inverse_class = torch.cat((Inverse_class, torch.inverse(CovarianceOfClass[7]).cuda()), 0)
    # Inverse_class = torch.cat((Inverse_class, torch.inverse(CovarianceOfClass[8]).cuda()), 0)
    # Inverse_class = torch.cat((Inverse_class, torch.inverse(CovarianceOfClass[9]).cuda()), 0)
    # Inverse_class = Inverse_class.view(10, 9, -1)
    #
    # output = torch.zeros(output2.shape[0], output2.shape[1] + 1).cuda()
    # # print((output2[1] - AverageOfClass[1]).view(1,-1).shape)
    # # print(Inverse_class[1].shape)
    # for i in range(output.shape[0]):
    #     for j in range(output.shape[1]):
    #         output[i][j] = 1/(pow(pow((2*math.pi), output2.shape[1])*Denterminant_class[j], 0.5)) * math.exp(-0.5*torch.mm(torch.mm((output2[i] - AverageOfClass[j]).T.view(1,-1), Inverse_class[j]), (output2[i] - AverageOfClass[j]).view(9,-1)))
    output = []
    for i in range(output2.shape[1] + 1):
        rate = multivariate_normal.pdf(output2.cpu().detach().numpy(), AverageOfClass[i].cpu().detach().numpy(), CovarianceOfClass[i].cpu().detach().numpy())
        # rate = np.array(rate).astype(float)
        # print('len(rate):', len(rate))
        output.append(rate)
        # print('output.shape:', len(output), len(output[0]))
        # rate = torch.from_numpy(rate)
        # output = torch.cat((output, rate.float().cuda()), 0)
    # print('len(output):', len(output))
    output = np.array(output).astype(float)
    # print('output.shape:', output.shape)
    output.reshape((10, output2.shape[0]))
    # print('output.shape:', output.shape)
    output = output.T
    # print(output.shape)
    return output

def kafang100_predict(output2, map_PEDCC, mapT_PEDCC, Freedom_Degree):
    diff_sqrt = torch.Tensor([]).cuda()
    map_PEDCC = map_PEDCC.cuda()
    output2 = torch.mm(output2, mapT_PEDCC)
    output2 = output2[:, -99:]
    # print("output2: ", output2.shape)
    map_PEDCC = torch.mm(map_PEDCC, mapT_PEDCC)
    # print("map_PEDCC: ", map_PEDCC.shape)
    map_PEDCC = map_PEDCC[:, -99:]
    # print("map_PEDCC: ", map_PEDCC.shape)
    for i in range(map_PEDCC.shape[0]):
        res = (output2 - map_PEDCC[i]) ** 2
        # print("res: ", res.shape)
        res = torch.sum(res, dim=1)
        # print("res: ", res.shape)
        res = res * 99 / Freedom_Degree[i]
        # print("res: ", res.shape)
        # res = res ** 2
        diff_sqrt = torch.cat((diff_sqrt, res.float().cuda()), 0)
    # print("diff_sqrt: ", diff_sqrt.shape)
    diff_sqrt = diff_sqrt.view(map_PEDCC.shape[0], -1)
    # diff_sqrt = diff_sqrt.T
    # print("diff_sqrt: ", diff_sqrt.shape)
    # print("Freedom_Degree[i]: ", Freedom_Degree[0].shape)
    # print("Freedom_Degree: ", Freedom_Degree.shape)
    result = []
    for i in range(Freedom_Degree.shape[0]):
        # print('diff_sqrt[i]: ', diff_sqrt[i])
        # print('diff_sqrt[i]_cpu(): ', diff_sqrt[i].cpu().detach().numpy())
        # print('Freedom_Degree[i]: ', Freedom_Degree[i])
        # print('Freedom_Degree[i]_cpu(): ', Freedom_Degree[i].cpu().detach().numpy())
        rate = chi2.pdf(diff_sqrt[i].cpu().detach().numpy(), 99)
        # print('rate: ', rate)
        # print('-1:', len(rate))
        result.append(rate)
    # print('0:', len(result), len(result[0]))
    result = np.array(result).astype(float)
    # print('1:', result.shape)
    result.reshape((Freedom_Degree.shape[0], -1))
    # print('2:', result.shape)
    result = result.T
    # print('3:', result.shape)
    return result

def kafang_predict(output2, map_PEDCC, mapT_PEDCC, Freedom_Degree):
    diff_sqrt = torch.Tensor([]).cuda()
    map_PEDCC = map_PEDCC.cuda()
    # output2 = torch.mm(output2, mapT_PEDCC)
    # output2 = output2[:, -9:]
    # # print("output2: ", output2.shape)
    # map_PEDCC = torch.mm(map_PEDCC, mapT_PEDCC)
    # # print("map_PEDCC: ", map_PEDCC.shape)
    # map_PEDCC = map_PEDCC[:, -9:]
    # print("map_PEDCC: ", map_PEDCC.shape)
    for i in range(map_PEDCC.shape[0]):
        res = (output2 - map_PEDCC[i]) ** 2
        # print("res: ", res.shape)
        res = torch.sum(res, dim=1)
        # print("res: ", res.shape)
        res = res * 9 / Freedom_Degree[i]
        # print("res: ", res.shape)
        # res = res ** 2
        diff_sqrt = torch.cat((diff_sqrt, res.float().cuda()), 0)
    # print("diff_sqrt: ", diff_sqrt.shape)
    diff_sqrt = diff_sqrt.view(map_PEDCC.shape[0], -1)
    # diff_sqrt = diff_sqrt.T
    # print("diff_sqrt: ", diff_sqrt.shape)
    # print("Freedom_Degree[i]: ", Freedom_Degree[0].shape)
    # print("Freedom_Degree: ", Freedom_Degree.shape)
    result = []
    for i in range(Freedom_Degree.shape[0]):
        # print('diff_sqrt[i]: ', diff_sqrt[i])
        # print('diff_sqrt[i]_cpu(): ', diff_sqrt[i].cpu().detach().numpy())
        # print('Freedom_Degree[i]: ', Freedom_Degree[i])
        # print('Freedom_Degree[i]_cpu(): ', Freedom_Degree[i].cpu().detach().numpy())
        rate = chi2.pdf(diff_sqrt[i].cpu().detach().numpy(), 9)
        # print('rate: ', rate)
        # print('-1:', len(rate))
        result.append(rate)
    # print('0:', len(result), len(result[0]))
    result = np.array(result).astype(float)
    # print('1:', result.shape)
    result.reshape((Freedom_Degree.shape[0], -1))
    # print('2:', result.shape)
    result = result.T
    # print('3:', result.shape)
    return result

    # ret = input * target
    # ret = torch.sum(ret, dim=1)
    ret = 1 - ret
    ret = ret.pow(2)
    ret = torch.mean(ret)
# def preceding_stage_loss(input, dim):
#     input_norm = l2_norm(input, axis=dim)
#     # print(input_norm.shape)
#     # print(input_norm)
#     input_norm_max = torch.max(input_norm, dim=1)[0]
#     # print(input_norm_max.shape)
#     # print(input_norm_max)
#     ret = (1 - input_norm_max)
#     # print(ret.shape)
#     # print(ret)
#     ret1 = torch.mean(ret)
#     # print(ret1)
#     return ret1
def preceding_stage_loss(input, dim):
    input_flatten = input.view(input.size(0), -1)
    input_avg = torch.mean(input_flatten, keepdim=True, dim=dim)
    input_std = torch.std(input_flatten, keepdim=True, dim=dim).clamp(min=1e-12)
    # input_pre = (input_flatten - input_avg) / input_std
    input_pre = (input_flatten - 2 * input_avg) / input_std
    max_avg = input_flatten[input_flatten > input_avg]
    min_avg = input_flatten[input_flatten <= input_avg]
    ret = 0
    for i in range(input_pre.size(0)):
        input_maxzero = input_pre[i][input_pre[i] > 0]
        input_minzero = input_pre[i][input_pre[i] <= 0]
        # var_maxzero = torch.mean((input_maxzero - torch.median(input_maxzero)).pow(2))
        var_maxzero = torch.var(input_maxzero)
        # var_minzero = torch.var(input_minzero + input_avg[i][0])
        # var_minzero = torch.mean((input_minzero + input_avg[i][0]).pow(2))
        var_minzero = torch.mean((input_minzero + 2 * input_avg[i][0]).pow(2))
        ret += 10 * var_maxzero + var_minzero
    return ret / input_pre.size(0), input_avg, max_avg.size(0), min_avg.size(0)


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    # print(total0.shape)
    # print(total1.shape)
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # L2_distance = ((total0 - total1) ** 4).sum(2) / 4
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=1.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    # batch_size_t = int(target.size()[0])
    source = l2_norm(source)
    target = l2_norm(target)
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = torch.mean(kernels[:batch_size, :batch_size])
    YY = torch.mean(kernels[batch_size:, batch_size:])
    XY = torch.mean(kernels[:batch_size, batch_size:])
    YX = torch.mean(kernels[batch_size:, :batch_size])
    loss = XX + YY - XY -YX
    # a=torch.mean(XX)
    # b=torch.mean(XY)
    # c=torch.mean(YX)
    # d=torch.mean(YY)
    # loss = torch.mean(XX) - 2 * torch.mean(XY) + torch.mean(YY)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


def weightReduceLoss(input, scale):
    output_norm = torch.norm(input, p=2, dim=1)
    output = torch.sum(output_norm)

    return scale*output

def sortDivideLoss(input, scale):
    input_norm = l2_norm(input)
    input_res = torch.max(input_norm, dim=1)[0]
    # print(input_res.shape)
    tendToOne = torch.mean((1 - input_res).pow(2))
    return scale * tendToOne


def reduceVarAndAvg(input, scale):
    # input = input.permute(0, 2, 3, 1)
    # input_flatten = input.reshape(-1, input.size(3))
    input_MoValue = torch.norm(input, p=2, dim=1)
    # norm = torch.norm(input, 2, axis, True).clamp(min=1e-12)
    input_ret = input_MoValue.view(-1)
    var = torch.var(input_ret)
    avg = torch.mean(input_ret)

    return scale * (var + avg)

def NewDecorrelationBetweenDim1to5(input2, scale):
    input2 = input2.view(input2.size(0), input2.size(1), -1)
    input2 = input2.permute(1, 0, 2)
    input_avg = torch.mean(input2, dim=1, keepdim=True)
    # input_avg = torch.mean(input2, dim=[0, 1], keepdim=True)
    input_subAvg = input2 - input_avg
    input_subAvg1 = input_subAvg.reshape(input_subAvg.size(0), -1)
    newTensor = torch.mm(input_subAvg1, input_subAvg1.T) / input_subAvg.size(2)
    input_sum = torch.sum(input_subAvg1.pow(2), dim=1, keepdim=True)
    input_sum = pow(input_sum / input_avg.size(2), 0.5)
    # input_sum = input_sum.view(input_sum.size(0), -1)
    ret_sum = torch.mm(input_sum, input_sum.T).clamp(min=5e-12)
    res = newTensor.clamp(min=1e-12) / ret_sum

    res_loss = torch.sum(abs(res)) - torch.sum(abs(res.diagonal()))
    # res_loss = res_loss / ((res.size(0) - 1) * res.size(0))
    res_loss = res_loss / max((res.size(0) - 1) * res.size(0), 1e-12)

    # print(torch.unique(res.diagonal().view(-1).type(torch.), return_counts=True))

    return scale * res_loss


def NewDecorrelationBetweenDim1to2(input2, scale):
    input2 = input2.view(input2.size(0), input2.size(1), -1)
    input2 = input2.permute(1, 0, 2)
    input_avg = torch.mean(input2, dim=1, keepdim=True)
    # input_avg = torch.mean(input2, dim=[0, 1], keepdim=True)
    input_subAvg = input2 - input_avg
    input_subAvg1 = input_subAvg.reshape(input_subAvg.size(0), -1)
    newTensor = torch.mm(input_subAvg1, input_subAvg1.T) / input_subAvg.size(2)
    input_sum = torch.sum(input_subAvg1.pow(2), dim=1, keepdim=True)
    input_sum = pow(input_sum / input_avg.size(2), 0.5)
    # input_sum = input_sum.view(input_sum.size(0), -1)
    ret_sum = torch.mm(input_sum, input_sum.T).clamp(min=2e-12)
    res = newTensor.clamp(min=1e-12) / ret_sum

    res_loss = torch.sum(abs(res)) - torch.sum(abs(res.diagonal()))
    # res_loss = res_loss / ((res.size(0) - 1) * res.size(0))
    res_loss = res_loss / max((res.size(0) - 1) * res.size(0), 1e-12)

    # print(torch.unique(res.diagonal().view(-1).type(torch.), return_counts=True))

    return scale * res_loss


def NewDecorrelationBetweenDim1(input2, scale):
    input2 = input2.view(input2.size(0), input2.size(1), -1)
    input2 = input2.permute(1, 0, 2)
    input_avg = torch.mean(input2, dim=1, keepdim=True)#old
    # input_avg = torch.mean(input2)
    # input_avg = torch.mean(input2, dim=[0, 1], keepdim=True)
    # input_subAvg = input2
    input_subAvg = input2 - input_avg
    input_subAvg1 = input_subAvg.reshape(input_subAvg.size(0), -1)
    newTensor = torch.mm(input_subAvg1, input_subAvg1.T) / input_subAvg.size(2)
    input_sum = torch.sum(input_subAvg1.pow(2), dim=1, keepdim=True)
    input_sum = pow(input_sum / input_avg.size(2), 0.5)#old
    # input_sum = pow(input_sum / input2.size(2), 0.5)
    # input_sum = input_sum.view(input_sum.size(0), -1)
    ret_sum = torch.mm(input_sum, input_sum.T).clamp(min=1e-8)
    res = newTensor / ret_sum

    # res_loss = torch.sum(abs(res)) - torch.sum(abs(res.diagonal()))
    res_loss = torch.sum(pow(res, 2)) - torch.sum(pow(res.diagonal(), 2))
    # res_loss = res_loss / ((res.size(0) - 1) * res.size(0))
    res_loss = res_loss / max((res.size(0) - 1) * res.size(0), 1e-8)

    # print(torch.unique(res.diagonal().view(-1).type(torch.), return_counts=True))

    return scale * res_loss


def NewDecorrelationBetweenDim2(input2, scale):
    input2 = input2.view(input2.size(0), input2.size(1), -1)
    input2 = input2.permute(1, 0, 2)
    input_avg = torch.mean(input2, dim=0, keepdim=True)
    # input_avg = torch.mean(input2, dim=[0, 1], keepdim=True)
    input_subAvg = input2 - input_avg
    input_subAvg1 = input_subAvg.reshape(input_subAvg.size(0), -1)
    newTensor = torch.mm(input_subAvg1, input_subAvg1.T) / input_subAvg.size(2)
    input_sum = torch.sum(input_subAvg1.pow(2), dim=1, keepdim=True)
    input_sum = pow(input_sum / input_avg.size(2), 0.5)
    # input_sum = input_sum.view(input_sum.size(0), -1)
    ret_sum = torch.mm(input_sum, input_sum.T).clamp(min=1e-12)
    res = newTensor / ret_sum

    res_loss = torch.sum(abs(res)) - torch.sum(abs(res.diagonal()))
    # res_loss = res_loss / ((res.size(0) - 1) * res.size(0))
    res_loss = res_loss / max((res.size(0) - 1) * res.size(0), 1e-12)

    # print(torch.unique(res.diagonal().view(-1).type(torch.), return_counts=True))

    return scale * res_loss


def NewDecorrelationBetweenDim3(input2, scale):
    input2 = input2.view(input2.size(0), input2.size(1), -1)
    input2 = input2.permute(1, 0, 2)
    input_avg = torch.mean(input2, dim=1, keepdim=True)
    # input_avg = torch.mean(input2, dim=[0, 1], keepdim=True)
    input_subAvg = input2 - input_avg
    input_subAvg1 = input_subAvg.reshape(input_subAvg.size(0), -1)
    newTensor = torch.mm(input_subAvg1, input_subAvg1.T) / input_subAvg.size(2)
    input_sum = torch.sum(input_subAvg1.pow(2), dim=1, keepdim=True)
    input_sum = pow(input_sum / input_avg.size(2), 0.5)
    # input_sum = input_sum.view(input_sum.size(0), -1)
    ret_sum = torch.mm(input_sum, input_sum.T).clamp(min=1e-12)
    res = newTensor.clamp(min=1e-12) / ret_sum

    res_loss = torch.sum(abs(res)) - torch.sum(abs(res.diagonal()))
    # res_loss = res_loss / ((res.size(0) - 1) * res.size(0))
    res_loss = res_loss / max((res.size(0) - 1) * res.size(0), 1e-12)

    # print(torch.unique(res.diagonal().view(-1).type(torch.), return_counts=True))

    return scale * res_loss

def DecorrelationBetweenDim1(input, scale):
    input1 = torch.mean(input, dim=2)
    input2 = torch.mean(input1, dim=2)
    input2 = input2.permute(1, 0)
    input_avg = torch.mean(input2, dim=1, keepdim=True)
    input_subAvg = input2 - input_avg
    ret = torch.mm(input_subAvg, input_subAvg.T)
    input_sum = torch.sum(input_subAvg.pow(2), dim=1, keepdim=True).pow(0.5)
    ret_sum = torch.mm(input_sum, input_sum.T).clamp(min=1e-12)
    res = ret / ret_sum
    # res_loss = torch.sum(pow(res, 2)) - torch.sum(pow(torch.diagonal(res), 2))
    # threshold = res.size(0) * 1 // 5
    # res_new = res[0:threshold, 0:threshold]
    # res_loss = torch.sum(abs(res_new)) - res_new.size(0)
    # res_loss = res_loss / ((res_new.size(0) - 1) * res_new.size(0))

    res_loss = torch.sum(abs(res)) - res.size(0)
    res_loss = res_loss / ((res.size(0) - 1) * res.size(0))

    return scale * res_loss


def DecorrelationBetweenDim2(input2, scale):
    # input1 = torch.mean(input, dim=2)
    # input2 = torch.mean(input1, dim=2)
    input2 = input2.permute(1, 0)
    input_avg = torch.mean(input2, dim=1, keepdim=True)
    input_subAvg = input2 - input_avg
    ret = torch.mm(input_subAvg, input_subAvg.T)
    input_sum = torch.sum(input_subAvg.pow(2), dim=1, keepdim=True).pow(0.5)
    ret_sum = torch.mm(input_sum, input_sum.T).clamp(min=1e-12)
    res = ret / ret_sum
    # res_loss = torch.sum(pow(res, 2)) - torch.sum(pow(torch.diagonal(res), 2))
    # threshold = res.size(0) * 1 // 5
    # res_new = res[0:threshold, 0:threshold]
    # res_loss = torch.sum(abs(res_new)) - res_new.size(0)
    # res_loss = res_loss / ((res_new.size(0) - 1) * res_new.size(0))

    res_loss = torch.sum(abs(res)) - res.size(0)
    res_loss = res_loss / ((res.size(0) - 1) * res.size(0))

    return scale * res_loss


def tendToPEDCC(input, map_PEDCC, scale):
    input_norm = l2_norm(input)
    input_norm = input_norm.permute(0, 2, 3, 1)
    input_flatten = input_norm.reshape(-1, input_norm.size(3))
    output = torch.mm(input_flatten, map_PEDCC.T)
    output_res = torch.max(output, dim=1)[0]
    tendToOne = torch.mean((1 - output_res).pow(2))
    return scale * tendToOne


def FeatureToOneOrZero(input, scale):
    input = input.permute(0, 2, 3, 1)
    input_flatten = input.reshape(-1, input.size(3))
    input_flatten_norm = l2_norm(input_flatten)
    ret = torch.mm(input_flatten_norm, input_flatten_norm.T)
    ret_1_abs = torch.abs(ret - 1)
    ret_abs = torch.abs(ret)

    res1 = ret_1_abs[ret_1_abs <= ret_abs]
    res2 = ret_abs[ret_abs < ret_1_abs]
    loss1 = torch.mean(res1)
    loss2 = torch.mean(res2)
    return scale * (loss1 + loss2) / 2


def DecorrelationBetweenDim(input, scale):
    # print(input.shape)
    input = input.permute(1, 0, 2, 3)
    # print(input.shape)
    input_flatten = input.reshape(input.size(0), -1)
    # print(input_flatten.shape)
    input_avg = torch.mean(input_flatten, dim=0)
    input_subAvg = input_flatten - input_avg
    ret = 1 / (input_subAvg.size(1) - 1) * torch.mm(input_subAvg, input_subAvg.T)
    # print(ret.shape)
    ret_loss = torch.sum(pow(ret, 2)) - torch.sum(pow(torch.diagonal(ret), 2))
    ret_loss = ret_loss / ((ret.size(0) - 1) * ret.size(0))
    # print(ret_loss)
    return scale * ret_loss

# def sortDivideLoss(input, scale):
#     input_mo = torch.norm(input=input, p=2, dim=1)
#
#     input_mo_sort = torch.sort(input_mo)[0]
#     idx = input_mo_sort.size(0)*1//100
#     threshold = input_mo_sort[idx]
#
#     input_min = input[input_mo < threshold]
#     tendToZero = torch.mean(input_min.pow(2))
#
#     input_max = input[input_mo >= threshold]
#     input_max_norm = l2_norm(input_max)
#     input_max_res = torch.max(input_max_norm, dim=1)[0]
#     tendToOne = torch.mean((1 - input_max_res).pow(2))
#
#     return scale * (tendToZero + tendToOne)


# def sortDivideLoss(input, maxTarget, scale):
#     output = torch.sort(input)[0]
#     threshold = output.size(0)*9//10
#     tendToZero = torch.mean(output[:threshold].pow(2))
#     tendToOne = torch.mean((output[threshold:] - maxTarget).pow(2))
#     # mean = torch.mean(output[threshold:]).data
#     # tendToOne = torch.var(output[threshold:])
#     return scale * (tendToZero + tendToOne)

def averageDivideLoss(input):
    average = torch.mean(input)
    lessThanAverage = input[input < average]
    tendToZero = torch.mean(lessThanAverage.pow(3))

    return 10 * tendToZero


def train_soft_mse_zl(net, train_data, valid_data, cfg, criterion, criterion1, criterion2, save_folder, save_txt, classes_num):
    LR = cfg['LR']
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    prev_time = datetime.now()
    # map_dict = read_pkl()
    # map_dict256 = read_pklFromFile('center_pedcc/GausiNew/256_256_s.pkl')
    # map_dict512 = read_pklFromFile('center_pedcc/GausiNew/512_512_s.pkl')
    # map_dict1024 = read_pklFromFile('center_pedcc/GausiNew/1024_1024_s.pkl')
    # map_dict2048 = read_pklFromFile('center_pedcc/GausiNew/2048_2048_s.pkl')
    # map_dict1280 = read_pklFromFile('center_pedcc/GausiNew/1280_1280_s.pkl')
    # gmap_T_dict = gT_read_pkl()
    history = dict()
    loss_train = []
    loss1_train = []
    loss2_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    ratio_train = []
    ratio_val = []
    # gausi_acc_val = []
    #
    # map_PEDCC = torch.Tensor([])
    # for i in range(classes_num):
    #     map_PEDCC = torch.cat((map_PEDCC, map_dict[i].float()), 0)
    # map_PEDCC = map_PEDCC.view(classes_num, -1)  # (class_num, dimension)

    # map_PEDCC256 = torch.Tensor([])
    # for i in range(256):
    #     map_PEDCC256 = torch.cat((map_PEDCC256, map_dict256[i].float()), 0)
    # map_PEDCC256 = map_PEDCC256.view(256, -1).cuda()  # (class_num, dimension)
    #
    # map_PEDCC512 = torch.Tensor([])
    # for i in range(512):
    #     map_PEDCC512 = torch.cat((map_PEDCC512, map_dict512[i].float()), 0)
    # map_PEDCC512 = map_PEDCC512.view(512, -1).cuda()  # (class_num, dimension)
    #
    # map_PEDCC1024 = torch.Tensor([])
    # for i in range(1024):
    #     map_PEDCC1024 = torch.cat((map_PEDCC1024, map_dict1024[i].float()), 0)
    # map_PEDCC1024 = map_PEDCC1024.view(1024, -1).cuda()  # (class_num, dimension)
    #
    # map_PEDCC2048 = torch.Tensor([])
    # for i in range(2048):
    #     map_PEDCC2048 = torch.cat((map_PEDCC2048, map_dict2048[i].float()), 0)
    # map_PEDCC2048 = map_PEDCC2048.view(2048, -1).cuda()  # (class_num, dimension)
    #
    # map_PEDCC1280 = torch.Tensor([])
    # for i in range(1280):
    #     map_PEDCC1280 = torch.cat((map_PEDCC1280, map_dict1280[i].float()), 0)
    # map_PEDCC1280 = map_PEDCC1280.view(1280, -1).cuda()  # (class_num, dimension)

    # Projection_matrix = torch.mm(map_PEDCC.T, map_PEDCC)
    # Projection_matrix = Projection_matrix.inverse()
    # Projection_matrix = torch.mm(map_PEDCC, Projection_matrix)
    # Projection_matrix = torch.mm(Projection_matrix, map_PEDCC.T)
    # distribution_zero_one = torch.zeros([48, 2048*4*4]).cuda()
    # distribution_zero_one[:, 2048*4*4*4//5:] = 1

    # distribution_zero_one = torch.Tensor([[0], [0], [0], [0], [1]]).cuda()
    # distribution_zero_one = torch.zeros([2048, 2048]).cuda()
    # for i in range(distribution_zero_one.size(0)):
    #     distribution_zero_one[i, i] = 1
    # mapgT_PEDCC = torch.Tensor([])
    # for i in range(map_PEDCC.shape[1]):
    #     mapgT_PEDCC = torch.cat((mapgT_PEDCC, gmap_T_dict[i].float()), 0)
    # mapgT_PEDCC = mapgT_PEDCC.view(map_PEDCC.shape[1], -1).cuda()  # (dimension, dimension)

    # criterion0 = CConvarianceLoss(map_PEDCC)
    # Y_Feature = torch.Tensor([]).cuda()
    # Y_label = torch.Tensor([]).cuda()
    #
    # net = net.eval()
    # for im, label in tqdm(train_data):
    #     if torch.cuda.is_available():
    #         # label = label+80
    #         im = im.cuda()  # (bs, 3, h, w)
    #         label = label.float().cuda()  # (bs, h, w)
    #         output2 = net(im)[1]
    #         Y_label = torch.cat((Y_label, label), 0)
    #         Y_Feature = torch.cat((Y_Feature, output2), 0)
    #
    # AverageOfClass, CovarianceOfClass = Obtain_AverageCovariance(train_data, net, mapgT_PEDCC)
    # gausi_valid_acc = 0
    # net = net.eval()
    # for im, label in valid_data:
    #     if torch.cuda.is_available():
    #         # label = label + 80
    #         im = im.cuda()
    #         label = label.cuda()
    #     output2 = net(im)[1]
    #     output2 = torch.mm(output2, mapgT_PEDCC)
    #     output2 = output2[:, -9:]
    #     output = Predict_Test(output2, AverageOfClass, CovarianceOfClass)
    #     gausi_valid_acc += get_acc(output, label)
    # print("GausiPredict Accuracy: ", gausi_valid_acc / len(valid_data))

    # covariance100 = torch.Tensor([]).cuda()
    # covariance100 = torch.zeros(256, 256).cuda()
    # maxTarget_eve = 0
    # for im, label in tqdm(train_data):
    #     if torch.cuda.is_available():
    #         # label = label+80
    #         im = im.cuda()  # (bs, 3, h, w)
    #         label = label.cuda()  # (bs, h, w)
    #     output, output1 = net(im)
    #     ret = torch.sort(output1.view(-1))[0]
    #     threshold = ret.size(0) * 4 // 5
    #     maxTarget_eve += torch.mean(ret[threshold:]).data
    # maxTarget = maxTarget_eve / len(train_data)
    # Y_Feature = torch.Tensor([]).cuda()
    # Y_label = torch.Tensor([]).cuda()
    for epoch in range(cfg['max_epoch']):
        if epoch in cfg['lr_steps']:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)#5e-4
            # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_acc = 0
        length, num = 0, 0
        length_test = 0
        max_size_train = 0
        min_size_train = 0
        max_size_val = 0
        min_size_val = 0
        # Y_Feature = torch.Tensor([]).cuda()
        # Y_label = torch.Tensor([]).cuda()
        # covariance100 = torch.zeros(512, 512).cuda()
        # Y_Feature = torch.Tensor([]).cuda()
        # Y_label = torch.Tensor([]).cuda()

        net = net.train()
        idx = 0
        # maxTarget_eve = 0
        # print(maxTarget)
        # Feature_Total = torch.zeros(map_PEDCC.shape[0], map_PEDCC.shape[1]).cuda()
        # Feature_Number = torch.zeros(map_PEDCC.shape[0]).cuda()
        for im, label in tqdm(train_data):
            if torch.cuda.is_available():
                #label = label+80
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                # print('label:', label)
                # error_tensor_empty = torch.Tensor([]).cuda()
                # tensor_empty = torch.Tensor([]).cuda()
                # for label_index in label:
                #     # map_dict_tensor = torch.from_numpy(map_dict[label_index.item()])
                #     tensor_empty = torch.cat((tensor_empty, map_PEDCC[label_index.item()].float().cuda()), 0)
                #     # error_tensor_empty = torch.cat((error_tensor_empty, map_PEDCC[0:label_index.item()].float().cuda()), 0)
                #     error_tensor_empty = torch.cat((error_tensor_empty, map_PEDCC[0:label_index.item()].float().cuda(),
                #                                     map_PEDCC[label_index.item() + 1:].float().cuda()), 0)
                # label_mse_tensor = tensor_empty.view(label.shape[0], -1)  # (batchSize, dimension)
                # error_label_mse_tensor = error_tensor_empty.view(label.shape[0], classes_num - 1,
                #                                                  -1)  # (batchSize, error_classes_num, dimension)
                # label_mse_tensor = label_mse_tensor.cuda()
                # error_label_mse_tensor = error_label_mse_tensor.cuda()

                # tensor_empty1 = map_PEDCC[label].float().cuda()
                # label_mse_tensor = tensor_empty1.view(label.shape[0], -1)  # (batchSize, dimension)
                # label_mse_tensor = label_mse_tensor.cuda()
                # all_label_mse_tensor = map_PEDCC[:].float().cuda()
                # all_label_mse_tensor = all_label_mse_tensor.unsqueeze(0)
                # all_label_mse_tensor = all_label_mse_tensor.expand(label.shape[0], -1, -1)

                # all_error_label_mse_tensor = torch.zeros(9, label_mse_tensor.shape[0], label_mse_tensor.shape[1])
                # for i in range(9):
                #     error_label = (label + i + 1) % 10
                #     error_tensor_empty = torch.Tensor([]).cuda()
                #     for error_label_index in error_label:
                #         # error_tensor_map_dict = torch.from_numpy(map_dict[error_label_index.item()])
                #         error_tensor_empty = torch.cat((error_tensor_empty, map_dict[error_label_index.item()].float().cuda()), 0)
                #     error_label_mse_tensor = error_tensor_empty.view(error_label.shape[0], -1)  # (batchSize, dimension)
                #     error_label_mse_tensor = error_label_mse_tensor.cuda()
                #     all_error_label_mse_tensor[i] = error_label_mse_tensor
                # all_error_label_mse_tensor = all_error_label_mse_tensor.cuda()

            # forward
            # ims = im.shape
            # output = net(im)
            # output, output0, output1, output2, output3, output4 = net(im)
            # x_0, x_11, x_12, x_13, x_21, x_22, x_23, x_24, x_31, x_32, x_33, x_34, x_35, x_36, x_41, x_42, x_43, out = net(im)
            output = net(im)
            # print("Value: ")
            # norm = torch.norm(output, p=2, dim=1, keepdim=True)
            # print(norm)
            # print("Value_total: ")
            # print(torch.mean(norm))
            # loss = criterion(output, label)
            # if epoch == cfg['max_epoch']-1:
            # loss1 = NewLoss(output1, label_mse_tensor)
            # loss2 = NewLoss(output2, label_mse_tensor)
            # loss = loss1 + loss2
            # loss = NewLoss(output1, label_mse_tensor)
            # Y_label = torch.cat((Y_label, label.float()), 0)
            # Y_Feature = torch.cat((Y_Feature, output2), 0)

            # Feature_Total[label] += output2
            # for i in range(cfg['batch_size']):
            #     Feature_Total[label[i]] += output2
            #     Feature_Number[label[i]] += 1
            # print("label:", label)
            # print('output2: ', output2)
            # print('Feature_Total: ', Feature_Total)
            # Feature_Number = Feature_Number.unsqueeze(1).expand(-1, Feature_Total.shape[1])
            # a = Feature_Total / Feature_Number
            # print("label: ", label)
            # print("Feature_Total: ", Feature_Total)
            # print("Feature_Number: ", Feature_Number)
            # print("a: ", a)
            #output = torch.log(output)
            # loss1 = criterion(output, label)
            # loss2 = criterion1(output2, label_mse_tensor)
            loss1 = criterion2(output[17], label)
            loss2 = blockDecorelation(output)
            # loss2 = criterion2(output, label)
            # print(output1.view(output1.size(0), -1).shape)
            # input = torch.Tensor([]).cuda()
            # for i in range(output1.size(0)):
            #     input = torch.cat((input, output1[i, idx:idx+1]), dim=0)
            #     idx = idx + 1
            #     if( idx >= output1.size(1) ):
            #         idx = 0
            # loss2 = mmd_rbf(input.view(-1, 1), distribution_zero_one)
            # loss2, input_avg, max_size_b, min_size_b = preceding_stage_loss(output1, dim=1)
            # loss2 = sortDivideLoss(output1, 1) + sortDivideLoss(output2, 1) + sortDivideLoss(output3, 1) + sortDivideLoss(output4, 1)
            # loss2 = tendToPEDCC(output1, map_PEDCC1280, 1)
            # loss2 = DecorrelationBetweenDim1(output2, 1)
            # loss3 = DecorrelationBetweenDim1(output3, 1)
            # loss4 = DecorrelationBetweenDim1(output3, 1)
            # loss2 = NewDecorrelationBetweenDim1(output0, 1)
            # loss2 = NewDecorrelationBetweenDim1(output0, 1)
            # loss2 = NewDecorrelationBetweenDim1to5(output0, 1)
            # loss3 = NewDecorrelationBetweenDim1(output1, 1)
            # loss4 = NewDecorrelationBetweenDim1(output2, 1)
            # loss5 = NewDecorrelationBetweenDim1(output3, 1)
            # loss6 = NewDecorrelationBetweenDim1(output4, 1)
            # loss7 = NewDecorrelationBetweenDim1(output5, 1)

            # loss4 = DecorrelationBetweenDim1(output3, 1)

            loss = loss1 + loss2
            #+ loss3 + loss4 + loss5 + loss6
            # loss = loss1
            # loss3 = tendToPEDCC(output2, map_PEDCC512, 1)
            # loss4 = tendToPEDCC(output3, map_PEDCC1024, 1)
            # loss2 = tendToPEDCC(output4, map_PEDCC2048, 1)
            # loss3 = tendToPEDCC(output5, map_PEDCC2048, 1)
            # loss2 = FeatureToOneOrZero(output3, 0.1)
            # loss2 = criterion2(output, label)
            # print(loss2)
            # print(loss3)
            # print(loss4)
            # print(loss5)
            # loss2 = DecorrelationBetweenDim(output3, 1)
            # loss3 = sortDivideLoss(output2.view(-1), 0.15, 0.1)
            # loss2 = 0.007 * epoch * reduceVarAndAvg(output3, 0.1)
            # loss2 = averageDivideLoss(output1.view(-1))
            # loss2 = sortDivideLoss(output4, 10)
            # loss2 = weightReduceLoss(output1, 1)
            # loss = loss1
            # maxTarget_eve += mean
            # max_size_train += max_size_b
            # min_size_train += min_size_b
            # f = open(save_folder + 'avg_train_val.txt', 'a+')
            # f.write("input_avg_train:" + str(input_avg.view(-1)) + '\n')
            # f.close()
            # print("input_avg_train:")
            # print(input_avg.view(-1))
            # loss2 = NewLoss(output2, label_mse_tensor)
            # loss1 = criterion2(output, label)
            # print('loss1:',loss1)
            # loss2 = criterion2(output, label)
            # print('loss2:', loss2)
            # loss1 = CorrectMinLoss(output2, label_mse_tensor)
            # loss2 = criterion0(output2, label)[0]
            # loss = loss1+loss2
            # loss1 = NewLoss(output2, label_mse_tensor)
            # if epoch > 0:
            #     covariance100 = CovarianceLoss(covariance100, output2, label, mapgT_PEDCC)
            #     triangle100 = torch.zeros(100, 99).cuda()
            #     for i in range(triangle100.shape[0]):
            #         triangle100[i] = torch.diagonal(covariance100[i])
            #     loss2 = MinCovarianceLoss(covariance100, triangle100)
            #     loss = loss1 + loss2
            # else:
            #     loss = loss1
            # covariance100 = covariance100.detach()
            # print("covariance100: ", covariance100.requires_grad)
            # print('covariance100[0][0]:', covariance100[0][0])

            # loss3 = criterion0(output2, label, covariance100)[0]

            # loss = LR * loss1 + loss2
            # loss = CovarianceLoss(covariance100, output2, label, map_PEDCC)
            # if epoch > 0:
            #     loss = CovarianceLoss(covariance100, output2, label, map_PEDCC)
            # else:
            #     loss = NewLoss(output2, label_mse_tensor)
            # loss1, ret_sum = CorrectMinLoss(output2, label_mse_tensor, label)
            # loss2 = ErrorMinLoss(output2, all_label_mse_tensor, ret_sum)
            # loss = NewLoss(output2, label_mse_tensor)
            # loss = loss1 + loss2 + loss3
            # loss = 256 * criterion1(output2, label_mse_tensor)
            # freedom_degree = GetFreedomDegree()
            # print(loss)
            # loss1 = CorrectMinLoss(output2, label_mse_tensor)
            # loss2 = ErrorMinLoss(output2, all_error_label_mse_tensor)
            # loss2 = ErrorMinLoss(output2, error_label_mse_tensor)
            # loss2 = ErrorMinLoss(output2, all_error_label_mse_tensor)
            # loss = criterion2(output, label)
            # loss1 = criterion2(output, label)
            # loss2 = criterion2(output, label)
            # loss = loss1 + 0.01*loss2
            # loss = loss1 + loss2
            # loss = ThreesqrtLoss(output2, label_mse_tensor)
            length += output[17].pow(2).sum().item()
            num += output[17].shape[0]
            # backward
            # if epoch > 0:
            optimizer.zero_grad()
            loss.backward()
                # loss1.backward()
                # if epoch > 0:
                #     loss2.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_acc += get_acc(output[17], label)
        # maxTarget = maxTarget_eve / len(train_data)
        # print(maxTarget)
        # Feature_Number = Feature_Number.unsqueeze(1).expand(-1, Feature_Total.shape[1])
        # Feature_Average = Feature_Total / Feature_Number
        # Feature_Average_T = torch.mm(Feature_Average, mapgT_PEDCC)
        # Feature_Average_Real = Feature_Average_T[:, -(classes_num-1):]
        # AverageOfClass, CovarianceOfClass = Obtain_AverageCovariance(train_data, net)
        # if epoch == 0:
        # covariance100 = Obtain_Covariance100(Y_label, Y_Feature, map_PEDCC)
        # covariance100 = covariance100.detach()
            # covariance100 = covariance100.requires_grad_(requires_grad=False)
            # covariance100 = covariance100.detach()
            # print("covariance100: ", covariance100.requires_grad)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_data
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    #label = label + 80
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)[17]
                loss = criterion2(output, label)
                # loss2, input_avg, max_size_b, min_size_b = preceding_stage_loss(output1, dim=1)
                # loss = loss1 + loss2
                # max_size_val += max_size_b
                # min_size_val += min_size_b
                # f = open(save_folder + 'avg_train_val.txt', 'a+')
                # f.write("input_avg_val:" + str(input_avg.view(-1)) + '\n')
                # f.close()
                # print("input_avg_val:")
                # print(input_avg.view(-1))
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
                length_test += output.pow(2).sum().item()/im.shape[0]
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f, "
                # "Max_size_train: %d, Min_size_train: %d, Max_size_val/Min_size_val: %f, Max_size_val: %d, Min_size_val: %d, Max_size_val/Min_size_val: %f"
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR, train_loss1 / len(train_data),
                   train_loss2 / len(train_data)))
                # , max_size_train, min_size_train, (float)(max_size_train) / min_size_train,
                #    max_size_val, min_size_val, (float)(max_size_val) / min_size_val))
            loss_train.append(train_loss / len(train_data))
            loss1_train.append(train_loss1 / len(train_data))
            loss2_train.append(train_loss2 / len(train_data))
            loss_val.append(valid_loss / len(valid_data))
            acc_train.append(train_acc / len(train_data))
            acc_val.append(valid_acc / len(valid_data))
            # ratio_train.append((float)(max_size_train) / min_size_train)
            # ratio_val.append((float)(max_size_val) / min_size_val)
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open(save_folder + save_txt + '.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        # if train_acc / len(train_data) > 0.9995:
        #     break
        if (epoch+1) % 10 == 0:
            torch.save(net.module.state_dict(), save_folder + save_txt + str(epoch+1) + '_epoch.pth')

        # before_time = datetime.now()
        # # print('Y_label:', Y_label.shape)
        # # print('Y_Feature:', Y_Feature.shape)
        #
        # Freedom_Degree = Get100FreedomDegree(Y_label, Y_Feature, map_PEDCC, mapgT_PEDCC, epoch)
        # # Freedom_Degree = GetFreedomDegree(Y_label, Y_Feature, map_PEDCC, mapgT_PEDCC, epoch)
        #
        #
        # # AverageOfClass, CovarianceOfClass = Obtain_AverageCovariance(Y_label, Y_Feature, mapgT_PEDCC)
        # # Gausi_all = ("Epoch %d :" % (epoch))
        # # fa = open(save_folder + 'Guasi_matrix.txt', 'a+')
        # # for i in range(10):
        # #     Gausi_eve = "Average: " + str(AverageOfClass[i]) +'\n'
        # #     Gausi_eve += "CovarianceOfClass: " + str(CovarianceOfClass[i]) + '\n'
        # #     Gausi_eve += "CovarianceOfClass_Det" + str(torch.det(CovarianceOfClass[i])) + '\n'
        # #     Gausi_all += Gausi_eve
        # # fa.write(Gausi_all + '\n')
        # # fa.close()
        # # gausi_valid_acc = 0
        #
        # kafang_valid_acc = 0
        # # v_Feature = torch.Tensor([]).cuda()
        # # v_label = torch.Tensor([]).cuda()
        #
        # net = net.eval()
        # for im, label in valid_data:
        #     if torch.cuda.is_available():
        #         # label = label + 80
        #         im = im.cuda()
        #         label = label.cuda()
        #     output2_1 = net(im)[1]
        #     # output2 = torch.mm(output2, mapgT_PEDCC)
        #     # output2 = output2[:, -9:]
        #     # v_label = torch.cat((v_label, label.float()), 0)
        #     # v_Feature = torch.cat((v_Feature, output2_1), 0)
        #     if epoch == 119:
        #         drawTestHistogram(output2_1, label, map_PEDCC, mapgT_PEDCC)
        #     # map_PEDCC = map_PEDCC.cuda()
        #     # pedcc_average = torch.mm(map_PEDCC, mapgT_PEDCC)
        #     # pedcc_average = pedcc_average[:, -9:]
        #     # output = Predict_Test(output2, pedcc_average, CovarianceOfClass)
        #
        #     # output = Predict_Test(output2, AverageOfClass, CovarianceOfClass)
        #     # print('output.shape: ', output.shape)
        #     # output = torch.from_numpy(output)
        #     # print('output.shape: ', output.shape)
        #     # print('label.shape: ', label.shape)
        #     # gausi_valid_acc += get_acc(output, label)
        #         output_1 = kafang100_predict(output2_1, map_PEDCC, mapgT_PEDCC, Freedom_Degree)
        #     # output_1 = kafang_predict(output2_1, map_PEDCC, mapgT_PEDCC, Freedom_Degree)
        #         output_1 = torch.from_numpy(output_1)
        #         kafang_valid_acc += get_acc(output_1, label.cpu())
        #
        # # vFreedom_Degree = vGet100FreedomDegree(v_label, v_Feature, map_PEDCC, mapgT_PEDCC, epoch)
        # if epoch == 119:
        #     after_time = datetime.now()
        #     gh, gremainder = divmod((after_time - before_time).seconds, 3600)
        #     gm, gs = divmod(gremainder, 60)
        #     gtime_str = "Time %02d:%02d:%02d" % (gh, gm, gs)
        #     Gausi_str = ("KafangPredict Acc: %f, " %
        #                         (kafang_valid_acc/len(valid_data)))
        #     gausi_acc_val.append(kafang_valid_acc/len(valid_data))
        #     print(Gausi_str + gtime_str)
        #     f = open(save_folder + 'tiny180_200pedcc.txt', 'a+')
        #     f.write(Gausi_str + gtime_str + '\n')
        #     f.close()

    history['loss_train'] = loss_train
    history['loss1_train'] = loss1_train
    history['loss2_train'] = loss2_train
    history['loss_val'] = loss_val
    history['acc_train'] = acc_train
    history['acc_val'] = acc_val
    history['ratio_train'] = ratio_train
    history['ratio_val'] = ratio_val
    # history['gausi_acc_val'] = gausi_acc_val
    draw(history)
#POD
# def read_pkl(path):
#     f = open(path, 'rb')
#     # print(123)
#     a = pickle.load(f)
#     f.close()
#     return a

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def draw(history):
    epochs = range(1, len(history['loss_train']) + 1)
    plt.plot(epochs, history['loss_train'], 'blue', label='Training loss')
    plt.plot(epochs, history['loss_val'], 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./Training and Validation loss.jpg')
    plt.figure()
    epochs = range(1, len(history['acc_train']) + 1)
    plt.plot(epochs, history['acc_train'], 'b', label='Training acc')
    plt.plot(epochs, history['acc_val'], 'r', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('./Training and validation acc.jpg')
    plt.show()


#####NaCLoss##################
def NaCLoss(input, target, delta):
    ret_before = input * target
    ret_before = torch.sum(ret_before, dim=1).view(-1, 1)

    add_feature = delta * torch.ones((input.shape[0], 1)).cuda()
    input_after = torch.cat((input, add_feature), dim=1)
    input_after_norm = torch.norm(input_after, p=2, dim=1, keepdim=True)

    ret = ret_before / input_after_norm
    # threshold t=500, when t>500, t =1000
    # only calc loss2 for classify, not influence loss1 for genaration
    # t_index = torch.where(t_index > 500, torch.tensor(999).to(t_index.device), t_index)
    ret = 1 - ret
    ret = ret.pow(2)
    ret = torch.mean(ret)

    return ret


#####SCLoss#########################
def SCLoss(map_PEDCC, label, feature):
    average_feature = map_PEDCC[label.long().data].float().cuda()
    feature_norm = l2_norm(feature)
    feature_norm = feature_norm - average_feature
    covariance100 = 1 / (feature_norm.shape[0] - 1) * torch.mm(feature_norm.T, feature_norm).float()
    covariance100_loss = torch.sum(pow(covariance100, 2)) - torch.sum(pow(torch.diagonal(covariance100), 2))
    covariance100_loss = covariance100_loss / (covariance100.shape[0] - 1)
    return covariance100_loss


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


#####CosineLoss##################
def CosineLoss(input, target):
    ret = input * target
    ret_sum = torch.sum(ret, dim=1)
    ret = 1 - ret_sum
    ret = ret ** 2
    ret = torch.mean(ret)

    return ret


#####RCLoss#########################
def RCLoss(map_PEDCC, label, feature):
    average_feature = map_PEDCC[label.long().data].float().cuda()
    feature = feature - average_feature
    covariance100 = 1 / (feature.shape[0] - 1) * torch.mm(feature.T, feature).float()
    covariance100_loss = torch.sum(pow(covariance100, 2)) - torch.sum(pow(torch.diagonal(covariance100), 2))
    covariance100_loss = covariance100_loss / (covariance100.shape[0] - 1)
    return covariance100_loss, covariance100

def train_soft_mse_zl731(net, train_data, valid_data, cfg, criterion, criterion1, criterion2, save_folder, save_txt, classes_num):
    LR = cfg['LR']
    freeze_layers = nn.Sequential(net.pre_layers, net.layer2[0].layers[0], net.layer2[0].layers[1], net.layer2[1].layers[0],
                     net.layer2[1].layers[1], net.layer4[0].layers[6], net.layer4[0].layers[7],
                     net.layer4[1].layers[0], net.layer4[1].layers[1], net.layer4[1].layers[6], net.layer4[1].layers[7],
                     net.layer4[2].layers[0], net.layer4[2].layers[1], net.layer4[2].layers[6], net.layer4[2].layers[7])
    for param in freeze_layers.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(cfg['max_epoch']):
        if epoch in cfg['lr_steps']:
            if epoch != 0:
                LR *= 0.1

            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, momentum=0.9, weight_decay=5e-4)#5e-4
            # print(net)
            # optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4)
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_acc1 = 0
        train_acc5 = 0
        length, num = 0, 0
        length_test = 0
        # l2_norm_trainSample = torch.Tensor([]).cuda()
        net = net.train()
        idx = 0
        for im, label in tqdm(train_data):
            if torch.cuda.is_available():
                #label = label+80
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                # tensor_empty1 = map_PEDCC[label].float().cuda()
                # label_mse_tensor = tensor_empty1.view(label.shape[0], -1)  # (batchSize, dimension)
                # label_mse_tensor = label_mse_tensor.cuda()

            output = net(im)
            # with torch.no_grad():
            #     l2_norm_trainSample = torch.cat((l2_norm_trainSample, torch.norm(output[13], p=2, dim=1)), dim=0)  # [b]
            loss1 = criterion2(output[17], label)#
            # print(output[16].shape)
            # print(label_mse_tensor.shape)
            # loss1 = NaCLoss(output[16], label_mse_tensor, delta)
            # loss2 = SCLoss(map_PEDCC, label, output[16])
            # loss2 = loss1
            # loss1 = loss1 + loss2
            # loss2 = blockDecorelation(output)#resnet50
            # loss2 = vgg_blockDecorelation(output)
            # loss2 = mobile_v2_blockDecorelation(output)
            # loss2 = NewDecorrelationBetweenDim1(output[18], 1) + NewDecorrelationBetweenDim1(output[0], 1)
            loss2 = NewDecorrelationBetweenDim1(output[0], 1) + NewDecorrelationBetweenDim1(output[4][0],
                                                                                            1) + NewDecorrelationBetweenDim1(
                output[5][0], 1) + NewDecorrelationBetweenDim1(output[15][2], 1) + NewDecorrelationBetweenDim1(
                output[14][2], 1) + NewDecorrelationBetweenDim1(output[16][0], 1) + NewDecorrelationBetweenDim1(
                output[16][2], 1) + NewDecorrelationBetweenDim1(output[15][0], 1)
            loss = loss1 + loss2
            # if (epoch+1) > 30:
            #     loss = 3*loss1 + loss2
            # if (epoch + 1) > 100:
            #     loss = loss1 + 0.1 * loss2
            length += output[17].pow(2).sum().item()
            num += output[17].shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_acc1 += get_acc(output[17], label)
            # train_acc5 += top5_accuracy(output[17], label)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_data
            valid_loss = 0
            valid_acc1 = 0
            valid_acc5 = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    #label = label + 80
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)[17]
                loss = criterion2(output, label)
                valid_loss += loss.data
                valid_acc1 += get_acc(output, label)
                # valid_acc5 += top5_accuracy(output, label)
                length_test += output.pow(2).sum().item()/im.shape[0]
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc1: %f, Train Acc5: %f, Valid Loss: %f, Valid Acc1: %f,  Valid Acc5: %f, LR: %f, Train Loss1: %f, Train Loss2: %f, "
                # "Max_size_train: %d, Min_size_train: %d, Max_size_val/Min_size_val: %f, Max_size_val: %d, Min_size_val: %d, Max_size_val/Min_size_val: %f"
                % (epoch, train_loss / len(train_data),
                   train_acc1 / len(train_data), train_acc5 / len(train_data), valid_loss / len(valid_data),
                   valid_acc1 / len(valid_data), valid_acc5 / len(valid_data), LR, train_loss1 / len(train_data),
                   train_loss2 / len(train_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc1: %f, Train Acc5: %f," %
                         (epoch, train_loss / len(train_data),
                          train_acc1 / len(train_data),train_acc5 / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open(save_folder + save_txt + '.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        # if train_acc / len(train_data) > 0.9995:
        #     break
        if (epoch+1) % 50 == 0:
        # if (epoch + 1) > 0:
            torch.save(net, save_folder + save_txt + str(epoch + 1) + '.pkl')
            # if (epoch+1) % 10 == 0:
            # torch.save(net.module.state_dict(), save_folder + save_txt + str(epoch+1) + '_epoch_model.pth')
            torch.save(net(im), save_folder + save_txt + str(epoch + 1) + 'output.pth')
        # if (epoch+1) % 10 == 0:
        #     torch.save(net.module.state_dict(), save_folder + save_txt + str(epoch+1) + '_epoch.pth')

    # history['loss_train'] = loss_train
    # history['loss1_train'] = loss1_train
    # history['loss2_train'] = loss2_train
    # history['loss_val'] = loss_val
    # history['acc_train'] = acc_train5
    # history['acc_val'] = acc_val5
    # history['ratio_train'] = ratio_train
    # history['ratio_val'] = ratio_val
    # # history['gausi_acc_val'] = gausi_acc_val
    # draw(history)


def train_soft_0602(net, train_data, valid_data, cfg, criterion, criterion1, criterion2, save_folder, save_txt, classes_num):
    LR = cfg['LR']
    # if n_gpu > 1:
    #     model = BalancedDataParallel(2, model, dim=0).to(device)
        # model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        # if n_gpu > 1:
        # net = torch.nn.DataParallel(net, device_ids=device_ids)
        # net = BalancedDataParallel(0, net, dim=0).to(device_ids[0])
        net = net.cuda()
    prev_time = datetime.now()
    history = dict()
    loss_train = []
    loss1_train = []
    loss2_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    ratio_train = []
    ratio_val = []
    for epoch in range(cfg['max_epoch']):
        # if epoch == 15:
        #     LR = 0.05
        if epoch in cfg['lr_steps']:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)#5e-4
            # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_acc = 0
        length, num = 0, 0
        length_test = 0
        max_size_train = 0
        min_size_train = 0
        max_size_val = 0
        min_size_val = 0
        net.train()
        idx = 0
        for im, label in tqdm(train_data):
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
            output = net(im)

            loss1 = criterion2(output[5], label)
            loss2 = blockDecorelation(output)
            # loss2 = loss1

            loss = loss1 + loss2


            length += output[5].pow(2).sum().item()
            num += output[5].shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_acc += get_acc(output[5], label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        with torch.no_grad():
            if valid_data is not None:
                valid_data
                valid_loss = 0
                valid_acc = 0
                net.eval()
                for im, label in valid_data:
                    if torch.cuda.is_available():
                        #label = label + 80
                        im = im.cuda()
                        label = label.cuda()
                    output = net(im)[5]
                    loss = criterion2(output, label)

                    valid_loss += loss.data
                    valid_acc += get_acc(output, label)
                    length_test += output.pow(2).sum().item()/im.shape[0]
                epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f, "
                    # "Max_size_train: %d, Min_size_train: %d, Max_size_val/Min_size_val: %f, Max_size_val: %d, Min_size_val: %d, Max_size_val/Min_size_val: %f"
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data), LR, train_loss1 / len(train_data),
                       train_loss2 / len(train_data)))
                loss_train.append(train_loss / len(train_data))
                loss1_train.append(train_loss1 / len(train_data))
                loss2_train.append(train_loss2 / len(train_data))
                loss_val.append(valid_loss / len(valid_data))
                acc_train.append(train_acc / len(train_data))
                acc_val.append(valid_acc / len(valid_data))

            else:
                epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                             (epoch, train_loss / len(train_data),
                              train_acc / len(train_data)))
            prev_time = cur_time
            print(epoch_str + time_str)
            f = open(save_folder + save_txt + '.txt', 'a+')
            f.write(epoch_str + time_str + '\n')
            f.close()
            # 100 or 150
            if (epoch+1) % 10 == 0:
                torch.save(net, save_folder + save_txt + '.pkl')
            # if (epoch+1) % 10 == 0:
                # torch.save(net.module.state_dict(), save_folder + save_txt + str(epoch+1) + '_epoch_model.pth')
                torch.save(net(im), save_folder + save_txt + str(epoch + 1) + 'output.pth')


        # history['loss_train'] = loss_train
        # history['loss1_train'] = loss1_train
        # history['loss2_train'] = loss2_train
        # history['loss_val'] = loss_val
        # history['acc_train'] = acc_train
        # history['acc_val'] = acc_val
        # history['ratio_train'] = ratio_train
        # history['ratio_val'] = ratio_val
        # history['gausi_acc_val'] = gausi_acc_val
        # draw(history)




def train_soft_mse_zl_2(net, train_data, valid_data, num_epochs, criterion, criterion1, modelname=None):
    LR = 0.1
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()

    prev_time = datetime.now()
    map_dict = read_pkl()
    for epoch in range(num_epochs):
        if epoch in [0, 10, 20, 50, 80]:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_acc = 0
        net = net.train()
        for im, label in tqdm(train_data):
            if torch.cuda.is_available():
                #label = label+80
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                # tensor_empty = torch.Tensor([]).cuda()
                # for label_index in label:
                #     tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()), 0)
                #
                # label_mse_tensor = tensor_empty.view(-1, 200)       #(-1, dimension)
                # label_mse_tensor = label_mse_tensor.cuda()

            # forward
            output = net(im)

            loss1 = criterion(output, label)
            # loss2 = criterion1(net(im)[2], label_mse_tensor)
            loss = loss1
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()

            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    #label = label + 80
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR, train_loss1 / len(train_data), train_loss2 / len(train_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open('./fashion_soft.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        # if train_acc / len(train_data) > 0.9995:
        #     break
    if modelname:
        torch.save(net, modelname)

def train_soft_mse_cov(net, train_data, valid_data, num_epochs, criterion, criterion1, modelname=None):
    LR = 0.1
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    prev_time = datetime.now()
    map_dict = read_pkl()
    for epoch in range(num_epochs):
        if epoch in [0, 20, 50, 80, 100]:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_acc = 0
        train_cov = {}
        net = net.train()
        for step, (im, label) in enumerate(train_data):
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                for i in range(100):
                    train_cov[i] = torch.Tensor([]).cuda()
                tensor_empty = torch.Tensor([]).cuda()
                for label_index in label:
                    tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()), 0)

                label_mse_tensor = tensor_empty.view(-1, 512)       #(-1, dimension)
                label_mse_tensor = label_mse_tensor.cuda()

            # forward
            output = net(im)[0]
            loss1 = criterion(output, label)
            loss2 = criterion1(net(im)[1], label_mse_tensor)
            loss3 = 0
            mat = torch.cat((label.reshape(-1, 1).float(), net(im)[1]), 1).cuda()
            for i in mat:
                train_cov[int(i[0].item())] = torch.cat((train_cov[int(i[0].item())], i[1:513]), 0)
            for i in range(100):
                train_cov[i] = train_cov[i].view(-1, 512)
                if train_cov[i].shape[0] > 1:
                    cov = torch.div(torch.mm(torch.t(train_cov[i]), train_cov[i]-map_dict[i].float().cuda()),
                                    train_cov[i].shape[0]-1).cuda()
                    loss3 +=  (cov**2).sum()
            if step>0:
                loss3.data = loss3_last.data*0.9+loss3.data*0.1
            loss3_last = loss3
            loss2 = loss2*100
            loss3 = loss3/100
            loss = loss1 + loss2 + loss3
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()

                output = net(im)[0]
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f, Train Loss3: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR, train_loss1 / len(train_data), train_loss2 / len(train_data), train_loss3 / len(train_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        # f = open('./log.txt', 'a+')
        # f.write(epoch_str + time_str + '\n')
        # f.close()
        if train_acc / len(train_data) > 0.9995:
            break
    if modelname:
        torch.save(net, modelname)

def train_A(net, train_data, valid_data, num_epochs, criterion):
    LR = 0.1
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        if epoch in [0, 20, 50, 80, 100]:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)

            # forward
            output = net(im)[0]
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_acc += get_acc(output[0], label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()

                output = net(im)[0]
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output[0], label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)

def train_LArc(net, train_data, valid_data, num_epochs, criterion):
    LR=0.1
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        if epoch in [0, 20, 50, 80, 100]:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)

            # forward
            output = net(x=im, target=label)[0]
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()

                output = net(x=im)[0]
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)

def train_arc_soft_mse(net, train_data, valid_data, num_epochs, criterion, criterion1, modelname=None):
    LR = 0.1
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
    prev_time = datetime.now()
    map_dict = read_pkl()
    for epoch in range(num_epochs):
        if epoch in [0, 20, 50, 80, 100]:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                tensor_empty = torch.Tensor([]).cuda()
                for label_index in label:
                    tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()), 0)

                label_mse_tensor = tensor_empty.view(-1, 512)
                label_mse_tensor = label_mse_tensor.cuda()

            # forward
            output = net(x=im, target=label)[0]
            loss1 = criterion(output, label)
            loss2 = criterion1(net(x=im, target=label)[1], label_mse_tensor)
            loss = loss1 + loss2
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()

                output = net(x=im)[0]
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR, train_loss1 / len(train_data), train_loss2 / len(train_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        # f = open('./log.txt', 'a+')
        # f.write(epoch_str + time_str + '\n')
        # f.close()
        if train_acc / len(train_data) > 0.9995:
            break
    if modelname:
        torch.save(net, modelname)

def train_det(net, train_data, valid_data, num_epochs, criterion, modelname=None):
    LR = 0.1
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        if epoch in [0, 20, 50, 80]:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)

            # forward
            output = net(im)[0]
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()

                output = net(im)[0]
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        ############## train-det
        train_det=0.
        for step, (im, label) in enumerate(train_data):
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)

            output = net(im)[1].cpu().detach().numpy()
            label = label.cpu().numpy()
            label = label.reshape(len(label), -1)
            out = np.hstack([label, output])
            if step == 0:
                mat = out
            else:
                mat = np.vstack((mat, out))
        mat = mat[np.lexsort(mat[:, ::-1].T)]
        for i in range(100):
            m = mat[np.where(mat[:, 0]==i)]
            m = m[:, 1:101].T
            train_det += np.linalg.det(10*np.cov(m))

        print("train-det:", train_det, np.log10(train_det))
        ############## test-det
        test_det = 0.
        for step, (im, label) in enumerate(valid_data):
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)

            output = net(im)[1].cpu().detach().numpy()
            label = label.cpu().numpy()
            label = label.reshape(len(label), -1)
            out = np.hstack([label, output])
            if step == 0:
                mat = out
            else:
                mat = np.vstack((mat, out))
        mat = mat[np.lexsort(mat[:, ::-1].T)]
        for i in range(100):
            m = mat[np.where(mat[:, 0] == i)]
            m = m[:, 1:101].T
            test_det += np.linalg.det(10*np.cov(m))

        print("test-det:", test_det, np.log10(test_det))
        #print("test/train:", test_det/train_det)
    if modelname:
        torch.save(net, modelname)

def plot_2d_features(model, test_loader):
    net_logits = np.zeros((10000, 2), dtype=np.float32)
    net_labels = np.zeros((10000,), dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for b_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            _, output2d = model(data)
            output2d = output2d.cpu().data.numpy()
            # for x in output2d:
            #     y = x[0]*x[0]+x[1]*x[1]
            #     y = y**0.5
            #     x[0] = 5*x[0]/y
            #     x[1] = 5*x[1]/y

            target = target.cpu().data.numpy()
            net_logits[b_idx * 100: (b_idx + 1) * 100, :] = output2d
            net_labels[b_idx * 100: (b_idx + 1) * 100] = target
        for label in range(10):
            idx = net_labels == label
            plt.scatter(net_logits[idx, 0], net_logits[idx, 1])
        plt.legend(np.arange(10, dtype=np.int32))
        plt.show()

def plot_3d_features(model, test_loader):
    net_logits = np.zeros((10000, 3), dtype=np.float32)
    net_labels = np.zeros((10000,), dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for b_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            _, output2d = model(data)
            output2d = output2d.cpu().data.numpy()
            # for x in output2d:
            #     y = x[0]*x[0]+x[1]*x[1]+x[2]*x[2]
            #     y = y**0.5
            #     x[0] = 1*x[0]/y
            #     x[1] = 1*x[1]/y
            #     x[2] = 1*x[2]/y
            target = target.cpu().data.numpy()
            net_logits[b_idx * 100: (b_idx + 1) * 100, :] = output2d
            net_labels[b_idx * 100: (b_idx + 1) * 100] = target
        fig = plt.figure()
        ax = p3d.Axes3D(fig)
        for label in range(10):
            idx = net_labels == label
            ax.scatter(net_logits[idx, 0], net_logits[idx, 1], net_logits[idx, 2])
        plt.legend(np.arange(10, dtype=np.int32))
        plt.show()

def extract_features(modelname, train_loader, test_loader):
    net = torch.load(modelname)
    for step, (im, label) in enumerate(train_loader):
        if torch.cuda.is_available():
            im = im.cuda()  # (bs, 3, h, w)
            label = label.cuda()  # (bs, h, w)

        output = net(im)[2].cpu().detach().numpy()
        label = label.cpu().numpy()
        label = label.reshape(len(label), -1)
        out = np.hstack([label, output])
        if step == 0:
            mat = out
        else:
            mat = np.vstack((mat, out))

    mat = mat[np.lexsort(mat[:, ::-1].T)]
    io.savemat('tiny_20Train.mat', {'name': mat})
    print("features_train is OK")
    for step, (im, label) in enumerate(test_loader):
        if torch.cuda.is_available():
            im = im.cuda()  # (bs, 3, h, w)l2
            label = label.cuda()  # (bs, h, w)

        output = net(im)[2].cpu().detach().numpy()
        label = label.cpu().numpy()
        label = label.reshape(len(label), -1)
        out = np.hstack([label, output])
        if step == 0:
            mat = out
        else:
            mat = np.vstack((mat, out))

    mat = mat[np.lexsort(mat[:, ::-1].T)]
    io.savemat('tiny_20TEST.mat', {'name': mat})
    print("features_test is OK")

def pkl2mat():
    map_dict = read_pkl()
    tensor_empty = torch.Tensor([]).cuda()
    for label in range(200):
        tensor_empty = torch.cat((tensor_empty,map_dict[label].float().cuda()),0)
    # b = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35])
    b=np.arange(200)
    output = tensor_empty.cpu().detach().numpy()
    label = b
    label = label.reshape(200,-1)
    mat = np.hstack([label,output])


    mat = mat[np.lexsort(mat[:, ::-1].T)]
    io.savemat('200_100.mat', {'name': mat})
    print("features_train is OK")


PEDCC_PATH = 'center_pedcc/10_256_s.pkl'
def read_pkl(PEDCC_PATH):
    #pedcc_path = os.path.join(conf.HOME, PEDCC_PATH)
    f = open(PEDCC_PATH, 'rb')
    a = pickle.load(f)
    f.close()
    return a
def read_pklFromFile(file_path):
    #pedcc_path = os.path.join(conf.HOME, PEDCC_PATH)
    f = open(file_path, 'rb')
    a = pickle.load(f)
    f.close()
    return a
PEDCCgT_PATH = 'center_pedcc/10_256T_s.pkl'
def gT_read_pkl():
    #pedcc_path = os.path.join(conf.HOME, PEDCC_PATH)
    f = open(PEDCCgT_PATH, 'rb')
    a = pickle.load(f)
    f.close()
    return a
# consieLinear层 实现了norm的fea与norm weight的点积计算，服务于margin based softmax loss
# 将提取的特征与pedcc求余弦相似度
class CosineLinear_PEDCC2(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear_PEDCC2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


    def forward(self, input):
        map_dict = read_pkl()
        tensor_empty = torch.Tensor([]).cuda()
        for label_index in range(self.out_features):
            tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
        label_40D_tensor = tensor_empty.view(-1, self.in_features)
        label_40D_tensor = label_40D_tensor.cuda()
        cos_theta = torch.Tensor([]).cuda()
        cos = nn.CosineSimilarity()
        for row in input:  # size=(B,F)    F is feature len
            out = torch.Tensor([]).cuda()
            row = row.unsqueeze(0)
            for row1 in label_40D_tensor:
                row1 = row1.unsqueeze(0)
                Similarity = cos(row, row1)
                out = torch.cat((out, Similarity),0)
            out = out.unsqueeze(0)
            cos_theta = torch.cat((cos_theta, out),0)

        return cos_theta  # size=(B,Classnum,1)

# consieLinear层 实现了norm的fea与norm weight的点积计算，服务于margin based softmax loss
# 将w替换成pedcc，固定
# 计算余弦距离
class CosineLinear_PEDCC(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear_PEDCC, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
        #self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        map_dict = read_pkl()
        tensor_empty = torch.Tensor([]).cuda()
        for label_index in range(self.out_features):
            # tensor_map_dict = torch.from_numpy(map_dict[label_index])
            tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
        label_40D_tensor = tensor_empty.view(-1, self.in_features).permute(1, 0)
        label_40D_tensor = label_40D_tensor.cuda()
        self.weight.data = label_40D_tensor
        #print(self.weight.data)

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        # ww = w.renorm(2, 1, 1e-5).mul(1e5)  # weights normed
        # xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        # wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(w)  # size=(B,Classnum)  x.dot(ww)
        # cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)  #
        # cos_theta = cos_theta.clamp(-1, 1)
        #cos_theta = cos_theta * xlen.view(-1, 1)

        # x = input
        # w = self.weight
        # x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        # x_norm = torch.div(x, x_norm)
        # w_norm = torch.norm(w, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        # w_norm = torch.div(w, w_norm)
        # cos_theta = torch.mm(x_norm, w_norm)

        return cos_theta  # size=(B,Classnum,1)


class Projection_Matrix(nn.Module):
    def __init__(self, in_features, out_features):
        super(Projection_Matrix, self).__init__()
        self.num_classes = 100
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
        #self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        map_dict = read_pkl()
        tensor_empty = torch.Tensor([]).cuda()
        for label_index in range(self.num_classes - 1):
            # tensor_map_dict = torch.from_numpy(map_dict[label_index])
            tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
        label_40D_tensor = tensor_empty.view(self.num_classes - 1, -1)
        # P = A*(A_T*A)_-1*A_T projection matrix
        A_Combine = label_40D_tensor.T
        Tmp = torch.mm(label_40D_tensor, A_Combine)
        Tmp = torch.inverse(Tmp)
        data = torch.mm(A_Combine, Tmp)
        data = torch.mm(data, label_40D_tensor)
        data = data.T
        data = data.cuda()
        # label_40D_tensor = label_40D_tensor.cuda()
        self.weight.data = data
        #print(self.weight.data)

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,F) F=in_features
        projection_after = x.mm(w)  # size=(B,F)

        return projection_after  # size=(B,F)
# AMSoftmax 层的pytorch实现，两个重要参数 scale，margin（不同难度和量级的数据对应不同的最优参数）
# 原始实现caffe->https://github.com/happynear/AMSoftmax
class AMSoftmax_PEDCC(nn.Module):
    def __init__(self, scale, margin, is_amp):
        super(AMSoftmax_PEDCC, self).__init__()
        self.scale = scale
        self.margin = margin
        self.is_amp = is_amp
    def forward(self, input, target):
        # self.it += 1
        cos_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index1 = ~index


        output = cos_theta * 1.0  # size=(B,Classnum)
        # for x in output[index]:
        #     if x > 0:
        #         x = x**3
        #     else:
        #         x = x**(1/3)
        output[index] -= self.margin

        # for x in output[index1]:
        #     if x > 0:
        #         x = x**(1/3)
        #     else:
        #         x = x**3

        if self.is_amp:
            output[index1] += self.margin
        output = output * self.scale


        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss

# consieLinear层 实现了norm的fea与norm weight的点积计算，服务于margin based softmax loss
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)  # weights normed
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)  x.dot(ww)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)  #
        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta = cos_theta * xlen.view(-1, 1)

        return cos_theta  # size=(B,Classnum,1)

# AMSoftmax 层的pytorch实现，两个重要参数 scale，margin（不同难度和量级的数据对应不同的最优参数）
# 原始实现caffe->https://github.com/happynear/AMSoftmax
class AMSoftmax(nn.Module):
    def __init__(self, scale, margin, is_amp=False):
        super(AMSoftmax, self).__init__()
        self.scale = scale
        self.margin = margin
        self.is_amp = is_amp
    def forward(self, input, target):
        # self.it += 1
        cos_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index1 = ~index


        output = cos_theta * 1.0  # size=(B,Classnum)
        index = index.bool()
        output[index] = output[index] - self.margin
        if self.is_amp:
            output[index1] += self.margin
        output = output * self.scale


        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss

class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 1000
        self.beta_min = 5
        self.scale = 0.99


        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).cuda()  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).cuda()  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).cuda()  # n
        self.signs = torch.ones(margin // 2 + 1).cuda()  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            assert target is None
            return input.mm(self.weight)


def myphi(x, m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = cos_theta.data.acos()
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta, phi_theta)
        return output # size=(B,Classnum,2)

class AngleLoss(nn.Module):
    def __init__(self, gamma=0, is_ap=False):
        super(AngleLoss, self).__init__()
        self.is_ap = is_ap
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index1 = ~index

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)
        if self.is_ap:
            output[index1] += cos_theta[index1] * (1.0 + 0) / (1 + self.lamb)
            output[index1] -= phi_theta[index1] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True).clamp(min=1e-12)
    output = torch.div(input, norm)
    return output

class ArcfaceLinear(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size, classnum, s, m, is_pedcc=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.classnum = classnum

        if is_pedcc:
            self.kernel = Parameter(torch.Tensor(self.embedding_size, self.classnum), requires_grad=False)
            map_dict = read_pkl()
            tensor_empty = torch.Tensor([]).cuda()
            for label_index in range(self.classnum):
                tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
            label_40D_tensor = tensor_empty.view(-1, self.embedding_size).permute(1, 0)
            label_40D_tensor = label_40D_tensor.cuda()
            self.kernel.data = label_40D_tensor
        else:
            self.kernel = Parameter(torch.Tensor(self.embedding_size, self.classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, input, target=None):
        # weights norm
        nB = len(input)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(input,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, target] = cos_theta_m[idx_, target]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

def testimgonebyone(net11,dataset):
    net1 = torch.load(net11)

    map_dict = read_pkl()

    tensor_empty1 = torch.Tensor([])

    #load pre-defined PEDCC center
    for label_index in range(150):
        print(label_index)
        tensor_empty1 = torch.cat((tensor_empty1, map_dict[label_index].float()), 0)
        print(tensor_empty1.numpy().shape)

    corr = 0
    total = 0
    corr_pred = 0
    for (im, label) in tqdm(dataset):  # Store the feature of training data
        if torch.cuda.is_available():
            label = label
            im = im.cuda()  # (bs, 3, h, w)
            label = label.cuda()  # (bs, h, w)

        label1 = label.cpu().numpy()

        output1 = net1(im)[2].cpu().detach().numpy()
        # output2 = net2(im)[2].cpu().detach().numpy()
        # output3 = net3(im)[2].cpu().detach().numpy()
        # output4 = net4(im)[2].cpu().detach().numpy()
        # output5 = net5(im)[2].cpu().detach().numpy()
        # print(label1.max())

        pred1 = np.dot(output1, tensor_empty1.numpy().transpose())
        # pred1 = toloss(torch.Tensor(pred1), torch.LongTensor([[1]]))
        # pred1 = tosoftmax(pred1)

        # pred2 = np.dot(output2, tensor_empty1.numpy().transpose())
        # pred2 = toloss(torch.Tensor(pred2), torch.LongTensor([[pred2.argmax()]]))
        # pred2 = criterion(pred2, pred2.argmax())
        # pred2 = tosoftmax(pred2)
        # pred3 = np.dot(output3, tensor_empty1.numpy().transpose())
        # pred4 = np.dot(output4, tensor_empty1.numpy().transpose())
        # pred5 = np.dot(output5, tensor_empty1.numpy().transpose())


        pred22 = pred1.argmax()
        # print(pred22)
        corr+=1
        if (label1.max() == pred22):
            corr_pred += 1

        # print(a_norm)
        # print(b_norm)
        # print("------------------------------------------------------")

        # pred22 = pred2.argmax()
        # # print(pred22)
        # corr+=1
        # if (label1.max() == pred22):
        #     corr_pred += 1



###############################两个网络的预测######################################################################
        # if(pred1.max() > pred2.max() ):
        #     corr += 1
        #     pred11 = pred1.argmax()
        #     # print(pred11)
        #     if (label1.max()==pred11):
        #         corr_pred += 1
        #
        # if (pred1.max() < pred2.max()):
        #     corr += 1
        #     pred22 = pred2.argmax()+50
        #     # print(pred22)
        #     if (label1.max() == pred22):
        #         corr_pred += 1

###############################三个网络的预测######################################################################
        # if((pred1.max() > pred2.max()) and (pred1.max() > pred3.max())):
        #     corr+=1
        #     pred11 = pred1.argmax()
        #     if (label1.max()==pred11):
        #         corr_pred += 1
        # if ((pred2.max() > pred1.max()) and (pred2.max() > pred3.max())):
        #     corr += 1
        #     pred22 = pred2.argmax()+40
        #     if (label1.max() == pred22):
        #         corr_pred += 1
        # if ((pred3.max() > pred1.max()) and (pred3.max() > pred2.max())):
        #     corr += 1
        #     pred33 = pred3.argmax()+70
        #     if (label1.max() == pred33):
        #         corr_pred += 1



# ###############################五个网络的预测######################################################################
#         if((pred1.max() > pred2.max()) and (pred1.max() > pred3.max()) and (pred1.max() > pred4.max()) and (pred1.max() > pred5.max()) ):
#             corr+=1
#             pred11 = pred1.argmax()
#             if (label1.max()==pred11):
#                 corr_pred += 1
#         if ((pred2.max() > pred1.max()) and (pred2.max() > pred3.max()) and (pred2.max() > pred4.max()) and (pred2.max() > pred5.max())):
#             corr += 1
#             pred22 = pred2.argmax()+20
#             if (label1.max() == pred22):
#                 corr_pred += 1
#         if ((pred3.max() > pred1.max()) and (pred3.max() > pred2.max()) and (pred3.max() > pred4.max()) and (pred3.max() > pred5.max())):
#             corr += 1
#             pred33 = pred3.argmax()+40
#             if (label1.max() == pred33):
#                 corr_pred += 1
#         if ((pred4.max() > pred1.max()) and (pred4.max() > pred2.max()) and (pred4.max() > pred3.max()) and (pred4.max() > pred5.max())):
#             corr += 1
#             pred44 = pred4.argmax()+60
#             if (label1.max() == pred44):
#                 corr_pred += 1
#         if ((pred5.max() > pred1.max()) and (pred5.max() > pred2.max()) and (pred5.max() > pred3.max()) and (pred5.max() > pred4.max())):
#             corr += 1
#             pred55 = pred5.argmax()+80
#             if (label1.max() == pred55):
#                 corr_pred += 1


#############################四个网络的预测############################################################################################
        # if((pred1.max() > pred2.max()) and (pred1.max() > pred3.max()) and (pred1.max() > pred4.max())):
        #     corr+=1
        #     pred11 = pred1.argmax()
        #     if (label1.max()==pred11):
        #         corr_pred += 1
        # if ((pred2.max() > pred1.max()) and (pred2.max() > pred3.max()) and (pred2.max() > pred4.max())):
        #     corr += 1
        #     pred22 = pred2.argmax()+25
        #     if (label1.max() == pred22):
        #         corr_pred += 1
        # if ((pred3.max() > pred1.max()) and (pred3.max() > pred2.max()) and (pred3.max() > pred4.max())):
        #     corr += 1
        #     pred33 = pred3.argmax()+50
        #     if (label1.max() == pred33):
        #         corr_pred += 1
        # if ((pred4.max() > pred1.max()) and (pred4.max() > pred2.max()) and (pred4.max() > pred3.max())):
        #     corr += 1
        #     pred44 = pred4.argmax()+75
        #     if (label1.max() == pred44):
        #         corr_pred += 1
        # print(output1.max())
        # print(output1.argmax(axis=1))
        # print(output2.argmax(axis=1))
        # print('----')
    print(corr)
    print(corr_pred)
        # print(total)
        # print(corr/total)
    # f = open('./50+50_use_net1.txt', 'a+')
    # f.write(str(corr_pred) + '/' + str(corr) +  '\n')
    # f.close()