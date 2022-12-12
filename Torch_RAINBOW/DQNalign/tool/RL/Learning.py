from __future__ import division

import numpy as np
import random
import functools
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
# from tensorflow import nn
import torch 
from operator import __add__ 
from functools import reduce
from torch import nn
import torch.nn.functional as F 
from collections import namedtuple
from operator import __add__
import matplotlib.pyplot as plt
import scipy.misc
import os
import math 
#matplotlib inline

BUFFER_SIZE = 500000


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class C51(nn.Module):
    def __init__(self, h_size, env, name, n_atoms, n_actions):
        super(C51, self).__init__()
        self.win_size = env.win_size
        self.input_shape = (2*(self.win_size + 2)*12*3)
        self.num_actions = n_actions
        self.n_atoms = n_atoms

        self.conv1 = SeparableConv2d(3, 8, kernel_size=(3,3), stride=(3,3), )
        self.conv2 = SeparableConv2d(8, 32, kernel_size=(3,3), stride=(1,1),)
        self.conv3 = SeparableConv2d(32, 32, kernel_size=(3,3), stride=(1,1),)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
        self.conv4 = Conv2dSame(32, 64, kernel_size=(3,3), stride=(3,3),)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv5 = nn.Conv2d(64, h_size, kernel_size=(1, int(np.floor(np.ceil(np.floor(np.ceil(2*(self.win_size+2)/3)/3)/3)/2))), stride=(3,3), padding='valid',)
        self.flat = nn.Flatten()
        self.adv1 = NoisyLayer(h_size, self.num_actions*self.n_atoms)
        self.val1 = NoisyLayer(h_size, self.n_atoms)


        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
    def reset_noise(self):
        # pass 
        self.adv1.reset_noise()
        self.val1.reset_noise()
    
    def set_training(self, training: bool):
        self.adv1.set_training(training)
        self.val1.set_training(training)

    def forward(self, x : torch.Tensor, training=True):
        x = x.reshape((-1, 2*(self.win_size+2), 8, 3)).type(torch.float32).transpose(dim0=1, dim1=3)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.pool2(x)
        x = F.leaky_relu(self.conv5(x))
        x = self.flat(x)
        
        adv = self.adv1(x).view(-1, self.num_actions, self.n_atoms)
        val = self.val1(x).view(-1, 1, self.n_atoms)
        final = val + adv - adv.mean(dim=1).view(-1, 1, self.n_atoms)
        return F.softmax(final.view(-1, self.num_actions , self.n_atoms), dim=2).clamp(min=torch.finfo(torch.float32).eps, max= 1.0 - torch.finfo(torch.float32).eps).transpose(dim0=0, dim1=1)



# https://github.com/higgsfield/RL-Adventure/blob/master/common/layers.py
class NoisyLayer(torch.nn.Module):
    def __init__(self, in_features: int,  out_features: int, std_init: float=0.1):
        super(NoisyLayer, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.std_init = std_init 
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range=1/math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range,mu_range)
        self.weight_sigma.data.fill_(self.std_init/math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range,mu_range)
        self.bias_sigma.data.fill_(self.std_init/math.sqrt(self.out_features))

    def set_training(self, training):
        self.training = training

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training: return F.linear(x, self.weight_mu + self.weight_sigma*self.weight_epsilon, self.bias_mu + self.bias_sigma*self.bias_epsilon)
        else: return F.linear(x, self.weight_mu, self.bias_mu)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class SeparableConv2d(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
        ):
        super().__init__()
        
        intermediate_channels = in_channels * depth_multiplier
        self.spatialConv = Conv2dSame(
             in_channels=in_channels,
             out_channels=intermediate_channels,
             kernel_size=kernel_size,
             stride=stride,
             dilation=dilation,
             groups=in_channels,
             bias=bias,
             padding_mode=padding_mode
        )
        self.pointConv = Conv2dSame(
             in_channels=intermediate_channels,
             out_channels=out_channels,
             kernel_size=1,
             stride=1,
             dilation=1,
             bias=bias,
             padding_mode=padding_mode,
        )
        nn.init.kaiming_normal_(self.spatialConv.weight)
        nn.init.kaiming_normal_(self.pointConv.weight)
    def forward(self, x):
        x = self.spatialConv(x)
        x = self.pointConv(x)
        return x

class SumTree(object):
    data_pointer = 0

    def __init__(self, buffer_size=BUFFER_SIZE):

        self.capacity = buffer_size
        self.tree = np.zeros(2*self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)

    def add(self, priority, data):

        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
    
    def update(self, tree_index, priority):
        
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1)//2
            self.tree[tree_index] += change
    
    def get_leaf(self, v):

        parent_index = 0
        while True:
            left_child_index = 2*parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]
        

class per_experience_buffer():
    
    PER_e = 0.01
    PER_a = 0.8
    PER_b = 0.5

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.

    def __init__(self, capacity=BUFFER_SIZE):
        self.tree = SumTree(buffer_size=capacity)

    def add(self, experience):

        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)
    
    def sample(self, n):
        minibatch = []
        b_idx = np.empty((n,), dtype=np.int32)
        priority_segment = self.tree.total_priority/n

        for i in range(n):
            a, b = priority_segment*i, priority_segment*(i+1)
            value = np.random.uniform(a,b)
            index, priority, data = self.tree.get_leaf(value)
            b_idx[i] = index
            minibatch.append([data[0][j] for j in range(5)])
        return b_idx, minibatch
    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class experience_buffer():
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)
        
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def processState(states):
    return np.reshape(states,states.size)

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def copyGraphOp(tfVars):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign(var.value()))
    return op_holder

def copyGraphOp2(tfVars):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign(var.value()))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def copyGraph(op_holder,sess):
    for op in op_holder:
        sess.run(op)
