from __future__ import division

import numpy as np
import random
import functools
import torch 
from operator import __add__ 
from functools import reduce
from torch import nn
import torch.nn.functional as F 
from operator import __add__
import matplotlib.pyplot as plt
import math 

BUFFER_SIZE = 60000


# Main agent network
class C51(nn.Module):
    def __init__(self, h_size, env, name, n_atoms, n_actions):
        super(C51, self).__init__()
        self.win_size = env.win_size
        self.input_shape = (2*(self.win_size + 2)*12*3)
        self.num_actions = n_actions
        self.n_atoms = n_atoms
        kernel_sizes = (3,3)
        conv_padding = reduce(__add__, 
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_sizes[::-1]])
        # self.pad1 = nn.ZeroPad2d(conv_padding)
        self.conv1 = SeparableConv2d(3, 8, kernel_size=(3,3), stride=(3,3),)
        self.conv2 = SeparableConv2d(8, 32, kernel_size=(3,3), stride=(1,1),)
        self.conv3 = SeparableConv2d(32, 32, kernel_size=(3,3), stride=(1,1),)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
        # self.conv4 = SeparableConv2d(32, 64, kernel_size=(3,3), stride=(3,3))
        self.conv4 = Conv2dSame(32, 64, kernel_size=(3,3), stride=(3,3),)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv5 = nn.Conv2d(64, h_size, kernel_size=(1, int(np.floor(np.ceil(np.floor(np.ceil(2*(self.win_size+2)/3)/3)/3)/2))), stride=(3,3), padding='valid',)
        self.flat = nn.Flatten()

        self.adv1 = nn.Linear(h_size, self.num_actions*self.n_atoms)
        self.val1 = nn.Linear(h_size, self.n_atoms)    

    def forward(self, x : torch.Tensor):
        x = torch.reshape(x, (-1, 2*(self.win_size + 2), 8, 3)).type(torch.float32)
        x = torch.transpose(x, dim0=1, dim1=3)
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
        

        # x = self.fc1(x)
        final = F.softmax(final.view(-1, self.num_actions , self.n_atoms), dim=2)
        return torch.transpose(final, dim0=0, dim1=1)




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
# https://gist.github.com/bdsaglam/84b1e1ba848381848ac0a308bfe0d84c
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
        # torch.nn.init.kaiming_normal_(self.spatialConv.weight)
        # torch.nn.init.kaiming_normal_(self.pointConv.weight)
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
