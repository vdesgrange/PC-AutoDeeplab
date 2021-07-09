import torch
import torch.nn as nn
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


def channel_shuffle(x, groups):
    """
    :param x: Tensor. Concatenation of selected/masked channels data
    :param groups: Proportion of selected channels is 1/groups
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()  # contiguous makes actual transpose tensor, not only layout change.

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module):
    """
    MixedOp implement channel sampling.
    Selected channels are sent into mixed computation, masked channels by-pass the operations
    """

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)  # Note: this MaxPool operation wasn't mention in the PC-Darts paper.
        self.k = 4  # Proportion of selected channels is 1/k
        for primitive in PRIMITIVES:  # There are 8 operations in PRIMITIVES for PC-Darts.
            op = OPS[primitive](C // self.k, stride,
                                False)  # Get operation from name. Nb of channels: floor division C by k
            if 'pool' in primitive:  # Add a batch normalization to the pooling operation
                op = nn.Sequential(op, nn.BatchNorm2d(C // self.k, affine=False))
            self._ops.append(op)  # Put these ops in the pre-defined module list

    def forward(self, x, weights):
        # Channel proportion k=4
        dim_2 = x.shape[1]
        x_temp = x[:, :dim_2 // self.k, :, :]  # Take Cin/k first channels
        x_temp_2 = x[:, dim_2 // self.k:, :, :]  # Take remaining channels
        y_temp = sum(w * op(x_temp) for w, op in zip(weights, self._ops))  # Add weight to operation of selected channels.

        # Reduction cell needs pooling before concat
        if y_temp.shape[2] == x.shape[2]:
            ans = torch.cat([y_temp, x_temp_2], dim=1)
        else:
            ans = torch.cat([y_temp, self.mp(x_temp_2)], dim=1)

        return channel_shuffle(ans, self.k)


class Cell(nn.Module):
    def __init__(self, steps, block_multiplier, C_prev_prev, C_prev, filter_multiplier, rate):
        super(Cell, self).__init__()

        self.steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()

        self.C_prev_prev = C_prev_prev
        self.C_prev = C_prev
        self.C_out = block_multiplier * filter_multiplier  # C
        self.C_in = block_multiplier * self.C_out

        if self.C_prev_prev != -1:  # Absence de reduction_prev == Falze
            self.preprocess0 = ReLUConvBN(self.C_prev_prev, self.C_out, 1, 1, 0, affine=False)  # Pareil que Darts

        # Dans darts, une seule valeur pour self.preprocess1
        if rate == 2:
            self.preprocess1 = FactorizedReduce(self.C_prev, self.C_out, affine= False)
        elif rate == 0:
            self.preprocess1 = FactorizedIncrease(self.C_prev, self.C_out)
        else :  # Unique cas darts: self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
            self.preprocess1 = ReLUConvBN(self.C_prev, self.C_out, 1, 1, 0, affine=False)

        for i in range(self._steps):  # Same
            for j in range(2+i):  # Same
                stride = 1
                if C_prev_prev != -1 and j != 0:
                    op = MixedOp(self.C_out, stride)
                else:
                    op = None
                self._ops.append(op)

        self.ReLUConvBN = ReLUConvBN(self.C_in, self.C_out, 1, 1, 0)

    def forward(self, s0, s1, alphas):
        states = []

        if s0 is not None:
            s0 = self.preprocess0(s0)  # Input node
            states.append(s0)

        s1 = self.preprocess1(s1)
        states.append(s1)

        offset = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if self._ops[branch_index] is None:
                    continue

                new_state = self._ops[branch_index](h, alphas[branch_index])
                new_states.append(new_state)

            s = sum(new_states)
            #s = sum(self._ops[offset+j](h, alphas[offset+j]) for j, h in enumerate(states) if h is not None)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
        return self.ReLUConvBN(concat_feature)