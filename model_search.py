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
    x = torch.transpose(x, 1, 2).contiguous()
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
        xtemp = x[:, :dim_2 // self.k, :, :]  # Take Cin/k first channels
        xtemp2 = x[:, dim_2 // self.k:, :, :]  # Take remaining channels
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))  # Add weight to operation of selected channels.

        # Reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)

        return channel_shuffle(ans, self.k)


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, rate):
        # Absence de reduction, reduction_prev
        super(Cell, self).__init__()
        self.C_out = C
        if C_prev_prev != -1:  # Absence de reduction_prev == False
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)  # Pareil que Darts
        # Dans darts, une seule valeur pour self.preprocess1
        if rate == 2:
            self.preprocess1 = FactorizedReduce(C_prev, C, affine=False)
        elif rate == 0:
            self.preprocess1 = FactorizedIncrease(C_prev, C)
        else:  # Unique cas darts: self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._steps = steps  # Same
        self._multiplier = multiplier  # Same
        self._ops = nn.ModuleList()  # Same

        for i in range(self._steps):  # Same
            for j in range(2 + i):  # Same
                if C_prev_prev != -1 and j != 0:
                    op = MixedOp(C, stride)  # D'ou vient la variable stride ?
                else:
                    stride = 1
                    op = None
                stride = 1
                self._ops.append(op)

        self.ReLUConvBN = ReLUConvBN(self._multiplier * self.C_out, self.C_out, 1, 1, 0)

    def forward(self, s0, s1, weights):
        if s0 is not None:
            s0 = self.preprocess0(s0)  # Input node
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states) if h is not None)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self._multiplier:], dim=1)
        return self.ReLUConvBN(concat_feature)

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        """
        :param steps:
        :param multiplier:
        :param C_prev_prev: number of channels from the previous-previous cell
        :param C_prev: number of channels from the previous cell
        :param C:  number of channels of the current cell
        :param reduction: True if current cell is a reduction cell
        :param reduction_prev: True if previous cell is a reduction cell
        """
        super(Cell, self).__init__()
        self.reduction = reduction # True = is reduction cell

        # The structure of input nodes is fixed and does not participate in the search
        # Decide the structure of the first input nodes
        if reduction_prev:  # If the previous cell is a reduction
            # The operation feature map, with C_prev_prev input channels, will be down-sampled
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:  # If not, we simply connect
            # The first input_nodes is the output of cell k-2, the number of output channels of cell k-2 is C_prev_prev,
            # so the number of input channels for operation here is C_prev_prev
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        # The structure of the second input nodes.
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)  # The second input_nodes is the output of cell k-1
        self._steps = steps  # steps = 4. The connection status of 4 nodes in each cell is to be determined
        self._multiplier = multiplier

        self._ops = nn.ModuleList()  # Build the module's list of operations
        self._bns = nn.ModuleList()

        for i in range(self._steps):  # Go through 4 intermediate nodes to build a mixed operation
            for j in range(2+i):   # For the i-th node, it has j predecessor nodes
                # (the input of each node is composed of the output of the first two cells
                # and the node in front of the current cell)
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)  # op is to build a mix between two nodes
                self._ops.append(op)  # Add the mixed operation of all edges to ops. Len of the list is 2+3+4+5=14[[],[],...,[]]

    # Cell in the calculation process, automatically called during forward propagation
    def forward(self, s0, s1, weights, weights2):
        """
        Implement the equation (1) from Darts paper
        x_j = \sum_{i<j} o^{(i,j)}(x^{i})
        :param s0: Input node from C_prev_prev
        :param s1: Input node from C_prev
        :param weights: Used by operation
        :return:
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]  # Predecessor node of current node
        offset = 0
        # Loop over all 4 intermediates nodes
        for i in range(self._steps): # i and j are reversed in comparison to the math equation in pc-darts paper
            # s is the output of the current node i, find the operation corresponding to i in ops,
            # and then do the corresponding operation on all the predecessor nodes of i (called the forward of MixedOp),
            # and then add the results

            # Difference with darts : we use an additional weights2 attribute in front of the operation.
            s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)  # We handle a 1D (offset + j) array instead of a 2D (j, i) array.
            states.append(s)  # Use the output of the current node i as the input of the next node.
        # states is [s0,s1,b1,b2,b3,b4] b1,b2,b3,b4 are the output of the four intermediate outputs

        # Concat the intermediate output as the output of the current cell
        # dim=1 refers to the channel dimension concat, so the number of output channels becomes 4 times the original
        return torch.cat(states[-self._multiplier:], dim=1)
