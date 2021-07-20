import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.k = 4  # Proportion of selected channels is 1/k
        self.mp = nn.MaxPool2d(2,2)  # Note: this MaxPool operation wasn't mention in the PC-Darts paper.

        for primitive in PRIMITIVES:  # There are 8 operations in PRIMITIVES for PC-Darts.
            op = OPS[primitive](C // self.k, stride, False, False)  # Get operation from name.
            if 'pool' in primitive:  # Add a batch normalization to the pooling operation
                # Number of channels: floor division of C (total number of channels) by k (channel proportion)
                op = nn.Sequential(op, nn.BatchNorm2d(C // self.k, affine=False))
            self._ops.append(op)  # Put these ops in the pre-defined module list

    def forward(self, x, weights):
        # channel proportion k=4
        if x is not None:
            C_dim = x.shape[1]
            xtemp_a = x[:, :C_dim//self.k, :, :]  # Selected channels
            xtemp_b = x[:, C_dim//self.k:, :, :]  # Remaining channels

            # 1 - Apply operation to selected channels
            xtemp_c = sum(w * op(xtemp_a) for w, op in zip(weights, self._ops))

            # 2 - Add the other channels
            # reduction cell needs pooling before concat
            if xtemp_c.shape[2] == x.shape[2]:
                ans = torch.cat([xtemp_c, xtemp_b], dim=1)
            else:
                ans = torch.cat([xtemp_c, self.mp(xtemp_b)], dim=1)

            ans = channel_shuffle(ans, self.k)

        else:
            ans = sum(w * op(x) for w, op in zip(weights, self._ops))
        # ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        # except channel shuffle, channel shift also works
        return ans


class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_fmultiplier_down, prev_fmultiplier_same, prev_fmultiplier_up,
                 filter_multiplier):
        super(Cell, self).__init__()

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.block_multiplier = block_multiplier

        self.C_prev_prev = int(prev_prev_fmultiplier * block_multiplier)
        self._prev_fmultiplier_same = prev_fmultiplier_same

        if prev_fmultiplier_down is not None:
            self.C_prev_down = int(prev_fmultiplier_down * block_multiplier)
            self.preprocess_down = ReLUConvBN(self.C_prev_down, self.C_out, 1, 1, 0, affine=False)

        if prev_fmultiplier_same is not None:
            self.C_prev_same = int(prev_fmultiplier_same * block_multiplier)
            self.preprocess_same = ReLUConvBN(self.C_prev_same, self.C_out, 1, 1, 0, affine=False)

        if prev_fmultiplier_up is not None:
            self.C_prev_up = int(prev_fmultiplier_up * block_multiplier)
            self.preprocess_up = ReLUConvBN(self.C_prev_up, self.C_out, 1, 1, 0, affine=False)

        if prev_prev_fmultiplier != -1:
            self.pre_preprocess = ReLUConvBN(self.C_prev_prev, self.C_out, 1, 1, 0, affine=False)

        self._steps = steps
        self._ops = nn.ModuleList()

        for i in range(self._steps):  # Go through 4 intermediate nodes to build a mixed operation
            for j in range(2 + i):  # For the i-th node, it has j predecessor nodes
                # (the input of each node is composed of the output of the first two cells
                # and the node in front of the current cell)
                stride = 1
                op = MixedOp(self.C_out, stride)  # op is to build a mix between two nodes
                # if prev_prev_fmultiplier == -1 and j == 0:
                #     op = None
                # else:
                #   op = MixedOp(self.C_out, stride)
                self._ops.append(op)  # Add the mixed operation of all edges to ops. List length is 2+3+4+5=14[[],[],...,[]]

        self._initialize_weights()

    def scale_dimension(self, dim, scale):
        assert isinstance(dim, int)
        return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 else int(dim * scale)

    def prev_feature_resize(self, prev_feature, mode):
        if mode == 'down':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 0.5)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 0.5)
        elif mode == 'up':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 2)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 2)

        return F.interpolate(prev_feature, (feature_size_h, feature_size_w), mode='bilinear', align_corners=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, s0, s1_down, s1_same, s1_up, n_alphas, n_gammas):
        """
        :param s0: ante-ante-cell
        :param s1_down: previous cell coming from bottom level
        :param s1_same: previous cell coming from same level
        :param s1_up: previous cell coming from upper level
        :param n_alphas: alpha weight (darts)
        :param n_gammas: beta weight for edge normalization (pc-darts)
        :return:
        """
        if s1_down is not None:
            s1_down = self.prev_feature_resize(s1_down, 'down')
            s1_down = self.preprocess_down(s1_down)
            size_h, size_w = s1_down.shape[2], s1_down.shape[3]

        if s1_same is not None:
            s1_same = self.preprocess_same(s1_same)
            size_h, size_w = s1_same.shape[2], s1_same.shape[3]

        if s1_up is not None:
            s1_up = self.prev_feature_resize(s1_up, 'up')
            s1_up = self.preprocess_up(s1_up)
            size_h, size_w = s1_up.shape[2], s1_up.shape[3]

        all_states = []

        if s0 is not None:
            s0 = F.interpolate(s0, (size_h, size_w), mode='bilinear', align_corners=True) if (s0.shape[2] != size_h) or (s0.shape[3] != size_w) else s0
            s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
            if s1_down is not None:
                states_down = [s0, s1_down]
                all_states.append(states_down)

            if s1_same is not None:
                states_same = [s0, s1_same]
                all_states.append(states_same)

            if s1_up is not None:
                states_up = [s0, s1_up]
                all_states.append(states_up)
        else:
            if s1_down is not None:
                states_down = [0, s1_down]
                all_states.append(states_down)

            if s1_same is not None:
                states_same = [0, s1_same]
                all_states.append(states_same)

            if s1_up is not None:
                states_up = [0, s1_up]
                all_states.append(states_up)

        final_concates = []
        # Loop over edges coming to this node of the stem (reduction cell, increasing cell and normal cell)
        for states in all_states:
            offset = 0

            # Loop over all 4 intermediates nodes
            for i in range(self._steps):  # i and j are reversed in comparison to the math equation in pc-darts paper
                # find the operation corresponding to i in ops
                # and then do the corresponding operation on all the predecessor nodes of i
                # (called the forward of MixedOp) and then add the results
                new_states = []
                for j, h in enumerate(states):
                    branch_index = offset + j
                    if self._ops[branch_index] is None:
                        continue

                    # Difference with DARTS : we use an additional weights attribute in front of the operation.
                    new_state = self._ops[branch_index](h, n_alphas[branch_index]) * n_gammas[branch_index]
                    new_states.append(new_state)

                # s is the output of the current node i
                s = sum(new_states)
                # We handle a 1D (offset + j) array instead of a 2D (j, i) array.
                offset += len(states)
                # Use the output of the current node i as the input of the next node.
                # states are [s0,s1,b1,b2,b3,b4] b1,b2,b3,b4 are the output of the four intermediate outputs
                states.append(s)

            # Concat the intermediate output as the output of the current cell
            # dim=1 refers to the channel dimension concat, so the number of output channels becomes 4 times the original
            concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
            final_concates.append(concat_feature)

        return final_concates
