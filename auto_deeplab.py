import torch.nn as nn
import cell_level_search
from genotypes import PRIMITIVES
import torch.nn.functional as F
from operations import *
from decoding_formulas import Decoder


class AutoDeeplab(nn.Module):
    def __init__(self, num_classes, num_layers, criterion=None, filter_multiplier=8, block_multiplier=5, step=5, cell=cell_level_search.Cell):
        super(AutoDeeplab, self).__init__()

        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._criterion = criterion
        self._initialize_alphas_betas_gammas()
        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)

        self.stem0 = nn.Sequential(   # Go from 3 channels images to 64 channels used in step structure. Downsample 2
            nn.Conv2d(3, half_f_initial * self._block_multiplier, 3, stride=2, padding=1),
            nn.BatchNorm2d(half_f_initial * self._block_multiplier),
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(  # Standardization
            nn.Conv2d(half_f_initial * self._block_multiplier, half_f_initial * self._block_multiplier, 3, stride=1, padding=1),
            nn.BatchNorm2d(half_f_initial * self._block_multiplier),
            nn.ReLU()
        )
        self.stem2 = nn.Sequential(  # Second operation reducing spatial resolution by factor of 2
            nn.Conv2d(half_f_initial * self._block_multiplier, f_initial * self._block_multiplier, 3, stride=2, padding=1),
            nn.BatchNorm2d(f_initial * self._block_multiplier),
            nn.ReLU()
        )

        # initial_fm = C_initial
        for i in range(self._num_layers):

            if i == 0:
                cell1 = cell(self._step, self._block_multiplier, -1,
                             None, f_initial, None,
                             self._filter_multiplier)
                cell2 = cell(self._step, self._block_multiplier, -1,
                             f_initial, None, None,
                             self._filter_multiplier * 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1 = cell(self._step, self._block_multiplier, f_initial,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier, self._filter_multiplier * 2, None,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, None, None,
                             self._filter_multiplier * 4)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, None, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            elif i == 3:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else:
                cell1 = cell(self._step, self._block_multiplier, self._filter_multiplier,
                             None, self._filter_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier)

                cell2 = cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                             self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                             self._filter_multiplier * 2)

                cell3 = cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                             self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                             self._filter_multiplier * 4)

                cell4 = cell(self._step, self._block_multiplier, self._filter_multiplier * 8,
                             self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                             self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

        self.aspp_4 = nn.Sequential(
            ASPP(self._filter_multiplier * self._block_multiplier, self._num_classes, 24, 24)  # 96 / 4 as in the paper
        )
        self.aspp_8 = nn.Sequential(
            ASPP(self._filter_multiplier * 2 * self._block_multiplier, self._num_classes, 12, 12)  # 96 / 8
        )
        self.aspp_16 = nn.Sequential(
            ASPP(self._filter_multiplier * 4 * self._block_multiplier, self._num_classes, 6, 6)  # 96 / 16
        )
        self.aspp_32 = nn.Sequential(
            ASPP(self._filter_multiplier * 8 * self._block_multiplier, self._num_classes, 3, 3)  # 96 / 32
        )

    def forward(self, x):
        # Important : no need to store all data in these lists, in the end the algorithm only use 2 latest items.
        self.level_4 = [] # Downsample by factor 4
        self.level_8 = [] # Downsample by factor 8
        self.level_16 = [] # Downsample by factor 16
        self.level_32 = [] # Downsample by factor 32

        # === Initialisation ===
        # Initialisation of first stem layer downsample by 4.
        temp = self.stem0(x)  # 3 to 64 channels
        temp = self.stem1(temp)  # Spacial resolution reduction by factor 2. Downsample 2
        self.level_4.append(self.stem2(temp))  # Spacial resolution reduction by factor 2. Downsample 4

        count = 0
        k = sum(1 for i in range(self._step) for n in range(i + 2))
        normalized_betas = torch.randn(self._num_layers, 4, 3).cuda()
        normalized_gammas = torch.zeros(k).cuda()

        # Softmax on alphas (operations), gammas (intermediate nodes) and betas (autodeeplab network grid)
        if torch.cuda.device_count() > 1:
            print('1')
            img_device = torch.device('cuda', x.get_device())
            normalized_alphas = F.softmax(self.alphas.to(device=img_device), dim=-1)

            # Softmax is applied over set of gammas corresponding to intermediate node inputs (2, 3, 4, 5)
            offset = 0

            for i in range(self._step):
                normalized_gammas[offset:offset + i + 2] = F.softmax(self.gammas[offset:offset + i + 2].to(device=img_device), dim=-1)
                offset += (i + 2)
            # normalized_betas[layer][ith node][0 : ➚, 1: ➙, 2 : ➘]
            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                else:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:2].to(device=img_device), dim=-1) * (2/3)
        else:
            # Set the values of each parameters

            # Softmax is applied over the full set of alpha (all operations)
            normalized_alphas = F.softmax(self.alphas, dim=-1)

            # Softmax is applied over set of gammas corresponding to intermediate node inputs (2, 3, 4, 5)
            offset = 0
            for i in range(self._step):
                normalized_gammas[offset:offset + i + 2] = F.softmax(self.gammas[offset:offset + i + 2], dim=-1)
                offset += (i + 2)

            # Set betas
            for layer in range(len(self.betas)):
                if layer == 0:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)

                elif layer == 1:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)

                elif layer == 2:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                else:
                    normalized_betas[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    normalized_betas[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    normalized_betas[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                    normalized_betas[layer][3][:2] = F.softmax(self.betas[layer][3][:2], dim=-1) * (2/3)


        # === Stem layers ===
        # Compute each possible stem layers of the network level search space.
        # See Figure 1 from AutoDeepLab paper arXiv:1901.02985v2
        for layer in range(self._num_layers):  # For each layer of network level search space
            if layer == 0:   # First layer: with reduction factor 4, 8;
                # No cell l-2 for levels 4 and 8. Only level 4 usable for level 8.
                # Can only use original layer 0 downsample by factor 4. L4-1.
                # normalized_alphas refer to Alpha (cf. paper)
                level4_new, = self.cells[count](None, None, self.level_4[-1], None, normalized_alphas, normalized_gammas)
                count += 1
                level8_new, = self.cells[count](None, self.level_4[-1], None, None, normalized_alphas, normalized_gammas)
                count += 1

                # Beta (cf. paper) from each node of previous layer, control outer network.
                self.level_4.append(normalized_betas[layer][0][1] * level4_new)
                self.level_8.append(normalized_betas[layer][0][2] * level8_new)

            elif layer == 1:  # Second layer: with reduction factor 4, 8, 16;
                # No cell l-2 for levels 8 and 16. Only level 8 usable for level 16.
                # No downsample, from L4-2 and L4-1 cells; # No downsample, from L4-2 and L8-1 cells
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas,
                                                               normalized_gammas)
                count += 1
                # No downsample, from L4-2 and L8-1 cells
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2
                # Downsample, from L4-1; # No downsample, from L8-1.
                level8_new_1, level8_new_2 = self.cells[count](None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               None,
                                                               normalized_alphas,
                                                               normalized_gammas)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][2] * level8_new_2

                # Downsample, from L8
                level16_new, = self.cells[count](None,
                                                 self.level_8[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas,
                                                 normalized_gammas)
                level16_new = normalized_betas[layer][1][2] * level16_new
                count += 1

                # Update stem layers list with latest
                self.level_4.append(level4_new)  # 3 items
                self.level_8.append(level8_new)  # 2 items
                self.level_16.append(level16_new)  # 1 item

                # Clean unused stem layer (overall list should never go beyond 3 elements, -2, -1 and new)
                self.level_4.pop(0)

            elif layer == 2:  # Third layer: with reduction factor 4, 8, 16, 32;
                # No cell l-2 for levels 16 and 32. Only level 16 usable for level 16.
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas,
                                                               normalized_gammas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas,
                                                                             normalized_gammas)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][
                    0] * level8_new_3

                level16_new_1, level16_new_2 = self.cells[count](None,
                                                                 self.level_8[-1],
                                                                 self.level_16[-1],
                                                                 None,
                                                                 normalized_alphas,
                                                                 normalized_gammas)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2

                level32_new, = self.cells[count](None,
                                                 self.level_16[-1],
                                                 None,
                                                 None,
                                                 normalized_alphas,
                                                 normalized_gammas)

                level32_new = normalized_betas[layer][2][2] * level32_new
                count += 1

                self.level_4.append(level4_new)  # 3 items
                self.level_8.append(level8_new)  # 3 items
                self.level_16.append(level16_new)  # 2 items
                self.level_32.append(level32_new)  # 1 item

                # Clean unused stem layer (overall list should never go beyond 3 elements, -2, -1 and new)
                self.level_4.pop(0)  # Back to 2 items
                self.level_8.pop(0)  # Back to 2 items

            elif layer == 3:  # Access to all stem layers
                # No cell l-2 for level 32.
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas,
                                                               normalized_gammas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas,
                                                                             normalized_gammas)
                count += 1
                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][
                    0] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count](self.level_16[-2],
                                                                                self.level_8[-1],
                                                                                self.level_16[-1],
                                                                                self.level_32[-1],
                                                                                normalized_alphas,
                                                                                normalized_gammas)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][
                    0] * level16_new_3

                level32_new_1, level32_new_2 = self.cells[count](None,
                                                                 self.level_16[-1],
                                                                 self.level_32[-1],
                                                                 None,
                                                                 normalized_alphas,
                                                                 normalized_gammas)
                count += 1
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2

                self.level_4.append(level4_new)  # 3 items
                self.level_8.append(level8_new)  # 3 items
                self.level_16.append(level16_new)  # 3 items
                self.level_32.append(level32_new)  # 2 items

                # Clean unused stem layer (overall list should never go beyond 3 elements, -2, -1 and new)
                self.level_4.pop(0)  # Back to 2 items
                self.level_8.pop(0) # Back to 2 items
                self.level_16.pop(0) # Back to 2 items

            else:  # Full access to cell l-2 and l1 for each level 4, 8, 16 and 32.
                level4_new_1, level4_new_2 = self.cells[count](self.level_4[-2],
                                                               None,
                                                               self.level_4[-1],
                                                               self.level_8[-1],
                                                               normalized_alphas,
                                                               normalized_gammas)
                count += 1
                level4_new = normalized_betas[layer][0][1] * level4_new_1 + normalized_betas[layer][1][0] * level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](self.level_8[-2],
                                                                             self.level_4[-1],
                                                                             self.level_8[-1],
                                                                             self.level_16[-1],
                                                                             normalized_alphas,
                                                                             normalized_gammas)
                count += 1

                level8_new = normalized_betas[layer][0][2] * level8_new_1 + normalized_betas[layer][1][1] * level8_new_2 + normalized_betas[layer][2][
                    0] * level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count](self.level_16[-2],
                                                                                self.level_8[-1],
                                                                                self.level_16[-1],
                                                                                self.level_32[-1],
                                                                                normalized_alphas,
                                                                                normalized_gammas)
                count += 1
                level16_new = normalized_betas[layer][1][2] * level16_new_1 + normalized_betas[layer][2][1] * level16_new_2 + normalized_betas[layer][3][
                    0] * level16_new_3

                level32_new_1, level32_new_2 = self.cells[count](self.level_32[-2],
                                                                 self.level_16[-1],
                                                                 self.level_32[-1],
                                                                 None,
                                                                 normalized_alphas,
                                                                 normalized_gammas)
                count += 1
                level32_new = normalized_betas[layer][2][2] * level32_new_1 + normalized_betas[layer][3][1] * level32_new_2

                self.level_4.append(level4_new)  # 3 items
                self.level_8.append(level8_new)  # 3 items
                self.level_16.append(level16_new)  # 3 items
                self.level_32.append(level32_new)  # 3 items

                # Clean unused stem layer (overall list should never go beyond 3 elements, -2, -1 and new)
                self.level_4.pop(0)  # Back to 2 items
                self.level_8.pop(0)  # Back to 2 items
                self.level_16.pop(0)  # Back to 2 items
                self.level_32.pop(0)  # Back to 2 items

        # Atrous Spatial Pyramid Pooling attached to each spatial resolution at the L-th layer.
        aspp_result_4 = self.aspp_4(self.level_4[-1])
        aspp_result_8 = self.aspp_8(self.level_8[-1])
        aspp_result_16 = self.aspp_16(self.level_16[-1])
        aspp_result_32 = self.aspp_32(self.level_32[-1])

        # Upsampled to the original resolution ...
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)

        aspp_result_4 = upsample(aspp_result_4)
        aspp_result_8 = upsample(aspp_result_8)
        aspp_result_16 = upsample(aspp_result_16)
        aspp_result_32 = upsample(aspp_result_32)

        # ... before summed to produce the prediction.
        sum_feature_map = aspp_result_4 + aspp_result_8 + aspp_result_16 + aspp_result_32

        return sum_feature_map

    def _initialize_alphas_betas_gammas(self):
        """
        Initial scalar Alpha and Beta which controls the outer network level.
        Alpha and Beta are sampled from a standard gaussian (normal) distribution x 0.001
        """
        k = sum(1 for i in range(self._step) for n in range(2 + i))

        num_ops = len(PRIMITIVES)

        # Alpha and Beta are sampled from a standard gaussian (normal) distribution x 0.001
        # num_input_for_nodes x num_operations
        alphas = (1e-3 * torch.randn(k, num_ops)).clone().detach().requires_grad_(True)

        # num_layer x num_spatial_levels x num_spatial_connections (up, straight, down)
        betas = (1e-3 * torch.randn(self._num_layers, 4, 3)).clone().detach().requires_grad_(True)

        # num_input_for_nodes
        gammas = (1e-3 * torch.randn(k)).clone().detach().requires_grad_(True)

        self._arch_parameters = [
            alphas,
            betas,
            gammas,
        ]
        self._arch_param_names = [
            'alphas',
            'betas',
            'gammas',
        ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters(self):
        # Get architecture parameters ["alphas", "betas"]
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        # Get module parameters, avoid architecture parameters: ["alphas", "betas"]
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def genotype(self):
        """
        Decode discrete cell architecture by first retaining the 2 strongest predecessors for each block,
        and then choose the most likely operator by taking the argmax. (See paper section 4.3).
        """
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)


def main():
    model = AutoDeeplab(7, 12, None)
    x = torch.tensor(torch.ones(4, 3, 224, 224))
    resultdfs = model.decode_dfs()
    resultviterbi = model.decode_viterbi()[0]

    print(resultviterbi)
    print(model.genotype())


if __name__ == '__main__':
    main()
