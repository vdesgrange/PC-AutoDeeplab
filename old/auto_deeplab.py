import numpy as np
import model_search
from genotypes import PRIMITIVES
from genotypes import Genotype
from decoding_formulas import Decoder
import torch.nn.functional as F
from operations import *


class AutoDeeplab(nn.Module):
    def __init__(self, num_classes, num_layers, criterion, num_channel=40, multiplier=5, step=5, crop_size=None,
                 cell=model_search.Cell):
        super(AutoDeeplab, self).__init__()

        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier  # 8
        # self._multiplier = multiplier
        # self._num_channel = num_channel
        # self._crop_size = crop_size
        self._criterion = criterion
        self._arch_param_names = ["alphas", "betas"]  # ["alphas_cell", "alphas_network"]
        self._initialize_alphas_betas()
        f_initial = int(self._filter_multiplier)  # 8
        half_f_initial = int(f_initial / 2)  # 4

        self.stem0 = nn.Sequential(  # Go from 3 channels images to 64 channels used in step structure. Downsample 2
            nn.Conv2d(3, half_f_initial * self._block_multiplier, 3, stride=2, padding=1), # 64
            nn.BatchNorm2d(half_f_initial * self._block_multiplier), # 64
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(  # Standardization
            nn.Conv2d(half_f_initial * self._block_multiplier, half_f_initial * self._block_multiplier, 3, padding=1),
            nn.BatchNorm2d(half_f_initial * self._block_multiplier),
            nn.ReLU()
        )
        self.stem2 = nn.Sequential(  # Second operation reducing spatial resolution by factor of 2
            nn.Conv2d(half_f_initial * self._block_multiplier, f_initial * self._block_multiplier, 3, stride=2, padding=1),
            nn.BatchNorm2d(f_initial * self._block_multiplier),
            nn.ReLU()
        )

        C_prev_prev = half_f_initial * self._block_multiplier # 64
        C_prev = f_initial * self._block_multiplier # 128
        for i in range(self._num_layers):
            # def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, rate) : rate = 0 , 1, 2  reduce rate
            if i == 0:
                cell1 = cell(self._step, self._multiplier, -1, C_prev, self._num_channel, 1)
                cell2 = cell(self._step, self._multiplier, -1, C_prev, self._num_channel * 2, 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1:
                cell1_1 = cell(self._step, self._multiplier, C_prev, self._num_channel, self._num_channel, 1)
                cell1_2 = cell(self._step, self._multiplier, C_prev, self._num_channel * 2, self._num_channel, 0)

                cell2_1 = cell(self._step, self._multiplier, -1, self._num_channel, self._num_channel * 2, 2)
                cell2_2 = cell(self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 2, 1)

                cell3 = cell(self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 4, 2)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell3]
            elif i == 2:
                cell1_1 = cell(self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell(self._step, self._multiplier, self._num_channel, self._num_channel * 2,
                               self._num_channel, 0)

                cell2_1 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel,
                               self._num_channel * 2, 2)
                cell2_2 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2,
                               self._num_channel * 2, 1)
                cell2_3 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4,
                               self._num_channel * 2, 0)

                cell3_1 = cell(self._step, self._multiplier, -1, self._num_channel * 2, self._num_channel * 4, 2)
                cell3_2 = cell(self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 4, 1)

                cell4 = cell(self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 8, 2)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell4]
            elif i == 3:
                cell1_1 = cell(self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell(self._step, self._multiplier, self._num_channel, self._num_channel * 2,
                               self._num_channel, 0)

                cell2_1 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel,
                               self._num_channel * 2, 2)
                cell2_2 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2,
                               self._num_channel * 2, 1)
                cell2_3 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4,
                               self._num_channel * 2, 0)

                cell3_1 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 2,
                               self._num_channel * 4, 2)
                cell3_2 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 4,
                               self._num_channel * 4, 1)
                cell3_3 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 8,
                               self._num_channel * 4, 0)

                cell4_1 = cell(self._step, self._multiplier, -1, self._num_channel * 4, self._num_channel * 8, 2)
                cell4_2 = cell(self._step, self._multiplier, -1, self._num_channel * 8, self._num_channel * 8, 1)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell3_3]
                self.cells += [cell4_1]
                self.cells += [cell4_2]
            else:
                cell1_1 = cell(self._step, self._multiplier, self._num_channel, self._num_channel, self._num_channel, 1)
                cell1_2 = cell(self._step, self._multiplier, self._num_channel, self._num_channel * 2,
                               self._num_channel, 0)

                cell2_1 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel,
                               self._num_channel * 2, 2)
                cell2_2 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 2,
                               self._num_channel * 2, 1)
                cell2_3 = cell(self._step, self._multiplier, self._num_channel * 2, self._num_channel * 4,
                               self._num_channel * 2, 0)

                cell3_1 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 2,
                               self._num_channel * 4, 2)
                cell3_2 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 4,
                               self._num_channel * 4, 1)
                cell3_3 = cell(self._step, self._multiplier, self._num_channel * 4, self._num_channel * 8,
                               self._num_channel * 4, 0)

                cell4_1 = cell(self._step, self._multiplier, self._num_channel * 8, self._num_channel * 4,
                               self._num_channel * 8, 2)
                cell4_2 = cell(self._step, self._multiplier, self._num_channel * 8, self._num_channel * 8,
                               self._num_channel * 8, 1)

                self.cells += [cell1_1]
                self.cells += [cell1_2]
                self.cells += [cell2_1]
                self.cells += [cell2_2]
                self.cells += [cell2_3]
                self.cells += [cell3_1]
                self.cells += [cell3_2]
                self.cells += [cell3_3]
                self.cells += [cell4_1]
                self.cells += [cell4_2]

        # Todo. Review ASPP implementation as suggested
        self.aspp_4 = nn.Sequential(
            # Atrous rate = 96 / 4 = 24, as in the paper.
            ASPP(self._num_channel, 24, 24, self._num_classes)
        )

        self.aspp_8 = nn.Sequential(
            # Atrous rate = 96 / 8 = 12
            ASPP(self._num_channel * 2, 12, 12, self._num_classes) # Number of filters remains = B x F x s/4
        )
        self.aspp_16 = nn.Sequential(
            ASPP(self._num_channel * 4, 6, 6, self._num_classes) # Atrous rate = 96 / 16 = 6
        )
        self.aspp_32 = nn.Sequential(
            ASPP(self._num_channel * 8, 3, 3, self._num_classes) # Atrous rate = 96 / 32 = 3
        )

    def forward(self, x):
        # Important : no need to store all data in these lists, in the end the algorithm only use 2 latest items.
        self.level_4 = []  # Downsample by factor 4
        self.level_8 = []  # Downsample by factor 8
        self.level_16 = []  # Downsample by factor 16
        self.level_32 = []  # Downsample by factor 32
        count = 0

        # === Initialisation ===
        # Initialisation of first stem layer downsample by 4.
        temp = self.stem0(x)  # 3 to 64 channels
        temp = self.stem1(temp)  # Spacial resolution reduction by factor 2. Downsample 2
        self.level_4.append(self.stem2(temp))  # Spacial resolution reduction by factor 2. Downsample 4

        weight_network = torch.randn(self._num_layers, 4, 3).cuda()

        # === Softmax on alphas and betas ===
        # Why does NoamRosenberg implementation changed the weight of beta ?
        # Why [0][1],[0][2] and no [0][0] ?
        if torch.cuda.device_count() > 1:
            img_device = torch.device('cuda', x.get_device())
            normalized_alphas = F.softmax(self.alphas.to(device=img_device), dim=-1)

            for layer in range(len(self.betas)):
                if layer == 0:
                    weight_network[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)

                elif layer == 1:
                    weight_network[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    weight_network[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)

                elif layer == 2:
                    weight_network[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    weight_network[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    weight_network[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                else:
                    weight_network[layer][0][1:] = F.softmax(self.betas[layer][0][1:].to(device=img_device), dim=-1) * (2/3)
                    weight_network[layer][1] = F.softmax(self.betas[layer][1].to(device=img_device), dim=-1)
                    weight_network[layer][2] = F.softmax(self.betas[layer][2].to(device=img_device), dim=-1)
                    weight_network[layer][3][:2] = F.softmax(self.betas[layer][3][:1].to(device=img_device), dim=-1) * (2/3)
        else:
            normalized_alphas = F.softmax(self.alphas, dim=-1) # normalized alphas (cell). original
            # weight_network = F.softmax(self.betas, dim=-1)  # normalized beta (outer network)

            for layer in range(len(self.betas)):
                if layer == 0:
                    weight_network[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)

                elif layer == 1:
                    weight_network[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    weight_network[layer][1] = F.softmax(self.betas[layer][1], dim=-1)

                elif layer == 2:
                    weight_network[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    weight_network[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    weight_network[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                else:
                    weight_network[layer][0][1:] = F.softmax(self.betas[layer][0][1:], dim=-1) * (2/3)
                    weight_network[layer][1] = F.softmax(self.betas[layer][1], dim=-1)
                    weight_network[layer][2] = F.softmax(self.betas[layer][2], dim=-1)
                    weight_network[layer][3][:2] = F.softmax(self.betas[layer][3][:2], dim=-1) * (2/3)

        # === Stem layers ===
        # Compute each possible stem layers of the network level search space.
        # See Figure 1 from AutoDeepLab paper arXiv:1901.02985v2
        for layer in range(self._num_layers):  # For each layer of network level search space
            if layer == 0:  # First layer: with reduction factor 4, 8;
                # No cell l-2 for levels 4 and 8. Only level 4 usable for level 8.
                # Can only use original layer 0 downsample by factor 4. L4-1.
                # normalized_alphas refer to Alpha (cf. paper)
                level4_new = self.cells[count](None, self.level_4[-1], normalized_alphas)  # No downsample, from L4
                count += 1
                level8_new = self.cells[count](None, self.level_4[-1], normalized_alphas)  # Downsample, from L4
                count += 1
                # Beta (cf. paper) from each node of previous layer, control outer network.
                self.level_4.append(level4_new * self.betas[layer][0][1]) # 2 items
                self.level_8.append(level8_new * self.betas[layer][0][2]) # 1 item

                # No unused stem layers

            elif layer == 1:  # Second layer: with reduction factor 4, 8, 16;
                # No cell l-2 for levels 8 and 16. Only level 8 usable for level 16.
                # No downsample, from L4-2 and L4-1 cells
                level4_new_1 = self.cells[count](self.level_4[-2], self.level_4[-1], normalized_alphas)
                count += 1
                # No downsample, from L4-2 and L8-1 cells
                level4_new_2 = self.cells[count](self.level_4[-2], self.level_8[-1], normalized_alphas)
                count += 1
                level4_new = self.betas[layer][0][1] * level4_new_1 + self.betas[layer][1][0] * level4_new_2

                # Downsample, from L4-1
                level8_new_1 = self.cells[count](None, self.level_4[-1], normalized_alphas)
                count += 1
                # No downsample, from L8-1
                level8_new_2 = self.cells[count](None, self.level_8[-1], normalized_alphas)
                count += 1
                level8_new = self.betas[layer][0][2] * level8_new_1 + self.betas[layer][1][1] * level8_new_2

                # Downsample, from L8
                level16_new = self.cells[count](None, self.level_8[-1], normalized_alphas)
                level16_new = level16_new * self.betas[layer][1][2]
                count += 1

                # Update stem layers list with latest
                self.level_4.append(level4_new) # 3 items
                self.level_8.append(level8_new) # 2 items
                self.level_16.append(level16_new) # 1 item

                # Clean unused stem layer (overall list should never go beyond 3 elements, -2, -1 and new)
                self.level_4.pop(0)

            elif layer == 2:  # Third layer: with reduction factor 4, 8, 16, 32;
                # No cell l-2 for levels 16 and 32. Only level 16 usable for level 16.
                level4_new_1 = self.cells[count](self.level_4[-2], self.level_4[-1], normalized_alphas)
                count += 1
                level4_new_2 = self.cells[count](self.level_4[-2], self.level_8[-1], normalized_alphas)
                count += 1
                level4_new = self.betas[layer][0][1] * level4_new_1 + self.betas[layer][1][0] * level4_new_2

                level8_new_1 = self.cells[count](self.level_8[-2], self.level_4[-1], normalized_alphas)
                count += 1
                level8_new_2 = self.cells[count](self.level_8[-2], self.level_8[-1], normalized_alphas)
                count += 1

                level8_new_3 = self.cells[count](self.level_8[-2], self.level_16[-1], normalized_alphas)
                count += 1
                level8_new = self.betas[layer][0][2] * level8_new_1 + self.betas[layer][1][1] * level8_new_2 + \
                             self.betas[layer][2][0] * level8_new_3

                level16_new_1 = self.cells[count](None, self.level_8[-1], normalized_alphas)
                count += 1
                level16_new_2 = self.cells[count](None, self.level_16[-1], normalized_alphas)
                count += 1
                level16_new = self.betas[layer][1][2] * level16_new_1 + self.betas[layer][2][1] * level16_new_2

                level32_new = self.cells[count](None, self.level_16[-1], normalized_alphas)
                level32_new = level32_new * self.betas[layer][2][2]
                count += 1

                self.level_4.append(level4_new) # 3 items
                self.level_8.append(level8_new) # 3 items
                self.level_16.append(level16_new) # 2 items
                self.level_32.append(level32_new) # 1 item

                # Clean unused stem layer (overall list should never go beyond 3 elements, -2, -1 and new)
                self.level_4.pop(0) # Back to 2 items
                self.level_8.pop(0) # Back to 2 items

            elif layer == 3:  # Access to all stem layers
                # No cell l-2 for level 32.
                level4_new_1 = self.cells[count](self.level_4[-2], self.level_4[-1], normalized_alphas)
                count += 1
                level4_new_2 = self.cells[count](self.level_4[-2], self.level_8[-1], normalized_alphas)
                count += 1
                level4_new = self.betas[layer][0][1] * level4_new_1 + self.betas[layer][1][0] * level4_new_2

                level8_new_1 = self.cells[count](self.level_8[-2], self.level_4[-1], normalized_alphas)
                count += 1
                level8_new_2 = self.cells[count](self.level_8[-2], self.level_8[-1], normalized_alphas)
                count += 1
                level8_new_3 = self.cells[count](self.level_8[-2], self.level_16[-1], normalized_alphas)
                count += 1
                level8_new = self.betas[layer][0][2] * level8_new_1 + self.betas[layer][1][
                    1] * level8_new_2 + self.betas[layer][2][0] * level8_new_3

                level16_new_1 = self.cells[count](self.level_16[-2], self.level_8[-1], normalized_alphas)
                count += 1
                level16_new_2 = self.cells[count](self.level_16[-2], self.level_16[-1], normalized_alphas)
                count += 1
                level16_new_3 = self.cells[count](self.level_16[-2], self.level_32[-1], normalized_alphas)
                count += 1
                level16_new = self.betas[layer][1][2] * level16_new_1 + self.betas[layer][2][
                    1] * level16_new_2 + self.betas[layer][3][0] * level16_new_3

                level32_new_1 = self.cells[count](None, self.level_16[-1], normalized_alphas)
                count += 1
                level32_new_2 = self.cells[count](None, self.level_32[-1], normalized_alphas)
                count += 1
                level32_new = self.betas[layer][2][2] * level32_new_1 + self.betas[layer][3][1] * level32_new_2

                self.level_4.append(level4_new)  # 3 items
                self.level_8.append(level8_new)  # 3 items
                self.level_16.append(level16_new)  # 3 items
                self.level_32.append(level32_new)  # 2 items

                # Clean unused stem layer (overall list should never go beyond 3 elements, -2, -1 and new)
                self.level_4.pop(0)  # Back to 2 items
                self.level_8.pop(0) # Back to 2 items
                self.level_16.pop(0) # Back to 2 items

            else:  # Full access to cell l-2 and l1 for each level 4, 8, 16 and 32.
                level4_new_1 = self.cells[count](self.level_4[-2], self.level_4[-1], normalized_alphas)
                count += 1
                level4_new_2 = self.cells[count](self.level_4[-2], self.level_8[-1], normalized_alphas)
                count += 1
                level4_new = self.betas[layer][0][1] * level4_new_1 + self.betas[layer][1][0] * level4_new_2

                level8_new_1 = self.cells[count](self.level_8[-2], self.level_4[-1], normalized_alphas)
                count += 1
                level8_new_2 = self.cells[count](self.level_8[-2], self.level_8[-1], normalized_alphas)
                count += 1
                level8_new_3 = self.cells[count](self.level_8[-2], self.level_16[-1], normalized_alphas)
                count += 1
                level8_new = self.betas[layer][0][2] * level8_new_1 + self.betas[layer][1][
                    1] * level8_new_2 + self.betas[layer][2][0] * level8_new_3

                level16_new_1 = self.cells[count](self.level_16[-2], self.level_8[-1], normalized_alphas)
                count += 1
                level16_new_2 = self.cells[count](self.level_16[-2], self.level_16[-1], normalized_alphas)
                count += 1
                level16_new_3 = self.cells[count](self.level_16[-2], self.level_32[-1], normalized_alphas)
                count += 1
                level16_new = self.betas[layer][1][2] * level16_new_1 + self.betas[layer][2][
                    1] * level16_new_2 + self.betas[layer][3][0] * level16_new_3

                level32_new_1 = self.cells[count](self.level_32[-2], self.level_16[-1], normalized_alphas)
                count += 1
                level32_new_2 = self.cells[count](self.level_32[-2], self.level_32[-1], normalized_alphas)
                count += 1
                level32_new = self.betas[layer][2][2] * level32_new_1 + self.betas[layer][3][1] * level32_new_2

                self.level_4.append(level4_new) # 3 items
                self.level_8.append(level8_new) # 3 items
                self.level_16.append(level16_new) # 3 items
                self.level_32.append(level32_new) # 3 items

                # Clean unused stem layer (overall list should never go beyond 3 elements, -2, -1 and new)
                self.level_4.pop(0)  # Back to 2 items
                self.level_8.pop(0) # Back to 2 items
                self.level_16.pop(0) # Back to 2 items
                self.level_32.pop(0) # Back to 2 items

        # Atrous Spatial Pyramid Pooling attached to each spatial resolution at the L-th layer.
        aspp_result_4 = self.aspp_4(self.level_4[-1])
        aspp_result_8 = self.aspp_8(self.level_8[-1])
        aspp_result_16 = self.aspp_16(self.level_16[-1])
        aspp_result_32 = self.aspp_32(self.level_32[-1])

        # Upsampled to the original resolution ...
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)  # Is it a good idea to use x ?
        # upsample = nn.Upsample(size=(self._crop_size, self._crop_size), mode='bilinear', align_corners=True)
        aspp_result_4 = upsample(aspp_result_4)
        aspp_result_8 = upsample(aspp_result_8)
        aspp_result_16 = upsample(aspp_result_16)
        aspp_result_32 = upsample(aspp_result_32)

        # ... before summed to produce the prediction.
        sum_feature_map = aspp_result_4 + aspp_result_8 + aspp_result_16 + aspp_result_32
        return sum_feature_map

    def _initialize_alphas_betas(self):
        """
        Initial scalar Alpha and Beta which controls the outer network level.
        Alpha and Beta are sampled from a standard gaussian (normal) distribution x 0.001
        """
        k = sum(1 for i in range(self._step) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        # Alpha and Beta are sampled from a standard gaussian (normal) distribution x 0.001
        alphas = (1e-3 * torch.randn(k, num_ops)).clone().detach().requires_grad_(True)  # alphas_cell

        # num_layer x num_spatial_levels x num_spatial_connections (up, straight, down)
        betas = (1e-3 * torch.randn(self._num_layers, 4, 3)).clone().detach().requires_grad_(True)  # alphas_network

        self.register_parameter(self._arch_param_names[0], nn.Parameter(alphas))
        self.register_parameter(self._arch_param_names[1], nn.Parameter(betas))
        # self.alphas_network_mask = torch.ones(self._num_layers, 4, 3) # Useless ? Not used anywhere

    def decode_network(self):
        """
        Process network architecture operations and weights associated.
        Goal is to find the path with the maximum probability from start to end.
        Use classic Viterbi algorithm.
        """
        best_result = []
        max_prop = 0

        def _parse(weight_network, layer, curr_value, curr_result, last):
            nonlocal best_result  # Variable from outer scope
            nonlocal max_prop  # Variable from outer scope

            if (layer == self._num_layers) and (max_prop < curr_value):
                # print (curr_result)
                best_result = curr_result[:]
                max_prop = curr_value
                return

            if layer == 0:
                print('begin0')
                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    print('end0-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

            elif layer == 1:
                print('begin1')

                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    print ('end1-1')

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                num = 1
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num, 2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()

            elif layer == 2:
                print ('begin2')

                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    print ('end2-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                num = 1
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num, 2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()

                num = 2
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num, 2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()
            else:

                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                num = 1
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num, 2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()

                num = 2
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num, 2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()

                num = 3
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

        network_weight = F.softmax(self.betas, dim=-1) * 5
        network_weight = network_weight.data.cpu().numpy()
        _parse(network_weight, 0, 1, [], 0)
        print (max_prop)
        return best_result

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
        decoder = Decoder(self.alphas, self._block_multiplier, self._step)
        return decoder.genotype_decode()

    def _loss(self, input, target):
        logits = self(input)  # Loss criteria to handle
        return self._criterion(logits, target)


def main():
    model = AutoDeeplab(5, 12, None)  # 7, 12, None
    x = torch.tensor(torch.ones(4, 3, 224, 224))
    result = model.decode_network()
    print(result)
    print(model.genotype())
    # x = x.cuda()
    # y = model (x)
    # print (model.arch_parameters ())
    # print (y.size())


if __name__ == '__main__':
    main()
