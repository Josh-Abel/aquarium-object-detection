import math
import torch
from torch import nn

class AnchorGenerator(nn.Module):

    def __init__(self, sizes, aspect_ratios):
        super().__init__()

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [self.generate_anchors(size, aspect_ratio) for size, aspect_ratio in zip(sizes, aspect_ratios)]

    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(self, scales,aspect_ratios):
        device = torch.device("cuda:0")
        scales = torch.as_tensor(scales).to(device)
        aspect_ratios = torch.as_tensor(aspect_ratios).to(device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()


    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        cell_anchors = self.cell_anchors


        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(0, grid_width).to(device) * stride_width
            shifts_y = torch.arange(0, grid_height).to(device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.shape[-2:]

        device = torch.device("cuda:0")
        strides = [
            [
                torch.empty(()).fill_(image_size[0] // g[0]).to(device),
                torch.empty(()).fill_(image_size[1] // g[1]).to(device),
            ]
            for g in grid_sizes
        ]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors = []
        for _ in range(len(image_list)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors