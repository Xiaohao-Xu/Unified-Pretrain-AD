from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser","RenderNet"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        input_cat = torch.cat(inputs, dim=1)
        return super().forward(input_cat)


@FUSERS.register_module()
class RenderNet(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
        )
        self.bev_head = nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False)
        self.pers_head = nn.Conv2d(80, 4, 1, padding=0, bias=False)

    def forward(self, inputs: List[torch.Tensor], pers_input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        bev_out = self.bev_head(torch.cat(inputs, dim=1))
        pers_out = self.pers_head(pers_input)
        return bev_out, pers_out
