from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform, BaseDepthTransform_img

import numpy as np

__all__ = ["DepthLSSTransform_PTR"]


@VTRANSFORMS.register_module()
class DepthLSSTransform_PTR(BaseDepthTransform_img):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        dep_mask_ratio: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            print("self.org_nx", self.org_nx)
            if self.org_nx > 1:
                print("using voxel feature!!!")
            else:
                print("using bev feature!!!")
            assert downsample == 2, downsample
            out_channels_ds = out_channels * self.org_nx
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels_ds, out_channels_ds, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels_ds),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels_ds,
                    out_channels_ds,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels_ds),
                nn.ReLU(True),
                nn.Conv2d(out_channels_ds, out_channels_ds, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels_ds),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        self.dep_mask_ratio = dep_mask_ratio
  
    @force_fp32()
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d_org = d.detach()
        if self.dep_mask_ratio > 1e-6:
            d = d.flatten()
            non_zero_idx = torch.nonzero(d)
            sampled_idx = np.random.permutation(non_zero_idx.shape[0])[:int(non_zero_idx.shape[0]*self.dep_mask_ratio)]
            non_zero_depth_idx_masked = non_zero_idx[sampled_idx]
            d[non_zero_depth_idx_masked] = 0.0
            d = d.view(d_org.size())

        d_org = d_org.detach()

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, d_org

    def forward(self, *args, **kwargs):
        x, x_cam, d_org, img_org = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x,x_cam, d_org, img_org
