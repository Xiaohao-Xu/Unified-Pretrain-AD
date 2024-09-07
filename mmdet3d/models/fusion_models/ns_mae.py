from typing import Any, Dict
import numpy as np
import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
    build_loss,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

from .base import Base3DFusionModel

from torchvision.transforms import Resize

from .utils_mim import MaskGenerator, Denormalize
from .utils_render import RenderToRGBD, RenderToD


__all__ = ["NS_MAE"]

@FUSIONMODELS.register_module()
class NS_MAE(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        MIM_flag = True,
        img_mask_size = 8,
        img_mask_ratio = 0.5,
        input_img_size = (256,704),
        lamda_img = 10000,
        lamda_dep_bev = 0.01,
        lamda_dep_per = 0.01,
        early_stop_epoch = 10,
        early_stop_all_depth = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        self.MIM_flag = MIM_flag   
        self.img_mask_size = img_mask_size
        self.input_img_size = input_img_size
        self.img_mask_ratio = img_mask_ratio
        self.mask_gen_train = MaskGenerator(input_size = self.input_img_size,mask_patch_size = self.img_mask_size, mask_ratio = self.img_mask_ratio)
        self.patch_emb = torch.nn.Parameter(torch.zeros(1,3,self.img_mask_size,self.img_mask_size))

        self.L1_criterion = nn.L1Loss()
        self.MSE_criterion = nn.MSELoss()
  
        self.early_stop_all_depth = early_stop_all_depth
        self.early_stop_epoch = early_stop_epoch     
        
        self.lamda_img = lamda_img
        self.lamda_dep_bev = lamda_dep_bev
        self.lamda_dep_per = lamda_dep_per
   

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        img_org = Denormalize(x).permute(0,2,3,1).detach().cpu().numpy()

        if self.MIM_flag ==True:
            rand_mask = self.mask_gen_train.__call__().to(x.device)
            learnable_mask_token = self.patch_emb.repeat_interleave(x.size()[-2]//self.img_mask_size,dim=-2).repeat_interleave(x.size()[-1]//self.img_mask_size,dim=-1).to(x.device)
            x = x * (1 - rand_mask) + learnable_mask_token * rand_mask
            
        
        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            img_org,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x, self.forward_re_dict = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x, feats, coords, sizes, self.forward_re_dict
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):

        ## Send Corruptted Multi-modal Inputs to the Multi-modal Embedding Network  
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature,cam_pers_feat,d_org,img_org = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                feature, feats, coords, sizes, self.forward_re_dict = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            
            features.append(feature)

        if not self.training:
            features = features[::-1]

        ## Perspective-view Rendering
        cam_pers_feat = cam_pers_feat.permute(0,1,2,5,3,4)
        cam_bs, cam_num, cam_dep_num, cam_c_num, cam_h_num, cam_w_num = cam_pers_feat.size()
        cam_pers_feat = cam_pers_feat.reshape(cam_bs*cam_num*cam_dep_num, cam_c_num, cam_h_num, cam_w_num)
        if self.fuser is not None:
            x,cam_pers_dep = self.fuser(features,cam_pers_feat)
            cam_pers_dep = cam_pers_dep.view(cam_bs * cam_num, cam_dep_num *4, cam_h_num, cam_w_num)
            img_org = torch.tensor(img_org).permute(0,3,1,2)
            render_list = RenderToRGBD(cam_pers_dep)
            torch_resize = Resize([img_org.size()[-2]//8,img_org.size()[-1]//8]) 
            img_org = torch_resize(img_org)
            torch_d_resize = Resize([d_org.size()[-2]//8,d_org.size()[-1]//8],interpolation = 0)
            d_org = torch_d_resize(d_org)

            cam_pers_depth = render_list["rendered_depth"]
            cam_pers_img = render_list["rendered_rgb"]
        else:
            assert len(features) == 1, features
            x = features[0]


        ##  Rendering Target Construction of BEV Depth
        gt_feat = self.forward_re_dict["target_masked_non"].squeeze(1).permute(0, 3, 1, 2)  
        gt_feat = F.interpolate(gt_feat, scale_factor=0.125) 
        gt_feat = torch.flip(gt_feat,dims=[1])
        gt_feat = (gt_feat>0).to(torch.long)
        gt_feat = torch.argmax(gt_feat,dim = 1)

        ##  BEV Rendering 
        batch_size = x.shape[0]
        x = RenderToD(x).squeeze(1)

        if self.training:
            
            #  Early Stopping Strategy to Avoid Overfitting of Model, especially Lidar Encoding Path
            if self.epoch >= self.early_stop_epoch:
                for name,param in self.encoders["lidar"]["backbone"].named_parameters():
                    param.require_grad = False
            if self.early_stop_all_depth == True:
                self.lamda_dep_per = 1e-8
                self.lamda_dep_bev = 1e-8

            outputs = {}

            d_org_flatten = d_org.flatten().to(cam_pers_img.device)
            cam_pers_depth_flatten = cam_pers_depth.flatten()

            ## Multi-modal Reconstruction-based Self-supervised Optimization
            losses_dep_per = self.L1_criterion(cam_pers_depth_flatten[d_org_flatten>0],d_org_flatten[d_org_flatten>0])
            losses_img_per = self.MSE_criterion(cam_pers_img.flatten(),img_org.flatten().to(cam_pers_img.device))
            losses_dep_bev = self.L1_criterion(x.flatten()[gt_feat.flatten()>0], gt_feat.flatten()[gt_feat.flatten()>0])  
            losses_all = self.lamda_dep_bev * losses_dep_bev + self.lamda_dep_per * losses_dep_per + self.lamda_img * losses_img_per
            outputs[f"loss"] = losses_all
            return outputs
           
        else:
            outputs = [{} for _ in range(batch_size)]
            return outputs



