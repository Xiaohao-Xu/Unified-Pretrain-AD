import torch
import torch.nn.functional as F

def RenderToRGBD(voxel_feat_org,near=1, far=80):
    """RenderToRGBD
    Render voxelized feature embedding into projected depth and rgb maps.
    Args:
        voxel_feat_org (tensor): The feature voxel for rendering.
        near (int): The nearest rendering depth position.
        far (int): The farthest rendering depth position.
    """
    voxel_channel_num = voxel_feat_org.shape[1]

    voxel_rgb, voxel_depth = torch.split(voxel_feat_org, [voxel_channel_num//4*3,voxel_channel_num//4], dim = 1)
    voxel_feat = voxel_depth

    voxel_rgb_sig = torch.sigmoid(voxel_rgb.permute(0,2,3,1).float())
    rgb = torch.reshape(voxel_rgb_sig,(voxel_rgb_sig.size()[0]*voxel_rgb_sig.size()[1]*voxel_rgb_sig.size()[2],voxel_rgb_sig.size()[-1]//3,3))

    voxel_feat = voxel_feat.permute(0,2,3,1)
    raw = torch.reshape(voxel_feat,(voxel_feat.size()[0]*voxel_feat.size()[1]*voxel_feat.size()[2],voxel_feat.size()[-1]))
    device = voxel_feat.device
    N_samples = raw.size()[-1]
    near, far = near * torch.ones_like(raw[...,:1]), far * torch.ones_like(raw[...,:1])
    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.reshape([raw.shape[0], N_samples])
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]

    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(device)], -1)
    alpha = raw2alpha(raw, dists).to(device) 
    trans = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    weights = alpha * trans
    depth_map = torch.sum(weights * z_vals, -1)
    depth_map_opt = torch.reshape(depth_map,(voxel_feat.size()[0],voxel_feat.size()[1],voxel_feat.size()[2])).unsqueeze(1)

    rgb_map = torch.sum(weights[...,None]*rgb,-2)
    rgb_map_opt = torch.reshape(rgb_map,(voxel_rgb_sig.size()[0],voxel_rgb_sig.size()[1],voxel_rgb_sig.size()[2],3)).permute(0,3,1,2)

    render_out = {'rendered_depth':depth_map_opt,  'rendered_rgb':rgb_map_opt} 
    return render_out


def RenderToD(voxel_feat_org,near=0, far=40):
    """RenderToD
    Render voxelized feature embedding into projected depth map.
    Args:
        voxel_feat_org (tensor): The feature voxel for rendering.
        near (int): The nearest rendering depth position.
        far (int): The farthest rendering depth position.
    """
    voxel_feat = voxel_feat_org
    voxel_feat = voxel_feat.permute(0,2,3,1)
    raw = torch.reshape(voxel_feat,(voxel_feat.size()[0]*voxel_feat.size()[1]*voxel_feat.size()[2],voxel_feat.size()[-1])) 
    device = voxel_feat.device
    N_samples = raw.size()[-1]
    near, far = near * torch.ones_like(raw[...,:1]), far * torch.ones_like(raw[...,:1])
    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.reshape([raw.shape[0], N_samples])
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(device)], -1)
    alpha = raw2alpha(raw, dists).to(device)  
    trans = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * trans
    depth_map = torch.sum(weights * z_vals, -1)
    depth_map_opt = torch.reshape(depth_map,(voxel_feat.size()[0],voxel_feat.size()[1],voxel_feat.size()[2])).unsqueeze(1)
    return depth_map_opt 


