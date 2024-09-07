import numpy as np
import torch

class MaskGenerator:
    """MaskGenerator
    Randomly generate a binary mask for an input image to condact masked image modeling.
    Args:
        input_size (tuple):The spatial dimension of the input image.
        mask_patch_size (int): The patch size of the randomized mask.
        model_patch_size (int): The patch size of the backbone model (e.g., SwinTransformer).
        mask_ratio (int): The masking ratio.
    """
    def __init__(self, input_size=(256,704), mask_patch_size=8, model_patch_size=1, mask_ratio=0.5):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size[0] % self.mask_patch_size == 0
        assert self.input_size[1] % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size_0 = self.input_size[0] // self.mask_patch_size
        self.rand_size_1 = self.input_size[1] // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size_0 * self.rand_size_1
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size_0, self.rand_size_1))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        return mask

def Denormalize(x):
    """denormalize
    Denormalize the tensor of an image to its original RGB distribution.
    Args:
        x (tensor): The tensor of an image.
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean


