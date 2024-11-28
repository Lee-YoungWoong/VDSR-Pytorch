import torch
import torchvision.io as io
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

def rgb_to_ycbcr(tensor):
    """
    Converts an RGB tensor to YCbCr using PIL.
    Args:
        tensor: PyTorch tensor of shape (B, C, H, W) in RGB format.
    Returns:
        A tensor of the same shape in YCbCr format.
    """
    tensor = tensor.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
    ycbcr_list = []
    for img in tensor:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))  # Denormalize to [0, 255]
        ycbcr_img = pil_img.convert("YCbCr")  # RGB -> YCbCr 
        ycbcr_array = np.array(ycbcr_img) / 255.0  # Normalize to [0, 1]
        ycbcr_list.append(ycbcr_array)
    ycbcr_tensor = torch.tensor(np.stack(ycbcr_list)).permute(0, 3, 1, 2)  # (B, C, H, W)
    return ycbcr_tensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img_rgb(filepath):
    return Image.open(filepath).convert('RGB')

def load_img_ycbcr(filepath):
    return Image.open(filepath).convert('YCbCr')

def load_y_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
