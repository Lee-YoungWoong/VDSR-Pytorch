import torch
import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFilter
from util import *


class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, scale_factor, crop_size, img_format):
        super(DatasetFromFolder, self).__init__()
        
        self.scale_factor = scale_factor
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.crop_size = crop_size
        self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size)])
        self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.img_format = img_format
        

    def __getitem__(self, index):
        if self.img_format == 'RGB':
            input = load_img_rgb(self.image_filenames[index])
        elif self.img_format == 'YCbCr':
            input = load_img_ycbcr(self.image_filenames[index])
        elif self.img_format == 'Y':
            input = load_y_img(self.image_filenames[index])
        else:
            raise ValueError("Image format must be 'RGB', 'YCbCr', or 'Y'")
        
        input = self.input_transform(input)
        target = input.copy()

        input_size = input.size  # Get size of the input image

        additional_transforms = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            transforms.Resize((input_size[1]//self.scale_factor, input_size[0]//self.scale_factor), interpolation=InterpolationMode.BICUBIC),  # Downscale the image
            transforms.Resize((input_size[1], input_size[0]), interpolation=InterpolationMode.BICUBIC),  # bicubic upsampling to get back the original size
            transforms.ToTensor()
        ])

        input = additional_transforms(input)
        target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

# calculate the PSNR for only the Y channel of the image
class TestsetFromFolder(Dataset):
    def __init__(self, image_dir, scale_factor, img_format):
        super(TestsetFromFolder, self).__init__()
        
        self.batch_size = 1
        self.scale_factor = scale_factor
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        
        self.scale_factor = scale_factor
        self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.img_format = img_format

    def __getitem__(self, index):
        if self.img_format == 'RGB':
            input = load_img_rgb(self.image_filenames[index])
        elif self.img_format == 'YCbCr':
            input = load_img_ycbcr(self.image_filenames[index])
        elif self.img_format == 'Y':
            input = load_y_img(self.image_filenames[index])
        else:
            raise ValueError("Image format must be 'RGB', 'YCbCr', or 'Y'")
        
        target = input.copy()
        input_size = input.size

        additional_transforms = transforms.Compose([
            transforms.Resize((input_size[1]//self.scale_factor, input_size[0]//self.scale_factor), interpolation=InterpolationMode.BICUBIC),  # Downscale the image
            transforms.Resize((input_size[1], input_size[0]), interpolation=InterpolationMode.BICUBIC),  # Bicubic upsampling to get back the original size
            transforms.ToTensor()
        ])
            
        input = additional_transforms(input)
        target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
