import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataloader import DatasetFromFolder
from dataloader import TestsetFromFolder
from model import VDSR

from skimage.metrics import peak_signal_noise_ratio
import numpy as np
from skimage import io
from util import *
import os

parser = argparse.ArgumentParser(description='VDSR training parameters')
parser.add_argument('--train_dataset', type=str, default="dataset/train", help='Training dataset path')
parser.add_argument('--validation_dataset', type=str, default="dataset/validation", help='Validation dataset path')
parser.add_argument('--save_root', type=str, default="checkpoint/VDSR", help='Model save path')
parser.add_argument('--crop_size', type=int, default=64, help='Training image crop size')
parser.add_argument('--scale_factor', type=int, default=2, help='Upscale factor')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--nb_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--img_format',  type=str, default="RGB", help=" ['RGB', 'YCbCr', 'Y'] Train Image format")
parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda')
parser.add_argument('--shuffle', action='store_true', default=False, help='data shuffle')
parser.add_argument('--pin_memory', action='store_true', default=True, help='pin_memory')
parser.add_argument('--drop_last', action='store_true', default=True, help='drop_last')


args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

if not os.path.exists(args.save_root):
    os.makedirs(args.save_root)

trainset = DatasetFromFolder(args.train_dataset, scale_factor=args.scale_factor, crop_size=args.crop_size, img_format = args.img_format)
testset = TestsetFromFolder(args.validation_dataset, scale_factor=args.scale_factor, img_format=args.img_format)

trainloader = DataLoader(
                            dataset=trainset, 
                            batch_size = args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers, 
                            pin_memory=args.pin_memory, 
                            drop_last=args.drop_last
                        )

testloader = DataLoader(
                            dataset=testset, 
                            batch_size=1, 
                            shuffle=args.shuffle, 
                            num_workers=args.num_workers, 
                            pin_memory=args.pin_memory, 
                            drop_last=False
                        )

model = VDSR(img_format=args.img_format).to(device)
mse_criterion = nn.MSELoss()

#VDSR
optimizer = optim.Adam(
    model.parameters(),  
    lr=0.0001,  
)

best_psnr = 0  # Initialize best PSNR
best_epoch = 0 # Initialize best epoch

print("\n")
print("*" * 100)
print("START Training VDSR!!")
print("*" * 100)

for epoch in range(args.nb_epochs):

    # Train
    epoch_loss = 0 # Initialize loss
    
    for iteration, batch in enumerate(trainloader):
        input, target = batch[0].to(device), batch[1].to(device)    # Load data to device
        optimizer.zero_grad() # Initialize gradients

        out = model(input) # Inference

        loss = mse_criterion(out, target) # MSE Loss

        loss.backward() # Backpropagation
        
        optimizer.step() # Update weights
        epoch_loss += loss.item() # Add loss


    print("")
    print(f"Epoch {epoch}. \n") # Print training loss
    print(f"MSE loss: {epoch_loss / len(trainloader)}") # Print training loss
    
    # Test
    avg_psnr = 0
    avg_psnr_value = 0
    with torch.no_grad():
        for batch in testloader:
            # calculate the PSNR for only the Y channel of the image
            
            input, target = batch[0].to(device), batch[1].to(device)

            out = model(input) # Inference

            mse_loss = mse_criterion(out, target) # MSE Loss
            
            if args.img_format == 'RGB':
                out = rgb_to_ycbcr(out) # Convert to YCbCr
                target = rgb_to_ycbcr(target) # Convert to YCbCr

                out = out[:, 0, :, :].cpu().numpy() # Only Y channel
                target = target[:, 0, :, :].cpu().numpy() # Only Y channel
            
            elif args.img_format == 'YCbCr':
                out = out[:, 0, :, :].cpu().numpy() # Only Y channel
                target = target[:, 0, :, :].cpu().numpy() # Only Y channel
            
            elif args.img_format == 'Y':
                out = out.cpu().numpy() # Only Y channel
                target = target.cpu().numpy() # Only Y channel
            
            else:
                print("Image format not supported")
                break

            psnr_value = peak_signal_noise_ratio(out, target)
            avg_psnr_value += psnr_value                        
            
    avg_psnr_value /= len(testloader)
        
    # Print Results
    print(f"PSNR: {avg_psnr_value:.2f} dB")
    
    # Save best model
    if avg_psnr_value > best_psnr:  
        best_psnr = avg_psnr_value
        best_epoch = epoch
        torch.save(model, args.save_root + "/best.pth")  # Save best model
        print("")
        print(f"New best model saved at epoch {epoch} with PSNR: {avg_psnr_value:.2f} dB.\n")

    # Save model
    torch.save(model, args.save_root + f"/VDSR_epoch{epoch}.pth")

print("*" * 100)
print("Training complete. \n")    
print(f"Best model was at epoch {best_epoch} with PSNR: {best_psnr:.2f} dB.")
print("*" * 100)