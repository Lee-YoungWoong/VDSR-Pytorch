import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import onnx
import onnxruntime
from PIL import Image
from util import *
from skimage.metrics import peak_signal_noise_ratio
from skimage import io
import os

# Argument parsing
parser = argparse.ArgumentParser(description='SRCNN test parameters')
parser.add_argument('--weight', type=str, required=True, help='Path to the saved model checkpoint')
parser.add_argument('--image', type=str, required=True, help='Path to the input image')
parser.add_argument('--scale_factor', type=int, default=2, help='Upscale factor')
parser.add_argument('--save_root', type=str, default="./result", help='Model Result path')
parser.add_argument('--architecture', type=str, default="955", help='Model architecture must be 3 layers')
parser.add_argument('--padding',  action='store_true', default=False, help='same with output size')
parser.add_argument('--img_format',  type=str, default="RGB", help=" ['RGB', 'YCbCr', 'Y'] Train Image format")
parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda')

args = parser.parse_args()

# Set device
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")

if not os.path.exists(args.save_root):
    os.makedirs(args.save_root)

if len(args.architecture) != 3: 
    raise ValueError("SRCNN Architecture must be 3 layers.")
else:
    j = int(args.architecture[0])
    k = int(args.architecture[1])
    l = int(args.architecture[2])    

if args.padding == True:
    # Load image
    if args.img_format == 'RGB':
        image = Image.open(args.image).convert('RGB')
        target = np.array(image.convert('YCbCr'))[:, :, 0]  # Convert to YCbCr and take only Y channel
    elif args.img_format == 'YCbCr':
        image = Image.open(args.image).convert('RGB')
        target = Image.open(args.image).convert('YCbCr')
        target = np.array(image)[:, :, 0]
    elif args.img_format == 'Y':
        image = Image.open(args.image).convert('L')
        target = np.array(image)
    else:
        raise ValueError("Image format must be 'RGB', 'YCbCr', or 'Y'")
    
    image_width = image.size[0]
    image_height = image.size[1]       
    
    target = target / 255.0
    target = target.astype(np.float32)
    image = image.resize((int(image.size[0]//args.scale_factor), int(image.size[1]//args.scale_factor)), Image.BICUBIC)  # downscale image using bicubic interpolation
    image = image.resize((int(image_width), int(image_height)), Image.BICUBIC)  # upscale image using bicubic interpolation
    
    bicubic_save_path = os.path.join(args.save_root, f"BICUBIC_{os.path.basename(args.image)}")
    image.save(bicubic_save_path)
    print(f"Image saved as {bicubic_save_path}")   
    
    img_to_tensor = transforms.ToTensor()
    input = img_to_tensor(image).unsqueeze(0)  # add batch dimension

else:
    # Load image
    if args.img_format == 'RGB':
        image = Image.open(args.image).convert('RGB')
        target = image.copy()
    elif args.img_format == 'YCbCr':
        image = Image.open(args.image).convert('YCbCr')
        target = image.copy()
    elif args.img_format == 'Y':
        image = Image.open(args.image).convert('L')
        target = image.copy()
    else:
        raise ValueError("Image format must be 'RGB', 'YCbCr', or 'Y'")  
    
    image_width = image.size[0]
    image_height = image.size[1]    
    
    non_padding_size_height = (image_height -(j+k+l) + 3)
    non_padding_size_width = (image_width -(j+k+l) + 3)
    
    target = target.resize((int(non_padding_size_width), int(non_padding_size_height)), Image.BICUBIC)  # downscale image using bicubic interpolation
    
    if args.img_format == 'RGB':
        target = np.array(target.convert('YCbCr'))[:, :, 0]  # Convert to YCbCr and take only Y channel
    elif args.img_format == 'YCbCr':
        target = np.array(target)[:, :, 0]
    elif args.img_format == 'Y':
        target = np.array(target)
    else:
        raise ValueError("Image format must be 'RGB', 'YCbCr', or 'Y'")  
    print(target.shape)
    target = target/255.0
    target = target.astype(np.float32)
    image = image.resize((int(image.size[0]//args.scale_factor), int(image.size[1]//args.scale_factor)), Image.BICUBIC)  # downscale image using bicubic interpolation
    image = image.resize((int(image_width), int(image_height)), Image.BICUBIC)  # upscale image using bicubic interpolation
    
    bicubic_save_path = os.path.join(args.save_root, f"BICUBIC_{os.path.basename(args.image)}")
    image.save(bicubic_save_path)
    print(f"Image saved as {bicubic_save_path}")
    
    img_to_tensor = transforms.ToTensor()
    input = img_to_tensor(image).unsqueeze(0)  # add batch dimension


if args.weight.endswith('.pth'):
    model = torch.load(args.weight, map_location=device)   # Load model

    input = input.to(device)  # Move input to device
    
    out = model(input) # Inference
    
    if args.img_format == 'RGB':
        output = out.detach().cpu().numpy().squeeze(0)  # remove batch dimension
        output = (output * 255).clip(0, 255).astype(np.uint8)  # clip to 0-255
        out_img = transforms.ToPILImage()(output.transpose(1, 2, 0))  # Convert to PIL image
        
        sr_save_path = os.path.join(args.save_root, f"SR_{os.path.basename(args.image)}")
        out_img.save(sr_save_path)
        print(f"Image saved as {sr_save_path}")
        
        out = out.detach()
        out = rgb_to_ycbcr(out) # Convert to YCbCr        
        out = out[:, 0, :, :].cpu().numpy() # Only Y channel
        out = out[0] # remove batch dimension
        out = out.astype(np.float32) # Convert to float32
        

    elif args.img_format == 'YCbCr':
        output = out.permute(0, 2, 3, 1).detach().cpu().numpy()   # remove batch dimension
        rgb_list = [] # List to store RGB images
        for idx, img in enumerate(output): # Iterate over each image
            out_img = Image.fromarray((img * 255).astype(np.uint8), mode="RGB") # Convert to RGB
        
        sr_save_path = os.path.join(args.save_root, f"SR_{os.path.basename(args.image)}")
        out_img.save(sr_save_path)
        print(f"Image saved as {sr_save_path}")
                
        out = out[:, 0, :, :].detach().cpu().numpy()  # Only Y channel
        out = out[0] # remove batch dimension
        out = out.astype(np.float32) # Convert to float32

    elif args.img_format == 'Y':
        output = out.detach().cpu().numpy().squeeze(0).squeeze(0)  # remove batch dimension
        print(output.shape) # (H, W)
        out_img = Image.fromarray((output * 255).astype(np.uint8), mode="L") # Convert to grayscale
        
        sr_save_path = os.path.join(args.save_root, f"SR_{os.path.basename(args.image)}")
        out_img.save(sr_save_path)
        print(f"Image saved as {sr_save_path}")
        
        out = out.detach().cpu().numpy().squeeze(0).squeeze(0) # Only Y channel
        out = out.astype(np.float32) # Convert to float32
        
    else:
        print("Image format not supported")
        
        
elif args.weight.endswith('.onnx'):
    onnx_session = onnxruntime.InferenceSession(args.weight)
    ort_inputs = {onnx_session.get_inputs()[0].name: to_numpy(input)}
    out = onnx_session.run(None, ort_inputs)
    out = np.array(out[0])
    
    if args.img_format == 'RGB':
        output = out.squeeze(0)  # remove batch dimension
        output = (output * 255).clip(0, 255).astype(np.uint8)  
        output = transforms.ToPILImage()(output.transpose(1, 2, 0))
          
        sr_save_path = os.path.join(args.save_root, f"SR_{os.path.basename(args.image)}")
        output.save(sr_save_path)
        print(f"Image saved as {sr_save_path}")
            
        output = output.convert('YCbCr')  # RGB -> YCbCr
        y, cb, cr = output.split()
        out = np.array(y) / 255.0  
        out = out.astype(np.float32)   

    elif args.img_format == 'YCbCr':
        output = out.squeeze(0)  # remove batch dimension
        output = (output * 255).clip(0, 255).astype(np.uint8)  
        output = transforms.ToPILImage()(output.transpose(1, 2, 0))
        y, cb, cr = output.split()
        out = np.array(y) / 255.0  
        out = out.astype(np.float32)
        output = output.convert('RGB')  # YCbCr  
        
        sr_save_path = os.path.join(args.save_root, f"SR_{os.path.basename(args.image)}")
        output.save(sr_save_path)
        print(f"Image saved as {sr_save_path}")

    elif args.img_format == 'Y':
        output = out.squeeze(0)  # remove batch dimension
        output = (output * 255).clip(0, 255).astype(np.uint8)
        output = Image.fromarray(output[0]) 
        
        sr_save_path = os.path.join(args.save_root, f"SR_{os.path.basename(args.image)}")
        output.save(sr_save_path)
        print(f"Image saved as {sr_save_path}")

        out = np.array(output) / 255.0
        out = out.astype(np.float32)
        
    else:
        print("Image format not supported")
    
else:
    raise ValueError("Unsupported weight file format. Supported formats are pth, onnx")

psnr_value = peak_signal_noise_ratio(out, target)

# Print Results
print(f"PSNR: {psnr_value:.2f} dB")

