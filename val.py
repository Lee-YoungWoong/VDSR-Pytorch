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


parser = argparse.ArgumentParser(description='VDSR Validation parameters')
parser.add_argument('--weight', type=str, required=True, help='Path to the saved model checkpoint')
parser.add_argument('--validation_dataset', type=str, default="dataset/validation", help='Validation dataset path')
parser.add_argument('--scale_factor', type=int, default=2, help='Upscale factor')
parser.add_argument('--img_format',  type=str, required=True, help=" ['RGB', 'YCbCr', 'Y'] Train Image format")
parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda')
args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")

avg_psnr_value = 0  # Initialize Average PSNR

print("\n")
print("*" * 100)
print("START Validation VDSR!!")
print("*" * 100)

image_files = [f for f in os.listdir(args.validation_dataset) if os.path.isfile(os.path.join(args.validation_dataset, f))]
for image_file in image_files:
    image_path = os.path.join(args.validation_dataset, image_file)
    
    # Load image
    if args.img_format == 'RGB':
        image = Image.open(image_path).convert('RGB')
        target = np.array(image.convert('YCbCr'))[:, :, 0]  # Convert to YCbCr and take only Y channel
    elif args.img_format == 'YCbCr':
        image = Image.open(image_path).convert('RGB')
        target = Image.open(image_path).convert('YCbCr')
        target = np.array(image)[:, :, 0]
    elif args.img_format == 'Y':
        image = Image.open(image_path).convert('L')
        target = np.array(image)
    else:
        raise ValueError("Image format must be 'RGB', 'YCbCr', or 'Y'")
    
    image_width = image.size[0]
    image_height = image.size[1]       
    
    target = target / 255.0
    target = target.astype(np.float32)
    image = image.resize((int(image.size[0]//args.scale_factor), int(image.size[1]//args.scale_factor)), Image.BICUBIC)  # downscale image using bicubic interpolation
    image = image.resize((int(image_width), int(image_height)), Image.BICUBIC)  # upscale image using bicubic interpolation
    
    img_to_tensor = transforms.ToTensor()
    input = img_to_tensor(image).unsqueeze(0)  # add batch dimension


    if args.weight.endswith('.pth'):
        model = torch.load(args.weight, map_location=device)   # Load model

        input = input.to(device)  # Move input to device
        
        out = model(input) # Inference
        
        if args.img_format == 'RGB':            
            out = out.detach()
            out = rgb_to_ycbcr(out) # Convert to YCbCr        
            out = out[:, 0, :, :].cpu().numpy() # Only Y channel
            out = out[0] # remove batch dimension
            out = out.astype(np.float32) # Convert to float32
            

        elif args.img_format == 'YCbCr':               
            out = out[:, 0, :, :].detach().cpu().numpy()  # Only Y channel
            out = out[0] # remove batch dimension
            out = out.astype(np.float32) # Convert to float32

        elif args.img_format == 'Y':    
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

        elif args.img_format == 'Y':
            output = out.squeeze(0)  # remove batch dimension
            output = (output * 255).clip(0, 255).astype(np.uint8)
            output = Image.fromarray(output[0]) 
            out = np.array(output) / 255.0
            out = out.astype(np.float32)
            
        else:
            print("Image format not supported")
        
    else:
        raise ValueError("Unsupported weight file format. Supported formats are pth, onnx")
    
    psnr_value = peak_signal_noise_ratio(out, target)
    avg_psnr_value += psnr_value
    print(f"{image_file} : {psnr_value:.2f} dB")

avg_psnr_value = avg_psnr_value / len(image_files)  # Calculate average PSNR

# Print Results
print("*" * 100)
print("Validation complete. \n")    
print(f"Validation Average PSNR: {avg_psnr_value:.2f} dB.")
print("*" * 100)