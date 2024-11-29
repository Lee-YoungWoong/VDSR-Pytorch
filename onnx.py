import torch
import argparse
from model import VDSR
import os

def main():
    # Argument parser 
    parser = argparse.ArgumentParser(description="Export VDSR model to ONNX format")
    
    parser.add_argument('--load_path', type=str, required=True , help="Path to the saved model checkpoint")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save the exported ONNX model")
    parser.add_argument('--img_format',  type=str, default="RGB", help=" ['RGB', 'YCbCr', 'Y'] Train Image format")
    parser.add_argument('--width', type=int, default=256, help="Width of the input image")
    parser.add_argument('--height', type=int, default=256, help="Height of the input image")
    parser.add_argument('--opset_version', type=int, default=9, help="ONNX opset version")
    parser.add_argument('--verbose', type=bool, default=True, help="Verbose output for onnx export")
    
    args = parser.parse_args()
    
    # Generate save_path from load_path if not provided
    if args.save_path is None:
        base_name = os.path.splitext(os.path.basename(args.load_path))[0]  # Extract base name without extension
        args.save_path = f"{base_name}.onnx"  # Generate save path with .onnx extension
    
    # Check if the save_path directory exists, if not create it
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize the model
    model = VDSR(args.img_format) 
    model = torch.load(args.load_path, map_location=torch.device('cpu'))
    model.eval()
    
    if args.img_format == 'RGB' or args.img_format == 'YCbCr':
        input_channels = 3
    elif args.img_format == 'Y':
        input_channels = 1
    else:
        raise ValueError("Image format must be 'RGB', 'YCbCr', or 'Y'")

    # Create an example input tensor with the given input channel, width, and height
    input_example = torch.randn(1, input_channels, args.height, args.width)
    
    # Export the model to ONNX format
    torch.onnx.export(model, 
                      input_example, 
                      args.save_path, 
                      verbose=args.verbose,
                      opset_version=args.opset_version)

if __name__ == '__main__':
    main()
