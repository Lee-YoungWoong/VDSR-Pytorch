import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class VDSR(nn.Module):
    def __init__(self, img_format) -> None:
        super(VDSR, self).__init__()
        if img_format == 'RGB' or img_format == 'YCbCr': 
            input_channels = 3
        elif img_format == 'Y':
            input_channels = 1
        else:
            raise ValueError('Image format not supported')
           
        self.conv_1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv_2_to_19 = nn.Sequential()
        for i in range(2, 20):
            self.conv_2_to_19.add_module(f"conv_{i}", nn.Conv2d(64, 64, 3, padding=1))
            self.conv_2_to_19.add_module(f"relu_{i}", nn.ReLU())
        self.conv_20 = nn.Conv2d(64, input_channels, 3, padding=1)
        self.init_weights()

    def forward(self, X_in):
        x_in = X_in.clone()
        x = self.conv_1(X_in)
        torch.relu_(x)
        x = self.conv_2_to_19(x)
        x = self.conv_20(x)
        x = torch.add(x, x_in)
        x = torch.clip(x, 0.0, 1.0)
        return x

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
        

        

