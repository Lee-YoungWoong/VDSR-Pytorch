import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SRCNN(nn.Module):
    def __init__(self, architecture : str, padding, img_format) -> None:
        super(SRCNN, self).__init__()
        if len(architecture) != 3:
            raise ValueError("SRCNN Architecture must be 3 layers.")
        
        j = int(architecture[0])
        k = int(architecture[1])
        l = int(architecture[2])
        
        if img_format == 'RGB' or img_format == 'YCbCr': 
            if padding:   
                self.conv1 = nn.Conv2d(3, 64, kernel_size=j, padding=(j-1)//2)
                self.conv2 = nn.Conv2d(64, 32, kernel_size=k, padding=(k-1)//2)
                self.conv3 = nn.Conv2d(32, 3, kernel_size=l, padding=(l-1)//2)
            else:
                self.conv1 = nn.Conv2d(3, 64, kernel_size=j)
                self.conv2 = nn.Conv2d(64, 32, kernel_size=k)
                self.conv3 = nn.Conv2d(32, 3, kernel_size=l)
                
        elif img_format == 'Y':
            if padding:   
                self.conv1 = nn.Conv2d(1, 64, kernel_size=j, padding=(j-1)//2)
                self.conv2 = nn.Conv2d(64, 32, kernel_size=k, padding=(k-1)//2)
                self.conv3 = nn.Conv2d(32, 1, kernel_size=l, padding=(l-1)//2)
            else:
                self.conv1 = nn.Conv2d(1, 64, kernel_size=j)
                self.conv2 = nn.Conv2d(64, 32, kernel_size=k)
                self.conv3 = nn.Conv2d(32, 1, kernel_size=l)
        else:
            raise ValueError("Unsupported image format. Use 'RGB' or 'YCbCr'.")
            
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out = torch.clip(out, 0.0, 1.0)

        return out
