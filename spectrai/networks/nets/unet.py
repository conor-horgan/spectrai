import torch
from torch import nn 
from spectrai.networks.layers import BasicConv, UNetConv, Upsampler, Up, Down

# ------------------------------------------------------------
#                            UNet
# ------------------------------------------------------------
# This code is adapted from: https://github.com/ELEKTRONN/elektronn3/blob/master/elektronn3/models/unet.py
# The corresponding paper is:
# Olaf Ronneberger, Philipp Fischer, and Thomas Brox
# U-net: Convolutional networks for biomedical image segmentation
# International Conference on Medical Image Computing and Computer-Assisted Intervention, 2015
# Available from: https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

class UNet(nn.Module):
    def __init__(self, dims, channels, img_channels = 3, num_classes = 0, n_blocks = 3, normalization = 'BatchNorm', activation = nn.ReLU(), task = None, scale = None, bilinear=True, res = False):
        super(UNet, self).__init__()
        self.dims = dims
        self.channels = channels
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.n_blocks = n_blocks
        self.normalization = normalization
        self.activation = activation
        self.task = task
        self.scale = scale
        self.bilinear = bilinear
        self.res = res
        
        if self.dims == 1 or self.channels <= 3:
            self.filter_config=(64, 128, 256, 512, 1024)
        else:
            self.filter_config=(self.channels, self.channels*2, self.channels*4, self.channels*8, self.channels*16)
            
        factor = 2 if bilinear else 1

        self.upsample = None
        if task == 'Super-Resolution':
            upsample = [Upsampler(self.scale, self.channels, kernel_size=3, activation=False), 
                        BasicConv(self.dims, self.channels, self.channels, padding = 1, normalization = self.normalization, activation = self.activation)]
            self.upsample = nn.Sequential(*upsample)

        self.inc = UNetConv(self.dims, channels, self.filter_config[0], None, self.normalization, self.activation, self.res)
        
        down_blocks = []
        up_blocks = []
        
        for i in range(n_blocks-1):
            down_blocks.append(Down(self.dims, self.filter_config[i], self.filter_config[i+1], self.normalization, self.activation, self.res))
            up_blocks.append(Up(self.dims, self.filter_config[n_blocks-i], self.filter_config[n_blocks-i-1] // factor, self.normalization, self.activation, self.bilinear, self.res))
        down_blocks.append(Down(self.dims, self.filter_config[n_blocks-1], self.filter_config[n_blocks] // factor, self.normalization, self.activation, self.res))
        up_blocks.append(Up(self.dims, self.filter_config[1], self.filter_config[0], self.normalization, self.activation, self.bilinear, self.res))

        self.down_blocks = nn.Sequential(*down_blocks)
        self.up_blocks = nn.Sequential(*up_blocks)

        self.out = None
        self.avgpool = None
        self.tail = None
        
        if self.dims == 1:
            self.out = BasicConv(self.dims, self.filter_config[0], 1, padding = 1, normalization = self.normalization, activation = self.activation)
            if self.task == 'Classification':
                self.avgpool = nn.AdaptiveAvgPool1d(1)
                self.tail = nn.Linear(1, self.num_classes)
        elif self.dims == 2:
            if self.task == 'Classification':
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.tail = nn.Linear(self.filter_config[0], self.num_classes)
            elif self.task == 'Segmentation':
                self.tail = BasicConv(self.dims, self.filter_config[0], self.num_classes, kernel_size = 1, normalization = None, activation = None)
        elif self.dims == 3:
            self.out = BasicConv(self.dims, self.filter_config[0], 1, padding = 1, normalization = self.normalization, activation = self.activation)
            if self.task == 'Classification':
                self.avgpool = nn.AdaptiveAvgPool3d(1)
                self.tail = nn.Linear(1, self.num_classes)
            elif self.task == 'Segmentation':
                self.tail = BasicConv(2, self.img_channels, self.num_classes, kernel_size = 1, normalization = None, activation = None)

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.inc(x)
        
        encoder_output = [x]
        
        for module in self.down_blocks:
            x = module(x)
            encoder_output.append(x)
                    
        for i, module in enumerate(self.up_blocks):
            x = module(x, encoder_output[-(i+2)])
            
        if self.out is not None:
            x = self.out(x)
        if self.avgpool is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        if self.dims == 3 and self.task == 'Segmentation':
            x = torch.flatten(x,1,2)
        if self.tail is not None:
            x = self.tail(x)
        return x
