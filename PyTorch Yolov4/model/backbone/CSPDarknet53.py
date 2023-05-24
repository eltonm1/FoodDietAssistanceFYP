import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def activation_function(name):
    if name == 'leaky':
        return nn.LeakyReLU(alpha=0.1)
    elif name == 'linear':
        return nn.Identity()
    elif name == 'mish':
        return Mish()

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        activation='mish'):
        super().__init__()

        #Padding = kernel_size//2 as Darknet only has kernel size of 1 and 3. 
        #Only kernel size of 3 need to add padding of 1 (3//2 = 1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_function(activation)
        )

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, inout_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = inout_channels

        self.block = nn.Sequential(
            ConvBlock(inout_channels, hidden_channels, kernel_size=1),
            ConvBlock(hidden_channels, inout_channels, kernel_size=3)
        )

    def forward(self, x):
        return x + self.block(x)

class CSPFirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsampleBlock = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2)
        
        self.split_conv0 = ConvBlock(out_channels, out_channels, kernel_size=1)
        self.split_conv1 = ConvBlock(out_channels, out_channels, kernel_size=1)

        self.res_conv_block = nn.Sequential(
            ResBlock(out_channels, hidden_channels=out_channels//2),
            ConvBlock(out_channels, out_channels, kernel_size=1)
        )
        self.conv_concat = ConvBlock(out_channels*2 , out_channels, kernel_size=1)


    def forward(self, x):
        x = self.downsampleBlock(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.res_conv_block(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.conv_concat(x)
        return x

class CSPInterBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, repeat):
        super().__init__()
        self.downsampleBlock = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2)
        
        self.split_conv0 = ConvBlock(out_channels, out_channels//2, kernel_size=1)
        self.split_conv1 = ConvBlock(out_channels, out_channels//2, kernel_size=1)

        self.res_conv_block = nn.Sequential(
            *[
                ResBlock(out_channels // 2, hidden_channels=out_channels // 2)
                for _ in range(repeat)
            ],
            ConvBlock(out_channels // 2, out_channels // 2, kernel_size=1)
        )
        self.conv_concat = ConvBlock(out_channels , out_channels, kernel_size=1)


    def forward(self, x):
        x = self.downsampleBlock(x)
        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.res_conv_block(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.conv_concat(x)
        return x

class CSPDarkNet53(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels=3, out_channels=32, kernel_size=3)
        self.csp1 = CSPFirstBlock(in_channels=32, out_channels=64)
        self.csp2 = CSPInterBlock(in_channels=64, out_channels=128, repeat=2)

        self.csp3 = CSPInterBlock(in_channels=128, out_channels=256, repeat=8)
        self.csp4 = CSPInterBlock(in_channels=256, out_channels=512, repeat=8)
        self.csp5 = CSPInterBlock(in_channels=512, out_channels=1024, repeat=4)
        
        self.load_weights()
        print("Loaded CSPDarkNet53 weights")

    def forward(self, x):
        x = self.conv1(x)
        if torch.isnan(x).any():
            raise Exception("nan")
        x = self.csp1(x)
        x = self.csp2(x)

        x3 = self.csp3(x)
        x4 = self.csp4(x3)
        x5 = self.csp5(x4)

        return x3, x4, x5

    def load_weights(self):
        with open("yolov4-csp.weights", "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        ptr = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_layer = m
                if conv_layer.bias is not None:
                    bias = torch.from_numpy(weights[ptr:ptr + conv_layer.bias.numel()]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(bias)
                    ptr += conv_layer.bias.numel()
                weight = torch.from_numpy(weights[ptr:ptr + conv_layer.weight.numel()]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(weight)
                ptr += conv_layer.weight.numel()
            elif isinstance(m, nn.BatchNorm2d):
                bn_layer = m
                if bn_layer.bias is not None:
                    bias = torch.from_numpy(weights[ptr:ptr + bn_layer.bias.numel()]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bias)
                    ptr += bn_layer.bias.numel()
                weight = torch.from_numpy(weights[ptr:ptr + bn_layer.weight.numel()]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(weight)
                ptr += bn_layer.weight.numel()
                running_mean = torch.from_numpy(weights[ptr:ptr + bn_layer.running_mean.numel()]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(running_mean)
                ptr += bn_layer.running_mean.numel()
                running_var = torch.from_numpy(weights[ptr:ptr + bn_layer.running_var.numel()]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(running_var)
                ptr += bn_layer.running_var.numel()

if __name__ == "__main__":
    cspDark = CSPDarkNet53()
    x = np.random.rand(416, 416)
    x = np.expand_dims(x, 0)
    x = np.concatenate((x, x, x), axis=0)
    x = np.expand_dims(x, 0)
    x =  torch.from_numpy(x).float()
    print("Input Shape:", x.shape) 
    #Input Shape: torch.Size([1, 3, 416, 416])
    x1, x2, x3 = cspDark.forward(x)
    print("Output Shape:", x1.shape, x2.shape, x3.shape) 
    #Output Shape: torch.Size([1, 256, 52, 52]) torch.Size([1, 512, 26, 26]) torch.Size([1, 1024, 13, 13])