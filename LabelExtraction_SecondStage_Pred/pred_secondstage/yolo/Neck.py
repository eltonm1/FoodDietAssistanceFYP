import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        #Only kernel size of 3 need to add padding of 1 (3//2 = 1)
        self.conv = nn.Conv2d(in_channels, out_channels, padding=kernel_size//2,
            kernel_size=kernel_size, stride=stride, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activate = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.activate(x)

class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x):
        return self.upsample(x)

class SpatialPyramidPoolingPANet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        # Spatial Pyramid Pooling (SPP)
        self.spp_convBlocks1 = nn.Sequential(
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3),
            ConvBlock(1024, 512, kernel_size=1)
        )
        self.spp_maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5//2)
        self.spp_maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9//2)
        self.spp_maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13//2)
        self.spp_convBlocks2 = nn.Sequential(
            ConvBlock(2048, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3),
            ConvBlock(1024, 512, kernel_size=1),
        )

        self.upsample1 = Upsample()
        self.upsample2 = Upsample()

        # Path Aggregation Network (PANet)
        self.convBlock1 = ConvBlock(512, 256, kernel_size=1)
        self.convBlock2 = ConvBlock(512, 256, kernel_size=1)
        self.convBlocks3 = nn.Sequential(
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            ConvBlock(512, 256, kernel_size=1),
        )
        self.convBlock4 = ConvBlock(256, 128, kernel_size=1)
        self.convBlock5 = ConvBlock(256, 128, kernel_size=1)
        self.convBlocks6 = nn.Sequential(
            ConvBlock(256, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3),
            ConvBlock(256, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3),
            ConvBlock(256, 128, kernel_size=1),
        )

        self.sbbox_convBlock = ConvBlock(128, 256, kernel_size=3)
        self.sbbox_conv = nn.Conv2d(256, 3*(5+num_classes), kernel_size=1, stride=1)
        self.convBlock7 = ConvBlock(128, 256, kernel_size=3, stride=2)
        self.convBlocks8 = nn.Sequential(
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            ConvBlock(512, 256, kernel_size=1),
        )
        self.mbbox_convBlock = ConvBlock(256, 512, kernel_size=3)
        self.mbbox_conv = nn.Conv2d(512, 3*(5+num_classes), kernel_size=1, stride=1)
        self.convBlock9 = ConvBlock(256, 512, kernel_size=3, stride=2)
        self.convBlocks10 = nn.Sequential(
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3),
            ConvBlock(1024, 512, kernel_size=1),
        )
        self.lbbox_convBlock = ConvBlock(512, 1024, kernel_size=3)
        self.lbbox_conv = nn.Conv2d(1024, 3*(5+num_classes), kernel_size=1, stride=1)

    def forward(self, input_256, input_512, input_1024):
        # input_256 = torch.Size([None, 256, 52, 52])
        # input_512 = torch.Size([None, 512, 26, 26])
        # input_1024 = torch.Size([None, 1024, 13, 13])

        # Spatial Pyramid Pooling (SPP)
        x = self.spp_convBlocks1(input_1024)
        m1 = self.spp_maxpool1(x)
        m2 = self.spp_maxpool2(x)
        m3 = self.spp_maxpool3(x)
        x = torch.cat([m3, m2, m1, x], dim=1)
        x = self.spp_convBlocks2(x)
        z = x

        # Path Aggregation Network (PANet)
        x = self.convBlock1(x)
        x = self.upsample1(x)
        x2 = self.convBlock2(input_512)
        x = torch.cat([x2, x], dim=1)
        x = self.convBlocks3(x)
        y = x

        x = self.convBlock4(x)
        x = self.upsample2(x)
        x3 = self.convBlock5(input_256)
        x = torch.cat([x3, x], dim=1)
        x = self.convBlocks6(x)

        sbbox = self.sbbox_convBlock(x)
        sbbox = self.sbbox_conv(sbbox)

        x = self.convBlock7(x)
        x = torch.cat([x, y], dim=1)
        x = self.convBlocks8(x)
        
        mbbox = self.mbbox_convBlock(x)
        mbbox = self.mbbox_conv(mbbox)

        x = self.convBlock9(x)
        x = torch.cat([x, z], dim=1)
        x = self.convBlocks10(x)

        lbbox = self.lbbox_convBlock(x)
        lbbox = self.lbbox_conv(lbbox)

        return sbbox, mbbox, lbbox
        # return x3, x2, x_spp

if __name__ == "__main__":
    net = SpatialPyramidPoolingPANet(1)
    x1, x2, x3 = torch.rand((9, 256, 52, 52)), torch.rand((9, 512, 26, 26)), torch.rand((9, 1024, 13, 13))
    print("Input Shape:", x1.shape, x2.shape, x3.shape)
    # Input Shape: torch.Size([9, 256, 52, 52]) torch.Size([9, 512, 26, 26]) torch.Size([9, 1024, 13, 13])
    x1, x2, x3 = net.forward(x1, x2, x3)
    print("Output Shape:", x1.shape, x2.shape, x3.shape)
    # Output Shape: torch.Size([9, 128, 52, 52]) torch.Size([9, 256, 26, 26]) torch.Size([9, 512, 13, 13])