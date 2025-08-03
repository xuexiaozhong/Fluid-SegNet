import torch
import torch.nn as nn

class DoubleConv_2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_2d, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DoubleConv_3d(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_3d, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down_2d(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down_2d, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_2d(in_channels, out_channels)
        )

class Down_3d(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down_3d, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_3d(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_2d(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()
        self.conv2d = nn.Conv2d(channels, channels, kernel_size=1)
        self.ReLu2d = nn.ReLU()
        self.conv3d = nn.Conv2d(channels, channels, kernel_size=1)
        self.ReLu3d = nn.ReLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x_2d = self.conv2d(x1)
        x_3d = self.conv3d(x2)

        return self.ReLu2d(x_2d) + self.ReLu3d(x_3d)

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class YNet(nn.Module):
    def __init__(self,
                 in_channels_2d: int = 1,
                 in_channels_3d: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32,
                 using_max_map: bool = False):
        super(YNet, self).__init__()
        self.in_channels_2d = in_channels_2d
        self.in_channels_3d = in_channels_3d
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.using_max_map = using_max_map

        factor = 2 if bilinear else 1

        self.in_conv_2d = DoubleConv_2d(in_channels_2d, base_c)
        self.down1_2d = Down_2d(base_c, base_c*2)
        self.down2_2d = Down_2d(base_c*2, base_c*4)
        self.down3_2d = Down_2d(base_c*4, base_c*8)
        self.down4_2d = Down_2d(base_c*8, base_c*16 // factor)

        self.in_conv_3d = DoubleConv_3d(in_channels_3d, base_c)
        self.down1_3d = Down_3d(base_c, base_c*2)
        self.down2_3d = Down_3d(base_c * 2, base_c * 4)
        self.down3_3d = Down_3d(base_c * 4, base_c * 8)
        self.down4_3d = Down_3d(base_c * 8, base_c * 16 // factor)

        self.fusion0 = Fusion(base_c)
        self.fusion1 = Fusion(base_c*2)
        self.fusion2 = Fusion(base_c*4)
        self.fusion3 = Fusion(base_c*8)
        self.fusion4 = Fusion(base_c*16 // factor)

        self.up1 = Up(base_c*16, base_c*8 // factor, bilinear)
        self.up2 = Up(base_c*8, base_c*4 // factor, bilinear)
        self.up3 = Up(base_c*4, base_c*2 // factor, bilinear)
        self.up4 = Up(base_c*2, base_c)

        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x_2d: torch.Tensor, x_3d: torch.Tensor):
        x1_2d = self.in_conv_2d(x_2d)
        x1_3d = self.in_conv_3d(x_3d)
        x1 = self.fusion0(x1_2d, x1_3d)

        x2_2d = self.down1_2d(x1_2d)
        x2_3d = self.down1_3d(x1_3d)
        x2 = self.fusion1(x2_2d, x2_3d)

        x3_2d = self.down2_2d(x2_2d)
        x3_3d = self.down2_3d(x2_3d)
        x3 = self.fusion2(x3_2d, x3_3d)

        x4_2d = self.down3_2d(x3_2d)
        x4_3d = self.down3_3d(x3_3d)
        x4 = self.fusion3(x4_2d, x4_3d)

        x5_2d = self.down4_2d(x4_2d)
        x5_3d = self.down4_3d(x4_3d)
        x5 = self.fusion4(x5_2d, x5_3d)

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.out_conv(x9)

        if self.using_max_map:
            x5, x5_index = torch.max(x5, 1)
            x6, x6_index = torch.max(x6, 1)
            x7, x7_index = torch.max(x7, 1)
            x8, x8_index = torch.max(x8, 1)

        return x5, x6, x7, x8, logits