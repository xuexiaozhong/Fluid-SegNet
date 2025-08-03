import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = self.relu(out)
        out = self.excitation(out)
        out = self.sigmoid(out)
        return out

class DoubleSEConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, ratio=8):
        super(DoubleSEConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.se1 = SE(mid_channels, ratio)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.se2 = SE(out_channels, ratio)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        coefficient = self.se1(x)
        x = x * coefficient
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        coefficient = self.se2(x)
        x = x * coefficient

        return x

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class SingleSEConv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=8):
        super(SingleSEConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.se = SE(out_channels, ratio)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        coefficient = self.se(x)
        x_out = x * coefficient

        return x_out

class input_conv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super(input_conv, self).__init__()
        self.single_conv = SingleConv(in_channels, out_channels)
        self.single_se_conv = SingleSEConv(out_channels, out_channels, ratio)

    def forward(self, x):
        x = self.single_conv(x)
        x = self.single_se_conv(x)

        return x

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, ratio=8):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleSEConv(in_channels, out_channels, ratio=ratio)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, ratio=8):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleSEConv(in_channels, out_channels, in_channels // 2, ratio=ratio)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleSEConv(in_channels, out_channels, ratio=ratio)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class SE_UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64,
                 ratio: int = 8):
        super(SE_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = input_conv(in_channels, base_c, ratio)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)


    def forward(self, x: torch.Tensor, _):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.out_conv(x9)

        return x5, x6, x7, x8, logits