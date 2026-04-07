import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # bias=False when using BN
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    
    def __init__(self):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)


        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)


        # Output layer
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self , x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # Bottleneck
        bn = self.bottleneck(p3)

        # Decoder
        up1 = self.up1(bn)
        merge1 = torch.cat([up1, d3], dim=1)
        c1 = self.conv1(merge1)

        up2 = self.up2(c1)
        merge2 = torch.cat([up2, d2], dim=1)
        c2 = self.conv2(merge2)

        up3 = self.up3(c2)
        merge3 = torch.cat([up3, d1], dim=1)
        c3 = self.conv3(merge3)

        output = self.out(c3)

        return output