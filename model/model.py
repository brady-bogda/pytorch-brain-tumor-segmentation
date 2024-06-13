import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class UNetModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.layer1_d = nn.Sequential(
            # 1x512x512 -> 64x256x256
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
        )
        self.layer2_d = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            # 64x256x256 -> 128x128x128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )
        self.layer3_d = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            # 128x128x128 -> 256x64x64
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256)
        )
        self.layer4_d = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            # 256x64x64 -> 512x32x32
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )
        self.layer5_d = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            # 512x32x32 -> 512x16x16
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )
        self.layer6_d = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            # 512x16x16 -> 512x8x8
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )
        self.layer7_d = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            # 512x8x8 -> 512x4x4
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )
        self.layer8_d = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            # 512x4x4 -> 512x2x2
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )

        self.layer9_d = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            # 512x2x2 -> 512x1x1
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )

        self.layer9_u = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(
                512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512))

        self.layer8_u = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(
                1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512))
        self.layer7_u = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(
                1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5))
        self.layer6_u = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(
                1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5))
        self.layer5_u = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(
                1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5))
        self.layer4_u = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(
                1024, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256))
        self.layer3_u = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(
                512, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128))
        self.layer2_u = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(
                256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64))
        self.layer1_u = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.up_layers = [self.layer1_u, self.layer2_u, self.layer3_u,
                          self.layer4_u, self.layer5_u, self.layer6_u, self.layer7_u, self.layer7_u]
        self.down_layers = [self.layer1_d, self.layer2_d, self.layer3_d,
                            self.layer4_d, self.layer5_d, self.layer6_d, self.layer7_d, self.layer8_d]

    def forward(self, x):
        # down (encode)
        e1 = self.layer1_d(input)
        e2 = self.layer2_d(e1)
        e3 = self.layer3_d(e2)
        e4 = self.layer4_d(e3)
        e5 = self.layer5_d(e4)
        e6 = self.layer6_d(e5)
        e7 = self.layer7_d(e6)
        e8 = self.layer8_d(e7)

        # up (decode)
        d8 = self.layer8_u(e8)
        d8 = torch.cat([d8, e7], 1)
        d7 = self.layer7_u(d8)
        d7 = torch.cat([d7, e6], 1)
        d6 = self.layer6_u(d7)
        d6 = torch.cat([d6, e5], 1)
        d5 = self.layer5_u(d6)
        d5 = torch.cat([d5, e4], 1)
        d4 = self.layer4_u(d5)
        d4 = torch.cat([d4, e3], 1)
        d3 = self.layer3_u(d4)
        d3 = torch.cat([d3, e2], 1)
        d2 = self.layer2_u(d3)
        d2 = torch.cat([d2, e1], 1)
        d1 = self.layer1_u(d2)

        return d1
