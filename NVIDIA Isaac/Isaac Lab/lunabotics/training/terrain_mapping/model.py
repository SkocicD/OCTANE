from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TerrainMappingModel(nn.Module):
    """ResNet-34 encoder + U-Net decoder + detection head.

    Input:  (B, 6, 200, 200) BEV grid
    Output: height    (B, 1, 200, 200)  terrain height in metres
            semantic  (B, 4, 200, 200)  per-cell logits (free/rock/crater/wall)
            detections (B, 110, 5)      [x, y, diam, conf, type] per slot
    """

    MAX_ROCKS: int = 50
    MAX_CRATERS: int = 60

    def __init__(self):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)

        orig = backbone.conv1
        backbone.conv1 = nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.conv1.weight[:, :3] = orig.weight
            backbone.conv1.weight[:, 3:] = orig.weight

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.dec3 = _DecoderBlock(512 + 256, 256)
        self.dec2 = _DecoderBlock(256 + 128, 128)
        self.dec1 = _DecoderBlock(128 + 64, 64)
        self.dec0 = _DecoderBlock(64 + 64, 64)
        self.upsample_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.height_head = nn.Conv2d(64, 1, 1)
        self.semantic_head = nn.Conv2d(64, 4, 1)

        n_slots = self.MAX_ROCKS + self.MAX_CRATERS
        self.detect_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_slots * 5),
        )

    def forward(self, x: torch.Tensor):
        e0 = self.enc0(x)
        ep = self.pool(e0)
        e1 = self.enc1(ep)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(e4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        out = self.upsample_final(d0)

        height = self.height_head(out)
        semantic = self.semantic_head(out)
        detections = self.detect_head(e4).reshape(
            -1, self.MAX_ROCKS + self.MAX_CRATERS, 5
        )
        return height, semantic, detections
