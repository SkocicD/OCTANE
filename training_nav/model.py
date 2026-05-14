"""Navigation policy network.

Input:
  terrain:  (B, 5, grid_size, grid_size)
              channels: height, rocks, craters, walls, goal_heatmap
  heading:  (B, 2)  — [sin(yaw), cos(yaw)]

Output:
  (B, 2)  — [left_motor, right_motor], both sigmoid-bounded to [0, 1]
             |left - right| ≤ diff_limit (0.8) enforced by expert supervision

Architecture: lightweight CNN encoder → global average pool →
concat with heading embedding → MLP → action head.
Small enough to run in real-time on Jetson Orin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NavPolicy(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        mc = cfg['model']
        ch = mc['cnn_channels']       # e.g. [32, 64, 128]
        fc = mc['fc_dims']             # e.g. [256, 128]
        he = mc['heading_embed_dim']   # e.g. 16

        # CNN: 5-channel terrain input
        layers = []
        in_ch = 5
        for out_ch in ch:
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)   # → (B, ch[-1], 1, 1)

        # Heading embedding
        self.heading_mlp = nn.Sequential(
            nn.Linear(2, he),
            nn.ReLU(inplace=True),
        )

        # MLP head
        mlp_in = ch[-1] + he
        mlp_layers = []
        for dim in fc:
            mlp_layers += [nn.Linear(mlp_in, dim), nn.ReLU(inplace=True), nn.Dropout(0.1)]
            mlp_in = dim
        self.mlp = nn.Sequential(*mlp_layers)

        self.action_head = nn.Linear(mlp_in, 2)

    def forward(self, terrain: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        x = self.cnn(terrain)
        x = self.gap(x).flatten(1)          # (B, ch[-1])
        h = self.heading_mlp(heading)        # (B, he)
        x = torch.cat([x, h], dim=1)
        x = self.mlp(x)
        return torch.sigmoid(self.action_head(x))  # (B, 2) in [0, 1] — (left, right)
