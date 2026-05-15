"""Navigation policy network.

Input:
  terrain:    (B, 5, grid_size, grid_size)
                channels: height, rocks, craters, walls, goal_heatmap
  heading:    (B, 2)  — [sin(yaw), cos(yaw)]
  arena_type: (B,)    — 0.0 = UCF, 1.0 = KSC

Output: (B, 5) tensor
  [0]   left_motor   — sigmoid [0, 1]
  [1]   right_motor  — sigmoid [0, 1]
  [2:5] bucket_logits — raw logits for CrossEntropy (classes: 0=UP 1=COLLECT 2=DUMP)

Architecture: lightweight CNN encoder → global average pool →
concat with heading + arena_type embeddings → MLP → two separate heads.
Small enough to run in real-time on Jetson Orin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NavPolicy(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        mc = cfg['model']
        ch = mc['cnn_channels']
        fc = mc['fc_dims']
        he = mc['heading_embed_dim']
        ae = mc.get('arena_embed_dim', 4)

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
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.heading_mlp = nn.Sequential(
            nn.Linear(2, he),
            nn.ReLU(inplace=True),
        )

        # arena_type is a scalar [0=UCF, 1=KSC]; small embedding lets the network
        # learn layout-specific navigation strategies
        self.arena_mlp = nn.Sequential(
            nn.Linear(1, ae),
            nn.ReLU(inplace=True),
        )

        mlp_in = ch[-1] + he + ae
        mlp_layers = []
        for dim in fc:
            mlp_layers += [nn.Linear(mlp_in, dim), nn.ReLU(inplace=True), nn.Dropout(0.1)]
            mlp_in = dim
        self.mlp = nn.Sequential(*mlp_layers)

        self.motor_head  = nn.Linear(mlp_in, 2)
        self.bucket_head = nn.Linear(mlp_in, 3)

    def forward(self, terrain: torch.Tensor, heading: torch.Tensor,
                arena_type: torch.Tensor) -> torch.Tensor:
        x = self.cnn(terrain)
        x = self.gap(x).flatten(1)
        h = self.heading_mlp(heading)
        a = self.arena_mlp(arena_type.unsqueeze(1).float())   # (B,) → (B,1) → (B, ae)
        x = torch.cat([x, h, a], dim=1)
        x = self.mlp(x)
        motors        = torch.tanh(self.motor_head(x))      # (B, 2) in [-1, 1] (normalised by nav_speed_limit)
        bucket_logits = self.bucket_head(x)                 # (B, 3) raw logits
        return torch.cat([motors, bucket_logits], dim=1)    # (B, 5)

    @staticmethod
    def decode(out: torch.Tensor):
        """Split a (B,5) or (5,) output into (left, right, bucket_class)."""
        motors = out[..., :2]
        bucket = out[..., 2:].argmax(dim=-1)
        return motors, bucket
