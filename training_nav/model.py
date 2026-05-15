"""Navigation policy network — LSTM-based temporal architecture.

Single-step inference (Jetson Orin, ~10 Hz):
  terrain:    (B, 5, grid_size, grid_size)
  heading:    (B, 2)  — [sin(yaw), cos(yaw)]
  arena_type: (B,)    — 0.0 = UCF, 1.0 = KSC
  hidden:     (h_n, c_n) or None

Sequence training:
  terrain:    (B, T, 5, grid_size, grid_size)
  heading:    (B, T, 2)
  arena_type: (B,)
  hidden:     None (reset each sequence)

Output: (output_tensor, new_hidden)
  output: (B, 5) at inference, (B, T, 5) during training
    [0]   left_motor   — tanh [-1, 1]  (normalised by nav_speed_limit)
    [1]   right_motor  — tanh [-1, 1]
    [2:5] bucket_logits — raw logits for CrossEntropy (classes: 0=UP 1=COLLECT 2=DUMP)
  new_hidden: (h_n, c_n) tuple

Architecture:
  CNN encoder (double-conv blocks, BN+ReLU+MaxPool, GlobalAvgPool → 128-d)
  → heading_mlp (2→16) + arena_mlp (1→4)
  → fc_in projects concat to lstm_hidden
  → LSTM(lstm_hidden, lstm_hidden)
  → MLP (fc_dims with Dropout 0.1)
  → motor_head (tanh) + bucket_head (logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NavPolicy(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        mc          = cfg['model']
        ch          = mc['cnn_channels']
        fc          = mc['fc_dims']
        he          = mc['heading_embed_dim']
        ae          = mc.get('arena_embed_dim', 4)
        lstm_hidden = mc.get('lstm_hidden', 256)

        # ── CNN encoder: 5-channel terrain input ─────────────────────────────
        layers = []
        in_ch  = 5
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
        self.gap = nn.AdaptiveAvgPool2d(1)   # → (B, 128, 1, 1)

        # ── Auxiliary encoders ────────────────────────────────────────────────
        self.heading_mlp = nn.Sequential(
            nn.Linear(2, he),
            nn.ReLU(inplace=True),
        )
        self.arena_mlp = nn.Sequential(
            nn.Linear(1, ae),
            nn.ReLU(inplace=True),
        )

        # ── Project concat features to LSTM input size ────────────────────────
        cnn_out  = ch[-1]
        self.fc_in = nn.Sequential(
            nn.Linear(cnn_out + he + ae, lstm_hidden),
            nn.ReLU(inplace=True),
        )

        # ── Temporal LSTM ─────────────────────────────────────────────────────
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, batch_first=True)

        # ── MLP head ─────────────────────────────────────────────────────────
        mlp_in = lstm_hidden
        mlp_layers = []
        for dim in fc:
            mlp_layers += [nn.Linear(mlp_in, dim), nn.ReLU(inplace=True), nn.Dropout(0.1)]
            mlp_in = dim
        self.mlp = nn.Sequential(*mlp_layers)

        self.motor_head  = nn.Linear(mlp_in, 2)
        self.bucket_head = nn.Linear(mlp_in, 3)

    def forward(self, terrain: torch.Tensor, heading: torch.Tensor,
                arena_type: torch.Tensor, hidden=None):
        """
        Args:
            terrain:    (B, 5, gs, gs) or (B, T, 5, gs, gs)
            heading:    (B, 2)         or (B, T, 2)
            arena_type: (B,)
            hidden:     (h_n, c_n) or None

        Returns:
            out:        (B, 5) or (B, T, 5)
            new_hidden: (h_n, c_n)
        """
        single_step = (terrain.dim() == 4)
        if single_step:
            # Add T=1 dimension
            terrain = terrain.unsqueeze(1)   # (B,1,5,gs,gs)
            heading = heading.unsqueeze(1)   # (B,1,2)

        B, T, C, H, W = terrain.shape

        # ── CNN: process all timesteps together ───────────────────────────────
        terrain_flat = terrain.reshape(B * T, C, H, W)
        x = self.cnn(terrain_flat)
        x = self.gap(x).flatten(1)                    # (B*T, 128)
        x = x.reshape(B, T, -1)                       # (B, T, 128)

        # ── Heading embedding ─────────────────────────────────────────────────
        h_emb = self.heading_mlp(heading)              # (B, T, he)

        # ── Arena embedding — expand scalar to (B, T, ae) ─────────────────────
        a_emb = self.arena_mlp(arena_type.unsqueeze(1).float())   # (B, ae)
        a_emb = a_emb.unsqueeze(1).expand(B, T, -1)               # (B, T, ae)

        # ── Concat + project to LSTM input ───────────────────────────────────
        feat = torch.cat([x, h_emb, a_emb], dim=2)   # (B, T, 128+he+ae)
        feat = self.fc_in(feat)                       # (B, T, lstm_hidden)

        # ── LSTM ─────────────────────────────────────────────────────────────
        lstm_out, new_hidden = self.lstm(feat, hidden)  # (B, T, lstm_hidden)

        # ── MLP + heads ───────────────────────────────────────────────────────
        mlp_out      = self.mlp(lstm_out)                         # (B, T, last_fc)
        motors       = torch.tanh(self.motor_head(mlp_out))       # (B, T, 2)
        bucket_logits = self.bucket_head(mlp_out)                 # (B, T, 3)
        out = torch.cat([motors, bucket_logits], dim=2)           # (B, T, 5)

        if single_step:
            out = out.squeeze(1)   # (B, 5)

        return out, new_hidden

    @staticmethod
    def decode(out: torch.Tensor):
        """Split a (..., 5) output into (motors, bucket_class)."""
        motors = out[..., :2]
        bucket = out[..., 2:].argmax(dim=-1)
        return motors, bucket
