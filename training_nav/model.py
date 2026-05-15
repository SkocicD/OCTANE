"""Navigation policy network — large LSTM-based temporal architecture.

~9 M parameters.  Designed to run on Jetson Orin at 10 Hz.

Single-step inference:
  terrain:    (B, 5, grid_size, grid_size)
  heading:    (B, 2)   — [sin(yaw), cos(yaw)]
  zone_idx:   (B,)     — long  (current zone: start/excavation/nav/deposit/berm/outside)
  arena_type: (B,)     — float  0.0=UCF  1.0=KSC
  phase_idx:  (B,)     — long  0=to_excavation 1=digging 2=to_deposit 3=dumping
  hidden:     (h_n, c_n) or None

Sequence training:
  terrain:    (B, T, 5, grid_size, grid_size)
  heading:    (B, T, 2)
  zone_idx:   (B, T)   — long, per-timestep (zone changes as robot moves)
  arena_type: (B,)
  phase_idx:  (B,)     — constant per sequence
  hidden:     None (reset each sequence)

Output: (output_tensor, new_hidden)
  output: (B, 5) at inference,  (B, T, 5) during training
    [0]   left_motor   — tanh [-1, 1]
    [1]   right_motor  — tanh [-1, 1]
    [2:5] bucket_logits — raw logits for CrossEntropy (0=UP 1=COLLECT 2=DUMP)
  new_hidden: (h_n, c_n)

Architecture (~9 M parameters):
  CNN     : 4× double-conv [64,128,256,512] BN+ReLU+MaxPool, GlobalAvgPool → 512
  heading : Linear(2→16) + ReLU
  zone    : Embedding(6→8)  per-timestep  ← start/exc/nav/deposit/berm/outside
  arena   : Linear(1→8)  + ReLU           constant per sequence
  phase   : Embedding(4→8)                constant per sequence
  fc_in   : Linear(544→512) + LayerNorm + ReLU
  LSTM    : 2-layer LSTM(512, 512), inter-layer dropout=0.1
  MLP     : [512, 256, 128] + ReLU + Dropout(0.1) each
  heads   : motor(128→2)+tanh,  bucket(128→3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NavPolicy(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        mc          = cfg['model']
        ch          = mc['cnn_channels']           # [64, 128, 256, 512]
        fc          = mc['fc_dims']                # [512, 256, 128]
        he          = mc['heading_embed_dim']      # 16
        ae          = mc.get('arena_embed_dim', 8) # 8
        ze          = mc.get('zone_embed_dim',  8) # 8  per-timestep zone
        ph          = mc.get('phase_embed_dim', 8) # 8  per-sequence mission phase
        lstm_hidden = mc.get('lstm_hidden', 512)
        lstm_layers = mc.get('lstm_layers', 2)

        # ── CNN encoder ───────────────────────────────────────────────────────
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
        self.gap = nn.AdaptiveAvgPool2d(1)         # → (B, ch[-1])

        # ── Auxiliary encoders ────────────────────────────────────────────────
        self.heading_mlp = nn.Sequential(nn.Linear(2, he), nn.ReLU(inplace=True))
        self.zone_emb    = nn.Embedding(6, ze)     # 6 zones, per-timestep
        self.arena_mlp   = nn.Sequential(nn.Linear(1, ae), nn.ReLU(inplace=True))
        self.phase_emb   = nn.Embedding(4, ph)     # 4 mission phases, constant/seq

        # ── Project to LSTM input ─────────────────────────────────────────────
        cnn_out   = ch[-1]                         # 512
        feat_size = cnn_out + he + ze + ae + ph    # 512+16+8+8+8 = 552
        self.fc_in = nn.Sequential(
            nn.Linear(feat_size, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(inplace=True),
        )

        # ── 2-layer LSTM ──────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            lstm_hidden, lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )

        # ── MLP head ──────────────────────────────────────────────────────────
        mlp_in     = lstm_hidden
        mlp_layers = []
        for dim in fc:
            mlp_layers += [nn.Linear(mlp_in, dim), nn.ReLU(inplace=True), nn.Dropout(0.1)]
            mlp_in = dim
        self.mlp = nn.Sequential(*mlp_layers)

        self.motor_head  = nn.Linear(mlp_in, 2)
        self.bucket_head = nn.Linear(mlp_in, 3)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, terrain: torch.Tensor, heading: torch.Tensor,
                zone_idx: torch.Tensor, arena_type: torch.Tensor,
                phase_idx: torch.Tensor, hidden=None):
        """
        Args:
            terrain:    (B, 5, gs, gs)  or (B, T, 5, gs, gs)
            heading:    (B, 2)           or (B, T, 2)
            zone_idx:   (B,) long        or (B, T) long    — per-timestep zone
            arena_type: (B,)
            phase_idx:  (B,) long        — constant per sequence
            hidden:     (h_n, c_n) or None

        Returns:
            out:        (B, 5) or (B, T, 5)
            new_hidden: (h_n, c_n)
        """
        single_step = (terrain.dim() == 4)
        if single_step:
            terrain  = terrain.unsqueeze(1)    # (B,1,5,gs,gs)
            heading  = heading.unsqueeze(1)    # (B,1,2)
            zone_idx = zone_idx.unsqueeze(1)   # (B,1)

        B, T, C, H, W = terrain.shape

        # ── CNN ───────────────────────────────────────────────────────────────
        x = self.cnn(terrain.reshape(B * T, C, H, W))
        x = self.gap(x).flatten(1).reshape(B, T, -1)      # (B, T, 512)

        # ── Heading embedding (per-timestep) ──────────────────────────────────
        h_emb = self.heading_mlp(heading)                  # (B, T, 16)

        # ── Zone embedding (per-timestep) ─────────────────────────────────────
        z_emb = self.zone_emb(zone_idx)                    # (B, T, 8)

        # ── Arena embedding (constant per sequence) ───────────────────────────
        a_emb = self.arena_mlp(arena_type.unsqueeze(1).float())   # (B, 8)
        a_emb = a_emb.unsqueeze(1).expand(B, T, -1)               # (B, T, 8)

        # ── Phase embedding (constant per sequence) ───────────────────────────
        ph_emb = self.phase_emb(phase_idx)                 # (B, 8)
        ph_emb = ph_emb.unsqueeze(1).expand(B, T, -1)     # (B, T, 8)

        # ── Concat → project → LSTM ───────────────────────────────────────────
        feat = torch.cat([x, h_emb, z_emb, a_emb, ph_emb], dim=2)  # (B, T, 552)
        feat = self.fc_in(feat)                            # (B, T, lstm_hidden)

        lstm_out, new_hidden = self.lstm(feat, hidden)     # (B, T, lstm_hidden)

        # ── MLP + heads ───────────────────────────────────────────────────────
        mlp_out       = self.mlp(lstm_out)
        motors        = torch.tanh(self.motor_head(mlp_out))   # (B, T, 2)
        bucket_logits = self.bucket_head(mlp_out)              # (B, T, 3)
        out = torch.cat([motors, bucket_logits], dim=2)        # (B, T, 5)

        if single_step:
            out = out.squeeze(1)   # (B, 5)

        return out, new_hidden

    @staticmethod
    def decode(out: torch.Tensor):
        """Split (..., 5) output into (motors, bucket_class)."""
        return out[..., :2], out[..., 2:].argmax(dim=-1)
