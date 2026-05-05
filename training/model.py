import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.cameras import POSE_TENSOR


class PoseEmbedding(nn.Module):
    """Projects fixed 9-dim pose vectors to embed_dim feature space."""
    def __init__(self, pose_dim: int = 9, embed_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(pose_dim, embed_dim)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.proj(pose)  # (12, embed_dim)


class UpBlock(nn.Module):
    """Bilinear upsample to target_size then double conv."""
    def __init__(self, in_ch: int, out_ch: int, target_size: int):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return self.conv(x)


class TerrainModel(nn.Module):
    """
    Inputs:
        images:   (B, 12, 3, 224, 224)
        rotation: (B, 6)  — [sin_r, cos_r, sin_p, cos_p, sin_y, cos_y]
    Outputs:
        dict with keys 'height', 'rocks', 'craters', 'walls' — each (B, 200, 200)
    """

    FEAT_DIM   = 1280   # EfficientNet-B0 output channels for 224x224 input
    PROJ_DIM   = 128    # per-camera projection dim
    POSE_DIM   = 9
    POSE_EMBED = 32
    FUSION_DIM = 512
    ROT_EMBED  = 32

    def __init__(self):
        super().__init__()

        # Register pose tensor as non-trainable buffer (moves with .to(device))
        self.register_buffer('pose_tensor', torch.from_numpy(POSE_TENSOR))  # (12, 9)

        # Shared image encoder — EfficientNet-B0 features
        # For 224x224 input: outputs (B, 1280, 7, 7)
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.encoder = backbone.features  # (B, 1280, 7, 7)

        # Per-camera feature projection: 1280 → PROJ_DIM
        self.feat_proj = nn.Sequential(
            nn.Conv2d(self.FEAT_DIM, self.PROJ_DIM, 1, bias=False),
            nn.BatchNorm2d(self.PROJ_DIM),
            nn.ReLU(inplace=True),
        )

        # Camera pose embedding: 9-dim fixed → POSE_EMBED-dim learned
        self.pose_emb = PoseEmbedding(self.POSE_DIM, self.POSE_EMBED)
        # Project pose embedding to the same feature space as feat_proj output
        self.pose_to_feat = nn.Conv2d(self.POSE_EMBED, self.PROJ_DIM, 1)

        # Fusion: 12 cameras × PROJ_DIM channels → FUSION_DIM
        self.fusion = nn.Sequential(
            nn.Conv2d(12 * self.PROJ_DIM, self.FUSION_DIM, 1, bias=False),
            nn.BatchNorm2d(self.FUSION_DIM),
            nn.ReLU(inplace=True),
        )

        # Robot rotation embedding: 6 → ROT_EMBED
        self.rot_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.ROT_EMBED),
        )
        self.rot_to_feat = nn.Conv2d(self.ROT_EMBED, self.FUSION_DIM, 1)

        # BEV decoder: 7×7 → 14 → 25 → 50 → 100 → 200
        self.decoder = nn.Sequential(
            UpBlock(self.FUSION_DIM, 256, 14),
            UpBlock(256, 128, 25),
            UpBlock(128, 64,  50),
            UpBlock(64,  32,  100),
            UpBlock(32,  32,  200),
        )

        # Output heads
        self.height_head  = nn.Conv2d(32, 1, 1)
        self.rocks_head   = nn.Conv2d(32, 1, 1)
        self.craters_head = nn.Conv2d(32, 1, 1)
        self.walls_head   = nn.Conv2d(32, 1, 1)

    def forward(self, images: torch.Tensor, rotation: torch.Tensor) -> dict:
        B = images.shape[0]

        # Encode all 12 images in a single batch pass through shared backbone
        imgs_flat = images.view(B * 12, 3, 224, 224)
        feats = self.encoder(imgs_flat)       # (B*12, 1280, 7, 7)
        feats = self.feat_proj(feats)         # (B*12, 128, 7, 7)

        # Camera pose embeddings: fixed 9-dim → 32-dim → add to features
        pose_emb = self.pose_emb(self.pose_tensor)        # (12, 32)
        pose_emb = pose_emb.unsqueeze(-1).unsqueeze(-1)   # (12, 32, 1, 1)
        # Expand to (B*12, 32, 1, 1)
        pose_emb = pose_emb.unsqueeze(0).expand(B, -1, -1, 1, 1).reshape(B * 12, self.POSE_EMBED, 1, 1)
        feats = feats + self.pose_to_feat(pose_emb)       # broadcast add

        # Fuse all 12 camera features
        feats = feats.view(B, 12 * self.PROJ_DIM, 7, 7)  # (B, 1536, 7, 7)
        bev   = self.fusion(feats)                        # (B, 512, 7, 7)

        # Robot rotation embedding
        rot_emb = self.rot_mlp(rotation)                  # (B, 32)
        rot_emb = rot_emb.unsqueeze(-1).unsqueeze(-1)     # (B, 32, 1, 1)
        bev = bev + self.rot_to_feat(rot_emb)             # broadcast add

        # Decode to BEV
        bev = self.decoder(bev)                           # (B, 32, 200, 200)

        return {
            'height':  self.height_head(bev).squeeze(1),
            'rocks':   torch.sigmoid(self.rocks_head(bev)).squeeze(1),
            'craters': torch.sigmoid(self.craters_head(bev)).squeeze(1),
            'walls':   torch.sigmoid(self.walls_head(bev)).squeeze(1),
        }
