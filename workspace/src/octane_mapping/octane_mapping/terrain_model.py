import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from octane_mapping.terrain_cameras import POSE_TENSOR, compute_ray_maps


class PoseEmbedding(nn.Module):
    def __init__(self, pose_dim: int = 9, embed_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(pose_dim, embed_dim)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.proj(pose)


class UpBlock(nn.Module):
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


def _proj(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TerrainModel(nn.Module):
    """
    Inputs:
        images:   (B, 12, 3, 224, 224)
        rotation: (B, 6)  — [sin_r, cos_r, sin_p, cos_p, sin_y, cos_y]
    Outputs:
        dict with keys 'height', 'rocks', 'craters', 'walls' — each (B, 200, 200)
    """

    _S4_CH = 24
    _S3_CH = 40
    _S2_CH = 112
    _S1_CH = 1280

    PROJ_S4  = 32
    PROJ_S3  = 64
    PROJ_S2  = 128
    PROJ_S1  = 256
    POSE_DIM   = 9
    POSE_EMBED = 32
    ROT_EMBED  = 32

    def __init__(self):
        super().__init__()

        self.register_buffer('pose_tensor', torch.from_numpy(POSE_TENSOR))

        f = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features
        self.enc_s4 = f[:3]
        self.enc_s3 = f[3:4]
        self.enc_s2 = f[4:6]
        self.enc_s1 = f[6:]

        self.proj_s4 = _proj(self._S4_CH, self.PROJ_S4)
        self.proj_s3 = _proj(self._S3_CH, self.PROJ_S3)
        self.proj_s2 = _proj(self._S2_CH, self.PROJ_S2)
        self.proj_s1 = _proj(self._S1_CH, self.PROJ_S1)

        self.pose_emb   = PoseEmbedding(self.POSE_DIM, self.POSE_EMBED)
        self.pose_to_s4 = nn.Conv2d(self.POSE_EMBED, self.PROJ_S4, 1)
        self.pose_to_s3 = nn.Conv2d(self.POSE_EMBED, self.PROJ_S3, 1)
        self.pose_to_s2 = nn.Conv2d(self.POSE_EMBED, self.PROJ_S2, 1)
        self.pose_to_s1 = nn.Conv2d(self.POSE_EMBED, self.PROJ_S1, 1)

        self.register_buffer('ray_maps', torch.from_numpy(compute_ray_maps(224)))
        self.ray_to_s4 = nn.Conv2d(3, self.PROJ_S4, 1, bias=False)
        self.ray_to_s3 = nn.Conv2d(3, self.PROJ_S3, 1, bias=False)
        self.ray_to_s2 = nn.Conv2d(3, self.PROJ_S2, 1, bias=False)
        self.ray_to_s1 = nn.Conv2d(3, self.PROJ_S1, 1, bias=False)

        self.rot_mlp = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(inplace=True),
            nn.Linear(64, self.ROT_EMBED),
        )
        self.rot_to_feat = nn.Conv2d(self.ROT_EMBED, self.PROJ_S1, 1)

        self.dec1  = UpBlock(self.PROJ_S1, 192, 14)
        self.fuse2 = _proj(192 + self.PROJ_S2, 192)
        self.dec2  = UpBlock(192, 128, 28)
        self.fuse3 = _proj(128 + self.PROJ_S3, 128)
        self.dec3  = UpBlock(128, 96, 56)
        self.fuse4 = _proj(96 + self.PROJ_S4, 96)
        self.dec4  = UpBlock(96,  64, 100)
        self.dec5  = UpBlock(64,  48, 200)

        self.depth_enc = nn.Sequential(
            nn.Conv2d(3,   32, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32,  64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(128, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        )
        self.depth_up1 = UpBlock(64, 48, 56)
        self.depth_up2 = UpBlock(48, 48, 200)

        self.height_refine = nn.Sequential(
            nn.Conv2d(48 + 48, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

        self.rocks_head   = nn.Conv2d(48, 1, 1)
        self.craters_head = nn.Conv2d(48, 1, 1)
        self.walls_head   = nn.Conv2d(48, 1, 1)

    def _fuse_cameras(self, images: torch.Tensor, B: int):
        imgs_flat = images.view(B * 12, 3, 224, 224)

        x   = self.enc_s4(imgs_flat); s4r = x
        x   = self.enc_s3(x);        s3r = x
        x   = self.enc_s2(x);        s2r = x
        x   = self.enc_s1(x);        s1r = x

        s4 = self.proj_s4(s4r)
        s3 = self.proj_s3(s3r)
        s2 = self.proj_s2(s2r)
        s1 = self.proj_s1(s1r)

        pe = self.pose_emb(self.pose_tensor)
        pe = pe.unsqueeze(0).expand(B, -1, -1).reshape(B * 12, self.POSE_EMBED, 1, 1)

        s4 = s4 + self.pose_to_s4(pe)
        s3 = s3 + self.pose_to_s3(pe)
        s2 = s2 + self.pose_to_s2(pe)
        s1 = s1 + self.pose_to_s1(pe)

        ray = self.ray_maps.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * 12, 3, 224, 224)
        s4 = s4 + self.ray_to_s4(F.interpolate(ray, (56, 56), mode='bilinear', align_corners=False))
        s3 = s3 + self.ray_to_s3(F.interpolate(ray, (28, 28), mode='bilinear', align_corners=False))
        s2 = s2 + self.ray_to_s2(F.interpolate(ray, (14, 14), mode='bilinear', align_corners=False))
        s1 = s1 + self.ray_to_s1(F.interpolate(ray,  (7,  7), mode='bilinear', align_corners=False))

        s4 = s4.view(B, 12, self.PROJ_S4, 56, 56).mean(1)
        s3 = s3.view(B, 12, self.PROJ_S3, 28, 28).mean(1)
        s2 = s2.view(B, 12, self.PROJ_S2, 14, 14).mean(1)
        s1 = s1.view(B, 12, self.PROJ_S1,  7,  7).mean(1)

        return s1, s2, s3, s4

    def forward(self, images: torch.Tensor, rotation: torch.Tensor) -> dict:
        B = images.shape[0]

        s1, s2, s3, s4 = self._fuse_cameras(images, B)

        rot_emb = self.rot_mlp(rotation).unsqueeze(-1).unsqueeze(-1)
        s1 = s1 + self.rot_to_feat(rot_emb)

        x = self.dec1(s1)
        x = self.fuse2(torch.cat([x, s2], dim=1))
        x = self.dec2(x)
        x = self.fuse3(torch.cat([x, s3], dim=1))
        x = self.dec3(x)
        x = self.fuse4(torch.cat([x, s4], dim=1))
        x = self.dec4(x)
        x = self.dec5(x)

        depth_flat = images[:, 6:].reshape(B * 6, 3, 224, 224)
        depth_feat = self.depth_enc(depth_flat)
        depth_feat = depth_feat.view(B, 6, 64, 14, 14).mean(1)
        depth_feat = self.depth_up2(self.depth_up1(depth_feat))

        return {
            'height':  self.height_refine(torch.cat([x, depth_feat], dim=1)).squeeze(1),
            'rocks':   torch.sigmoid(self.rocks_head(x)).squeeze(1),
            'craters': torch.sigmoid(self.craters_head(x)).squeeze(1),
            'walls':   torch.sigmoid(self.walls_head(x)).squeeze(1),
        }
