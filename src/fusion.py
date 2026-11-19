"""연구 당시 체크포인트의 모듈 이름을 유지하는 두 시공간 특징 결합 모델."""

import torch
from torch import nn

from model import CLIPVAD, QuickGELU


class GatedVadCLIP(CLIPVAD):
    def __init__(self, *args, backbone, fusion_dropout=0.1):
        super().__init__(*args)
        self.aux_dim = {"i3d": 2048, "swin": 1024}[backbone]
        self.projection_name = {"i3d": "i3d_projection", "swin": "videoswin_projection"}[backbone]
        projection = nn.Sequential(
            nn.Linear(self.aux_dim, self.visual_width),
            nn.LayerNorm(self.visual_width),
            QuickGELU(),
            nn.Dropout(fusion_dropout),
        )
        self.add_module(self.projection_name, projection)
        self.fusion_gate = nn.Linear(2 * self.visual_width, self.visual_width)
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        # sigmoid(-2) ≈ 0.119. 학습을 시작할 때 원래 CLIP 경로를 우선한다.
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, -2.0)

    @property
    def projection(self):
        return getattr(self, self.projection_name)

    def is_fusion_parameter(self, name):
        return name.startswith((self.projection_name + ".", "fusion_gate.", "fusion_dropout."))

    def fuse_features(self, clip_visual, aux_visual):
        if clip_visual.shape[:-1] != aux_visual.shape[:-1]:
            raise ValueError(f"Unaligned features: {clip_visual.shape} and {aux_visual.shape}")
        if clip_visual.shape[-1] != self.visual_width or aux_visual.shape[-1] != self.aux_dim:
            raise ValueError(f"Expected feature dimensions {self.visual_width} and {self.aux_dim}")
        clip_visual = clip_visual.float()
        projected = self.projection(aux_visual.float())
        gate = torch.sigmoid(self.fusion_gate(torch.cat([clip_visual, projected], dim=-1)))
        return clip_visual + self.fusion_dropout(gate * projected)

    def forward(self, clip_visual, aux_visual, padding_mask, text, lengths):
        fused = self.fuse_features(clip_visual, aux_visual)
        return super().forward(fused, padding_mask, text, lengths)


def build_model(backbone, device):
    # UCF-Crime 공식 모델과 실험 체크포인트가 공유하는 구성.
    settings = (14, 512, 256, 512, 1, 2, 8, 10, 10, device)
    if backbone == "baseline":
        return CLIPVAD(*settings).to(device)
    return GatedVadCLIP(*settings, backbone=backbone).to(device)
