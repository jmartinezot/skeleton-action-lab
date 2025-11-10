# stgcn_backbone.py
"""
A small ST-GCN-style backbone for NTU60 skeletons.

Input shape: (B, C=3, T, V=25) or (B, C=3, T, V=25, M=2)
We keep only the first person when M=2.

This is intentionally simple and not a full reproduction of ST-GCN,
but it has:
- fixed skeleton adjacency A over joints
- spatial graph convolution over joints
- temporal conv over time
- residual connections
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_ntu_adjacency(
    num_joints: int = 25,
    self_loops: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Build a simple undirected adjacency matrix for NTU's 25 joints.

    Joints are 1-based in the NTU docs; here we use 0-based indices.
    Edges follow natural body connections (approximate, good enough
    for experimentation; you can swap in the official ST-GCN A later).
    """
    A = torch.zeros((num_joints, num_joints), dtype=torch.float32)

    # Edges in 1-based indexing (from NTU skeleton layout, approx)
    edges_1based = [
        # spine
        (1, 2), (2, 3), (3, 4),       # base -> mid -> neck -> head
        (1, 21),                      # base <-> spine (extra joint)
        # left arm
        (3, 5), (5, 6), (6, 7), (7, 8),
        (8, 22), (7, 23),             # hand tips & thumb
        # right arm
        (3, 9), (9, 10), (10, 11), (11, 12),
        (12, 24), (11, 25),           # hand tips & thumb
        # left leg
        (1, 13), (13, 14), (14, 15), (15, 16),
        # right leg
        (1, 17), (17, 18), (18, 19), (19, 20),
    ]

    for i1, j1 in edges_1based:
        i = i1 - 1
        j = j1 - 1
        A[i, j] = 1.0
        A[j, i] = 1.0

    if self_loops:
        idx = torch.arange(num_joints)
        A[idx, idx] = 1.0

    if normalize:
        # Row-normalize: each node's outgoing weights sum to 1
        deg = A.sum(dim=1, keepdim=True)
        deg[deg == 0] = 1.0
        A = A / deg

    return A


class SpatialGCN(nn.Module):
    """
    Simple spatial graph conv:
        x_agg[v] = sum_w A[v, w] * x[w]
        y = Conv1x1(x_agg)

    Input:  x (B, C_in, T, V)
    Output: y (B, C_out, T, V)
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        # Register adjacency as a buffer so it moves with the model's device
        self.register_buffer("A", A)  # (V, V)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, V)
        A = self.A  # (V, V)

        # Aggregate over neighbors in joint space:
        # x_agg[b, c, t, v] = sum_w A[v, w] * x[b, c, t, w]
        x_agg = torch.einsum("vw,bctw->bctv", A, x)

        y = self.conv(x_agg)
        y = self.bn(y)
        y = self.relu(y)
        return y


class STGCNBlock(nn.Module):
    """
    Spatial GCN + Temporal Conv + Residual.

    Input / output: (B, C_out, T, V)
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, stride: int = 1):
        super().__init__()

        self.gcn = SpatialGCN(in_channels, out_channels, A)

        # Temporal conv over T dimension only (kernel 9, padding 4)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Residual path
        if (in_channels != out_channels) or (stride != 1):
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x


class STGCNBackbone(nn.Module):
    """
    Small ST-GCN-style network for NTU60:

    - Input:  (B, 3, T, 25) or (B, 3, T, 25, 2)
    - 3 ST-GCN blocks with channels [64, 64, 128]
    - Global average pooling over T and V
    - Linear classifier to 60 classes
    """

    def __init__(
        self,
        num_classes: int = 60,
        in_channels: int = 3,
        num_joints: int = 25,
    ):
        super().__init__()
        A = build_ntu_adjacency(num_joints=num_joints, self_loops=True, normalize=True)

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.layer1 = STGCNBlock(in_channels, 64, A, stride=1)
        self.layer2 = STGCNBlock(64, 64, A, stride=1)
        self.layer3 = STGCNBlock(64, 128, A, stride=2)  # downsample in time

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # over T and V
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, V) or (B, C, T, V, M)
        """
        if x.ndim == 5:
            x = x[..., 0]  # keep first person: (B, C, T, V)

        b, c, t, v = x.shape

        # Optional input BN over (C, V) per time-step
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()  # (B, T, C, V)
        x_reshaped = x_reshaped.view(b * t, c * v)
        x_reshaped = self.data_bn(x_reshaped)
        x = x_reshaped.view(b, t, c, v).permute(0, 2, 1, 3).contiguous()  # back to (B, C, T, V)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global average pool over time and joints: (B, C, 1, 1)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # (B, C)
        x = self.fc(x)
        return x

