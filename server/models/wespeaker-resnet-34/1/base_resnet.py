from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch import nn


class StatsPool(nn.Module):
    def _pool(self, sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.unsqueeze(dim=1)

        v1 = weights.sum(dim=2) + 1e-8
        mean = torch.sum(sequences * weights, dim=2) / v1

        dx2 = torch.square(sequences - mean.unsqueeze(2))
        v2 = torch.square(weights).sum(dim=2)

        var = torch.sum(dx2 * weights, dim=2) / (v1 - v2 / v1 + 1e-8)
        std = torch.sqrt(var)

        return torch.cat([mean, std], dim=1)

    def forward(
        self, sequences: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if weights is None:
            mean = sequences.mean(dim=-1)
            std = sequences.std(dim=-1, correction=1)
            return torch.cat([mean, std], dim=-1)

        if weights.dim() == 2:
            has_speaker_dimension = False
            weights = weights.unsqueeze(dim=1)

        else:
            has_speaker_dimension = True

        _, _, num_frames = sequences.shape
        _, _, num_weights = weights.shape
        if num_frames != num_weights:
            logger.warning(
                f"Mismatch between frames ({num_frames}) and weights ({num_weights}) numbers."
            )
            weights = F.interpolate(weights, size=num_frames, mode="nearest")

        output = rearrange(
            torch.vmap(self._pool, in_dims=(None, 1))(sequences, weights),
            "speakers batch features -> batch speakers features",
        )

        if not has_speaker_dimension:
            return output.squeeze(dim=1)

        return output


class TSTP(nn.Module):
    def __init__(self, in_dim=0):
        super(TSTP, self).__init__()
        self.in_dim = in_dim
        self.stats_pool = StatsPool()

    def forward(self, features, weights: torch.Tensor = None):
        features = rearrange(
            features,
            "batch dimension channel frames -> batch (dimension channel) frames",
        )

        return self.stats_pool(features, weights=weights)

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


POOLING_LAYERS = {"TSTP": TSTP}


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        m_channels=32,
        feat_dim=40,
        embed_dim=128,
        pooling_func="TSTP",
        two_emb_layer=True,
    ):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)

        self.pool = POOLING_LAYERS[pooling_func](
            in_dim=self.stats_dim * block.expansion
        )
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, weights: torch.Tensor = None):
        x = x.permute(0, 2, 1)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out, weights=weights)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a
