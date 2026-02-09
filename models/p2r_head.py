"""
P2R (Point-to-Region) density estimation head.

Supports variable input channels (512 standard or 514 for ZIP-as-feature).
Uses adaptive GroupNorm, skip connections, learnable log_scale parameter,
and optional dropout regularization. Includes a multi-scale variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_3x3(in_channels, out_channels, stride=1, padding=1, bn=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3,
                  stride=stride, padding=padding, bias=not bn)
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv_1x1(in_channels, out_channels, bn=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not bn)
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _get_group_norm_groups(channels: int) -> int:
    preferred = [32, 16, 8, 4, 2, 1]

    for g in preferred:
        if channels % g == 0:
            return g

    for g in range(min(32, channels), 0, -1):
        if channels % g == 0:
            return g

    return 1


class P2RHead(nn.Module):

    def __init__(
        self,
        in_channel: int = 512,
        fea_channel: int = 256,
        out_stride: int = 16,
        log_scale_init: float = 4.0,
        log_scale_clamp: tuple = (-2.0, 10.0),
        use_input_norm: bool = True,
        dropout_rate: float = 0.0,
        final_dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.in_channel = in_channel
        self.out_stride = out_stride
        self.log_scale_clamp = log_scale_clamp
        self.use_input_norm = use_input_norm
        self.dropout_rate = dropout_rate
        self.final_dropout_rate = final_dropout_rate

        self.log_scale = nn.Parameter(
            torch.tensor(log_scale_init, dtype=torch.float32),
            requires_grad=True
        )

        if use_input_norm:
            num_groups = _get_group_norm_groups(in_channel)
            self.input_norm = nn.GroupNorm(num_groups, in_channel)
            self._norm_groups = num_groups
        else:
            self.input_norm = nn.Identity()
            self._norm_groups = 0

        self.layer1 = self._make_layer(in_channel, fea_channel, dropout_rate)
        self.layer2 = self._make_layer(fea_channel, fea_channel, dropout_rate)
        self.layer3 = self._make_layer(fea_channel, fea_channel, dropout_rate)
        self.layer4 = self._make_layer(fea_channel, fea_channel, final_dropout_rate)

        self.pred_layer = nn.Sequential(
            nn.Conv2d(fea_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.skip_conv = conv_1x1(in_channel, fea_channel, bn=True)

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, dropout):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.in_channel:
            raise ValueError(
                f"P2RHead expects {self.in_channel} input channels, "
                f"got {features.shape[1]}"
            )

        x = self.input_norm(features)

        skip = self.skip_conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x + skip

        raw_density = self.pred_layer(x)

        clamped_log_scale = torch.clamp(
            self.log_scale,
            min=self.log_scale_clamp[0],
            max=self.log_scale_clamp[1]
        )
        scale = torch.exp(clamped_log_scale)

        density = F.softplus(raw_density) * scale

        return density

    def get_scale(self) -> float:
        with torch.no_grad():
            clamped = torch.clamp(
                self.log_scale,
                min=self.log_scale_clamp[0],
                max=self.log_scale_clamp[1]
            )
            return torch.exp(clamped).item()

    def get_log_scale(self) -> float:
        with torch.no_grad():
            return self.log_scale.item()

    def set_log_scale(self, value: float):
        with torch.no_grad():
            self.log_scale.fill_(value)


class P2RHeadMultiScale(P2RHead):

    def __init__(
        self,
        in_channel: int = 512,
        fea_channel: int = 256,
        out_stride: int = 16,
        log_scale_init: float = 4.0,
        log_scale_clamp: tuple = (-2.0, 10.0),
        scales: list = [1, 2, 4]
    ):
        super().__init__(
            in_channel=in_channel,
            fea_channel=fea_channel,
            out_stride=out_stride,
            log_scale_init=log_scale_init,
            log_scale_clamp=log_scale_clamp
        )
        self.scales = scales

    def forward_multiscale(self, features: torch.Tensor) -> dict:
        density = self.forward(features)

        multiscale = {}
        for s in self.scales:
            if s == 1:
                multiscale[s] = density
            else:
                pooled = F.avg_pool2d(density, kernel_size=s, stride=s)
                multiscale[s] = pooled * (s ** 2)

        return {
            'density': density,
            'multiscale_densities': multiscale
        }
