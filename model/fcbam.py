import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqChannelAttention(nn.Module):
    def __init__(self, reduction_ratio=16, pool_size=2):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.pool_size = pool_size
        self.ca_low = None
        self.ca_high = None
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _build(self, in_channels, device):
        mid = max(in_channels // self.reduction_ratio, 4)

        def make_ca():
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, mid, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, in_channels, 1, bias=False),
                nn.Sigmoid()
            ).to(device)

        self.ca_low = make_ca()
        self.ca_high = make_ca()

    def forward(self, x):
        if self.ca_low is None:
            self._build(x.size(1), x.device)

        low_freq = F.avg_pool2d(x, kernel_size=self.pool_size,
                                stride=1, padding=self.pool_size // 2)
        low_freq = low_freq[:, :, :x.shape[2], :x.shape[3]]
        high_freq = x - low_freq

        att_low = self.ca_low(low_freq)
        att_high = self.ca_high(high_freq)

        alpha = torch.sigmoid(self.alpha)
        fused = alpha * att_low + (1 - alpha) * att_high

        return x * fused


class FreqSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class FCBAM(nn.Module):
    def __init__(self, in_channels=None, reduction_ratio=16,
                 kernel_size=7, pool_size=2):
        super().__init__()
        self.fca = FreqChannelAttention(reduction_ratio, pool_size)
        self.sa = FreqSpatialAttention(kernel_size)

    def forward(self, x):
        x = self.fca(x)
        x = self.sa(x)
        return x
