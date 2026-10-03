import torch
from torch import nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        x = x.unsqueeze(1)
        x = F.pad(x, (padding, padding), mode="replicate")
        trend = F.avg_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
        )
        return trend.squeeze(1)


class DLinear(nn.Module):
    def __init__(self, lookback=168, horizon=24, kernel_size=25):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size=kernel_size)
        self.trend_layer = nn.Linear(lookback, horizon)
        self.remainder_layer = nn.Linear(lookback, horizon)

    def forward(self, x):
        trend = self.moving_average(x)
        remainder = x - trend

        trend_forecast = self.trend_layer(trend)
        remainder_forecast = self.remainder_layer(remainder)

        return trend_forecast + remainder_forecast
