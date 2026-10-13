import torch
from torch import nn


class PatchTSTForecaster(nn.Module):
    def __init__(
        self,
        lookback=168,
        horizon=24,
        patch_len=24,
        stride=12,
        d_model=64,
        n_heads=4,
        num_layers=2,
        d_ff=128,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride

        self.num_patches = (
            (lookback - patch_len) // stride
        ) + 1

        self.patch_embedding = nn.Linear(
            patch_len,
            d_model,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches,
                d_model,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Linear(
            self.num_patches * d_model,
            horizon,
        )

    def forward(self, x):
        x = x.squeeze(-1)

        patches = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride,
        )

        x = self.patch_embedding(patches)

        x = x + self.position_embedding

        x = self.encoder(x)

        x = x.flatten(start_dim=1)

        return self.head(x)
