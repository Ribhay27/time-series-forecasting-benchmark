from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=1,
        horizon=24,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            horizon,
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        final_hidden = h_n[-1]
        return self.output_layer(final_hidden)
