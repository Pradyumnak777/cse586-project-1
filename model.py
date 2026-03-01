import torch
import torch.nn as nn

class NextLatentTransformer(nn.Module):
    def __init__(self, d_in=32, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, d_in)

    def forward(self, x):
        # x: (B, context, 32)
        h = self.in_proj(x)
        h = self.encoder(h)
        return self.out_proj(h[:, -1, :])  # (B, 32)