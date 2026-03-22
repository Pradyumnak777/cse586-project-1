import torch
import torch.nn as nn
import math

#this adds "time stamps" to your frames
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) #sinusoidal pos embeddings
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) #shape: (1, max_len, d_model)

    def forward(self, x):
        #x: (B, seq_len, d_model)
        #add the positional embedding to the input
        x = x + self.pe[:, :x.size(1), :]
        return x

class NextLatentTransformer(nn.Module):
    def __init__(self, d_in=32, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        
        #instantiate the positional encoder
        self.pos_encoder = PositionalEncoding(d_model)

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
        #x: (B, context, 32)
        
        last_frame = x[:, -1, :] 

        h = self.in_proj(x)
        h = self.pos_encoder(h) #inject pos embeddings here...
        h = self.encoder(h)
        
        #the transformer now only predicts the "change" or "velocity"
        delta = self.out_proj(h[:, -1, :])  
        
        #add the change to the last known frame
        return last_frame + delta