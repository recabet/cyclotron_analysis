import torch
import torch.nn as nn


class AttnBridge(nn.Module):
    """
    Transformer-style self-attention bridge between encoder and decoder.
    Projects encoder output to decoder dimension and applies multiple
    self-attention layers with feed-forward networks.
    """

    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 nhead: int = 4,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 ff_mult: int = 2):
        super().__init__()
        self.proj_in = nn.Linear(enc_dim, dec_dim)
        self.layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.ff = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.MultiheadAttention(dec_dim, nhead, dropout=dropout, batch_first=True))
            self.norms1.append(nn.LayerNorm(dec_dim))
            self.ff.append(nn.Sequential(
                nn.Linear(dec_dim, ff_mult * dec_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_mult * dec_dim, dec_dim)
            ))
            self.norms2.append(nn.LayerNorm(dec_dim))

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_out: (batch, seq_len, enc_dim)
        Returns:
            (batch, seq_len, dec_dim)
        """
        x = torch.relu(self.proj_in(enc_out))  # (B, L, dec_dim)
        for mha, ln1, ffn, ln2 in zip(self.layers, self.norms1, self.ff, self.norms2):
            y, _ = mha(x, x, x, need_weights=False)
            x = ln1(x + y)
            z = ffn(x)
            x = ln2(x + z)
        return x
