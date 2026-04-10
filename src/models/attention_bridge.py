import torch
import torch.nn as nn


class AttnBridge(nn.Module):
    """
    Transformer-style self-attention bridge between encoder and decoder.
    Uses Pre-LayerNorm for improved training stability and GELU activation.

    Improvements over original:
    - Pre-LayerNorm instead of Post-LayerNorm (more stable training)
    - GELU in projection layer (better gradients)
    - ff_mult=4 for more expressive feed-forward networks
    """

    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 nhead: int = 4,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 ff_mult: int = 4):
        super().__init__()
        self.proj_in = nn.Linear(enc_dim, dec_dim)
        self.layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.ff = nn.ModuleList()

        for _ in range(num_layers):
            # Pre-LayerNorm: norm before operations
            self.norms1.append(nn.LayerNorm(dec_dim))
            self.layers.append(nn.MultiheadAttention(dec_dim, nhead, dropout=dropout, batch_first=True))

            self.norms2.append(nn.LayerNorm(dec_dim))
            # ff_mult=4 is standard for Transformers (more expressive than 2)
            self.ff.append(nn.Sequential(
                nn.Linear(dec_dim, ff_mult * dec_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_mult * dec_dim, dec_dim)
            ))

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc_out: (batch, seq_len, enc_dim)
        Returns:
            (batch, seq_len, dec_dim)
        """
        # Project to decoder dimension (no activation; let attention layers handle it)
        x = self.proj_in(enc_out)  # (B, L, dec_dim)

        # Pre-LayerNorm pattern: norm first, then residual
        for ln1, mha, ln2, ffn in zip(self.norms1, self.layers, self.norms2, self.ff):
            # Self-attention block
            x_norm = ln1(x)
            y, _ = mha(x_norm, x_norm, x_norm, need_weights=False)
            x = x + y

            # Feed-forward block
            z = ffn(ln2(x))
            x = x + z

        return x
