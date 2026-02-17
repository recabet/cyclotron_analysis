import torch.nn as nn
from src.models.attention_bridge import AttnBridge


class LSTMSeq2Seq(nn.Module):
    """
    Encoder-Decoder LSTM for sequence-to-sequence super-resolution.
    Features:
    - Bidirectional LSTM encoder
    - Optional self-attention bridge (Transformer-style) or linear projection
    - Unidirectional LSTM decoder
    - Pointwise output head
    """

    def __init__(self,
                 in_dim: int = 1,
                 enc_hidden: int = 128,
                 enc_layers: int = 2,
                 dec_hidden: int = None,
                 dec_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 use_attn_bridge: bool = True,
                 attn_heads: int = 16,
                 attn_layers: int = 2):

        super().__init__()

        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.enc_hidden = enc_hidden


        self.encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=enc_hidden,
            num_layers=enc_layers,
            batch_first=True,
            dropout=dropout if enc_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        enc_out_dim = enc_hidden * self.num_dirs
        if dec_hidden is None:
            dec_hidden = enc_hidden

        # Bridge
        if use_attn_bridge:
            self.bridge = AttnBridge(enc_out_dim,
                                     dec_hidden,
                                     nhead=attn_heads,
                                     num_layers=attn_layers,
                                     dropout=dropout)
        else:
            self.bridge = nn.Linear(enc_out_dim, dec_hidden)
            nn.init.xavier_uniform_(self.bridge.weight)
            nn.init.zeros_(self.bridge.bias)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=dec_hidden,
            hidden_size=dec_hidden,
            num_layers=dec_layers,
            batch_first=True,
            dropout=dropout if dec_layers > 1 else 0.0
        )

        # Output head
        self.head = nn.Linear(dec_hidden, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, in_dim)
        Returns:
            (batch, seq_len, 1)
        """
        # Encode
        enc_out, _ = self.encoder(x)  # (B, L, enc_hidden * num_dirs)

        # Bridge: project to decoder space
        z = self.bridge(enc_out)  # (B, L, dec_hidden)

        # Decode (many-to-many aligned)
        dec_out, _ = self.decoder(z)  # (B, L, dec_hidden)

        # Predict
        y_hat = self.head(dec_out)  # (B, L, 1)
        return y_hat
