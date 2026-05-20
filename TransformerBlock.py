import torch
import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = FeedForwardNetwork(model_dim, ff_dim)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        # Causal mask: lower triangle is 1 (allowed), upper triangle is 0 (future — blocked)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()

        attn_output = self.self_attn(x, x, x, mask=causal_mask)
        x = self.layer_norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x
