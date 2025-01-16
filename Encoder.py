import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork

class Encoder(nn.module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = FeedForwardNetwork(model_dim, ff_dim)
        
        # Defining the layer normalization, applied to smooth the layer's input.
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        
        # Defining the dropout layer used for regularization, to prevent overfitting 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Applying the self-attention mechanism to the input tensor
        attn_output = self.self_attn(x, x, x, mask) # x is passed as query, key, and value to the MultiHeadAttention layer because this is self-attention 
        
        # Applying residual connection and layer normalization
        x = self.norm1(x + self.dropout(attn_output)) 
        
        # Passing the output through the feedforward network
        ff_output = self.feed_forward(x)
        
        # Applying a second residual connection and layer normalization
        x = self.norm2(x + self.dropout(ff_output))
        
        return x        