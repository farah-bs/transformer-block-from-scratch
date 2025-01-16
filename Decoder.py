import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork

class Decoder(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout):
        super(Decoder, self).__init__()
        
        # Defining the Multi-head self-attention mechanism for the target sequence
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        
        # Defining the Multi-head attention mechanism that attends to the encoder's output
        self.cross_attn = MultiHeadAttention(model_dim, num_heads)
        
        # Defining the feedforward network
        self.feed_forward = FeedForwardNetwork(model_dim, ff_dim)
        
        # Defining the layer normalization, applied to smooth the layer's input
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)
        
        # Defining the dropout layer used for regularization, to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # enc_output: The output from the corresponding encoder (used in the cross-attention step)
        # src_mask: Source mask to ignore certain parts of the encoder's output
        # tgt_mask: Target mask to ignore certain parts of the decoder's input
        
        # Applying the self-attention mechanism to the input tensor
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        
        # Applying residual connection and layer normalization
        x = self.layer_norm1(x + self.dropout(self_attn_output))
        
        # Applying the multi-head attention mechanism to the encoder's output
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        
        # Applying a second residual connection and layer normalization
        x = self.layer_norm2(x + self.dropout(cross_attn_output))
        
        # Passing the output through the feedforward network
        ff_output = self.feed_forward(x)
        
        # Applying a third residual connection and layer normalization
        x = self.layer_norm3(x + self.dropout(ff_output))
        
        return x
        