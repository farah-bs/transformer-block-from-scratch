import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len): 
        super(PositionalEncoding, self).__init__()
        
        # Initializing the positional encoding tensor
        pe = torch.zeros(max_seq_len, model_dim) 
        
        # Defining the tensor that contains he position indices for each position in the sequence
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Defining the scaling factor for the positional encoding
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        
        # Applying the sine function to the even indices 
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Applying the cosine function to the odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Registering the positional encoding tensor as a buffer (is part of the model but not a trainable parameter)
        self.register_buffer('pe', pe.unsqueeze(0)) 
        
    def forward(self, x):
        # Adding the positional encoding to the input tensor
        return x + self.pe[:, :x.size(1)] # Slicing the positional encoding tensor to match the input tensor's sequence length