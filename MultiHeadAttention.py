import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module): 
    def __init__(self, model_dim, num_heads): # Inheriting all functionalities required to work with neural network layers
        super(MultiHeadAttention, self).__init__() 
        # Verifying that the model dimension is divisible by the number of heads
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads" 
        
        # Initializing the dimensions 
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads # Dimension of each head's key, value, and query
        
        # Defining the linear layers for transforming inputs 
        self.query= nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim, model_dim)
        
    def scaled_dot_product(self, Q, K, V, mask = None):
        # Matmul of Q and transposed K
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Reducing the magnitude of the scores
        scores = scores / math.sqrt(self.head_dim)
        
        # Applying the mask (if it exists) to the scores: it is used to prevent the model from attending to the padding tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Replacing the masked values with a large negative number
            
        # Applying the softmax function to the scores
        attention_weights = torch.softmax(scores, dim = -1)
        
        # Applying the attention to the values
        output = torch.matmul(attention_weights, V)
        
        return output
    
    def split_heads(self,x):
        # Reshaping the input tensor to have num_heads for multi-head attention
        batch_size, seq_len, model_dim = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def combine_heads(self, x):
        # Reshaping the input tensor to have the original shape
        batch_size, _, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        
    def forward(self, Q, K, V, mask = None): 
        # Applying the  linear transformations to q 
        Q = self.query(Q)
        
        # Splitting the heads of Q
        Q = self.split_heads(Q)
        
        # Applying the linear transformations and splitting the heads of K and V*
        K = self.split_heads(self.key(K))
        V = self.split_heads(self.value(V))
        
        # Applying the scaled dot product attention
        attn_output = self.scaled_dot_product(Q, K, V, mask)
        
        # Combining the heads
        attn_output = self.output(self.combine_heads(attn_output))
        
        return attn_output