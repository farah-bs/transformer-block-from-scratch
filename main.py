import torch
from TransformerBlock import TransformerBlock

batch_size = 2
seq_len = 10
model_dim = 512
num_heads = 8
ff_dim = 2048

block = TransformerBlock(model_dim=model_dim, num_heads=num_heads, ff_dim=ff_dim)

x = torch.randn(batch_size, seq_len, model_dim)
out = block(x)

assert out.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {out.shape}"
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print("Test passed.")
