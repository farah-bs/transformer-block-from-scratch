import torch
from TransformerBlock import TransformerBlock

block = TransformerBlock(model_dim=512, num_heads=8, ff_dim=2048)
block.eval()

x = torch.randn(2, 10, 512)

# 1. Output shape matches input shape
out = block(x)
assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
print("PASS  output shape:", out.shape)

# 2. Output is not identical to input — the block is doing something
assert not torch.allclose(out, x), "Block output is identical to input"
print("PASS  output differs from input")

# 3. Causal mask is enforced — changing a future token must not affect past token outputs
x2 = x.clone()
x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :])  # modify tokens 5-9
out2 = block(x2)
assert torch.allclose(out[:, :5, :], out2[:, :5, :], atol=1e-6), \
    "Causal mask broken: future tokens affected past outputs"
print("PASS  causal mask enforced: positions 0-4 unaffected by changes at 5-9")

# 4. Gradients flow back through the block
x_grad = torch.randn(2, 10, 512, requires_grad=True)
out_grad = block(x_grad)
out_grad.sum().backward()
assert x_grad.grad is not None, "No gradient on input"
assert not torch.all(x_grad.grad == 0), "Gradient is all zeros"
print("PASS  gradients flow through the block")

print("\nAll tests passed.")
