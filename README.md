# Transformer Block From Scratch in PyTorch

This repository implements a single Transformer block from scratch in PyTorch, without using the Hugging Face Transformers library.

It is based on the core ideas from *Attention Is All You Need* by Vaswani et al. (2017), but reduced to the smallest useful unit.

## What the repo implements

The code contains one self-contained Transformer block with these parts:

- Multi-head scaled dot-product attention with causal masking
- Position-wise feed-forward network with GELU activation
- Layer normalization using post-LN, meaning normalization happens after each residual addition
- Residual connections around both sublayers

## Why these choices were made

- GELU is used instead of ReLU because it is smoother near zero and is common in modern Transformer implementations.
- Post-LN is used because it is easy to explain and matches the pattern `LayerNorm(x + sublayer(x))`.
- The causal mask is created inside the block from the input sequence length, so the module stays self-contained and does not require an external mask argument.

## Files

| File | Purpose |
|---|---|
| `TransformerBlock.py` | Main Transformer block implementation |
| `MultiHeadAttention.py` | Multi-head attention with scaled dot-product attention and masking |
| `FeedForwardNetwork.py` | Feed-forward network with GELU |
| `main.py` | Minimal harness that runs the block on a sample input and checks the shape |
| `test_block.py` | Extra test script that checks shape, non-trivial output, causal masking, and gradient flow |

## Setup with uv

The repository is configured for `uv` and creates a local virtual environment in `.venv`.

Install the dependencies:

```bash
uv sync
```

This reads `pyproject.toml`, creates `.venv`, and installs the declared packages, including `torch` and `numpy`.

If you want to use the environment directly, run commands through `uv`:

```bash
uv run python main.py
uv run python test_block.py
```

## How to run the tests

Run the minimal harness:

```bash
uv run python main.py
```

Run the fuller test script:

```bash
uv run python test_block.py
```

## What the tests are for

The tests are intentionally small because the exercise is about a correct forward pass, not training.

- `main.py` checks that the block accepts a tensor of shape `[batch_size, seq_len, model_dim]` and returns the same shape.
- `test_block.py` adds a few stronger checks:
	- the output shape matches the input shape
	- the block does not return the input unchanged
	- the causal mask prevents future tokens from affecting earlier outputs
	- gradients flow back through the block

## Expected output

When you run the scripts, you should see output similar to this:

```text
Input shape:  torch.Size([2, 10, 512])
Output shape: torch.Size([2, 10, 512])
Test passed.
```

## Paper reference

"Attention Is All You Need" by Vaswani et al. (2017)

https://arxiv.org/abs/1706.03762
