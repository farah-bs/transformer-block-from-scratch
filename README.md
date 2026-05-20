# Transformer Block From Scratch in PyTorch

A single Transformer block implemented from scratch in PyTorch, without using the Hugging Face Transformers library.

Adapted from a full encoder-decoder implementation based on *"Attention Is All You Need"* (Vaswani et al., 2017).

## What this implements

A self-contained GPT-style Transformer block consisting of:

- **Multi-head scaled dot-product attention** with causal masking (generated internally from sequence length)
- **Position-wise feed-forward network** with GELU activation
- **Layer normalization** (post-LN: applied after each residual connection)
- **Residual connections**

## Design choices

**GELU over ReLU** — The original 2017 paper used ReLU. Modern transformer implementations (GPT-2, BERT) switched to GELU for smoother gradients near zero, which reduces dead neurons and improves training stability.

**Post-LN** — Layer normalization is applied after the residual addition: `LayerNorm(x + sublayer(x))`, following the original paper's formulation.

**Causal mask generated inside the block** — The mask is built from the input sequence length at forward time. This makes the block self-contained: no upstream class needs to construct or pass a mask.

## File structure

| File | Description |
|---|---|
| `TransformerBlock.py` | Single Transformer block — the main deliverable |
| `MultiHeadAttention.py` | Multi-head scaled dot-product attention |
| `FeedForwardNetwork.py` | Position-wise FFN with GELU |
| `main.py` | Test harness — runs the block on a small input and checks output shape |

## Running the test harness

```bash
python main.py
```

Expected output:
```
Input shape:  torch.Size([2, 10, 512])
Output shape: torch.Size([2, 10, 512])
Test passed.
```

## Paper reference

"Attention Is All You Need" — Vaswani et al., 2017
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
