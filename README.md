# Transformer From Scratch in PyTorch

This repository contains a **From-Scratch Implementation of the Transformer** model in PyTorch, based on the *“Attention Is All You Need”* paper (Vaswani et al., 2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

It builds the core components of the Transformer architecture step by step, without relying on high-level libraries.

## Features

* **Multi-Head Attention**
* **Positional Encoding**
* **Feed-Forward Network**
* **Encoder and Decoder**
* **Full Transformer Model**
* **Training and Evaluation Script (`main.py`)**

## File Structure

| File                    | Description                           |
| ----------------------- | ------------------------------------- |
| `MultiHeadAttention.py` | Multi-head self-attention module      |
| `PositionalEncoding.py` | Positional encoding implementation    |
| `FeedForwardNetwork.py` | Position-wise feed-forward layer      |
| `Encoder.py`            | Transformer encoder layers            |
| `Decoder.py`            | Transformer decoder layers            |
| `Transformer.py`        | Complete Transformer model            |
| `main.py`               | Training and evaluation driver script |
| `LICENSE`               | Repository license                    |

## Requirements

* Python 3.x
* PyTorch
* NumPy

## Paper Reference

“Attention Is All You Need” — Vaswani et al., 2017
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
