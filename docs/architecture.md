# Mini GPT Architectural Overview

The Mini GPT language model is an **autoregressive decoder-only Transformer** built from scratch in PyTorch following the architecture popularized by GPT-2 and GPT-3.

## High-Level Pipeline

```
Raw Text ──> Tokenizer ──> Token IDs ──> Token Embedding + Positional Embedding
                                                      │
                                                      ▼
                              N × Transformer Blocks (Pre-LN + Causal MHA + FFN)
                                                      │
                                                      ▼
                                              Final LayerNorm
                                                      │
                                                      ▼
                                            Linear LM Head (Logits)
                                                      │
                                                      ▼
                                            Next-Token Softmax & Sampling
```

## Component Details

### 1. Embeddings (`src/mini_gpt/model/embeddings.py`)
- **Token Embedding**: Maps discrete token indices $i \in \{0, \dots, V-1\}$ to continuous $d_{\text{model}}$-dimensional vectors.
- **Positional Embedding**: Maps sequence positions $t \in \{0, \dots, T-1\}$ to continuous $d_{\text{model}}$-dimensional vectors.
- **Combination**: Input representation $E = \text{Dropout}(\text{TokenEmb}(X) + \text{PosEmb}(P))$.

### 2. Transformer Block (`src/mini_gpt/model/transformer_block.py`)
Modern Pre-Layer Normalization (Pre-LN) layout:
$$\mathbf{x}^{(1)} = \mathbf{x} + \text{CausalSelfAttention}(\text{LayerNorm}(\mathbf{x}))$$
$$\mathbf{x}^{(2)} = \mathbf{x}^{(1)} + \text{FeedForward}(\text{LayerNorm}(\mathbf{x}^{(1)}))$$

Pre-LN ensures stable gradient flow during deep network training without requiring intricate learning rate warmup tricks.

### 3. Language Modeling Head (`src/mini_gpt/model/gpt.py`)
- Standard linear transformation projecting $d_{\text{model}}$ hidden states to $V$ vocabulary logits.
- Supports **weight tying**: sharing weights between `token_embedding` matrix and `lm_head` matrix ($W_{\text{head}} = W_{\text{emb}}^T$).
