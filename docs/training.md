# Mini GPT Training & Optimization Guide

This document details the training pipeline, optimization strategies, and metrics used in Mini GPT.

## Cross-Entropy Loss & Perplexity

The model is trained using standard autoregressive next-token Cross-Entropy Loss:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$

**Perplexity (PPL)** measures how well the probability distribution predicts the sample:

$$\text{PPL} = \exp(\mathcal{L})$$

A lower perplexity indicates better predictive performance.

## Optimization Strategies

1. **AdamW Optimizer**:
   - $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$
   - Weight decay of $0.1$ applied only to 2D weight matrices (Linear, Embedding).
   - 1D parameters (biases, LayerNorm affine weights) are excluded from weight decay.

2. **Learning Rate Schedule**:
   - Linear warmup phase for $N_{\text{warmup}}$ steps.
   - Cosine decay phase scaling LR down to $10\%$ of peak LR.

3. **Gradient Clipping**:
   - Norm of gradients is clipped at $1.0$ (`torch.nn.utils.clip_grad_norm_`) to prevent exploding gradients.

4. **Automatic Mixed Precision (AMP)**:
   - Evaluates forward pass in FP16/BF16 on CUDA devices to accelerate compute and reduce VRAM footprint.
