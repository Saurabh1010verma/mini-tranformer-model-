# Causal Multi-Head Self-Attention Guide

Multi-Head Self-Attention allows tokens to attend to information from different representation subspaces at different sequence positions.

## Scaled Dot-Product Attention Equation

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

Where:
- $Q \in \mathbb{R}^{B \times h \times T \times d_k}$: Query tensor
- $K \in \mathbb{R}^{B \times h \times T \times d_k}$: Key tensor
- $V \in \mathbb{R}^{B \times h \times T \times d_v}$: Value tensor
- $d_k$: Head dimension ($d_{\text{model}} / h$)
- $M$: Causal mask matrix

## Causal Masking

To enforce autoregressive sequence modeling (tokens cannot see future tokens during prediction), an upper-triangular causal mask $M$ is added before softmax:

$$M_{i,j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

After softmax, positions where $i < j$ receive an attention weight of $0$, ensuring strict causality.
