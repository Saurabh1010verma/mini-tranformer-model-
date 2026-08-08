# Autoregressive Text Generation & Sampling Guide

Mini GPT generates text token-by-token autoregressively.

## Generation Pipeline

```
Prompt ──> Tokenize ──> Model Forward ──> Extract Last Logits ──> Temperature Scale
                                                                         │
                                                                         ▼
Output Text <── Decode <── Append Token <── Sample <── Softmax <── Top-K Filter
```

## Sampling Strategies

### 1. Greedy Decoding ($T = 0.0$)
Selects the token with maximum probability at each step:
$$x_{t} = \arg\max_{v} z_v$$

### 2. Temperature Scaling ($T > 0$)
Adjusts output logit variance before applying Softmax:
$$p_v = \frac{\exp(z_v / T)}{\sum_w \exp(z_w / T)}$$
- **High Temperature ($T > 1.0$)**: Flattens distribution, increasing diversity and randomness.
- **Low Temperature ($T < 1.0$)**: Sharpens distribution, focusing probability on top candidates.

### 3. Top-K Sampling
Restricts the candidate token pool to the $K$ highest-probability logits, setting all other logits to $-\infty$ prior to Softmax:
$$z'_v = \begin{cases} z_v & \text{if } z_v \in \text{TopK}(z) \\ -\infty & \text{otherwise} \end{cases}$$
This prevents low-probability out-of-context tokens from being sampled.
