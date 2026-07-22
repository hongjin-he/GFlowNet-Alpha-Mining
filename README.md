# GFlowNet Alpha Mining

**Author**: HongJin HE (何泓锦) · HKUST AI + Risk Management · Stanford CS Exchange 2026  
**Affiliation**: WorldQuant Research Consultant (Alpha Mining) · Alpha Flow Co-founder

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IC](https://img.shields.io/badge/Mean_IC-0.148-brightgreen.svg)](#results)

## The Core Idea

Standard RL-based alpha search has a fundamental flaw: it converges to **one** local optimum. Production quant strategies need a **portfolio** of low-correlation factors — which means you need diverse sampling, not maximization.

GFlowNets fix this by treating alpha discovery as a flow problem. The network learns to sample formulaic alpha expressions with probability proportional to their reward (IC), which by construction generates diverse, high-quality factors.

```
Traditional RL: maximize R → single best alpha, high correlation risk
GFlowNet:       P(α) ∝ R(α)  → portfolio of diverse alphas
```

This is the exploration-exploitation problem solved at the distribution level, not the trajectory level.

## Implementation

### MDP Formulation

```
State:      Partial alpha expression tree (sequence of tokens)
Actions:    {ret_1d, ret_5d, ret_20d, vol_5d, vol_20d,
             vol_ratio, close, volume, +, -, *, /, STOP}
Termination: STOP action or max depth
Reward:     |IC(α, r_{t+5d})|  — absolute Information Coefficient
```

### Architecture

```
Forward policy:  3-layer MLP  [130-dim → 128 hidden → 13 logits]
Loss:            Trajectory Balance (TB)    — guarantees flow conservation
Optimizer:       Adam  lr=1e-3
Training:        500 episodes + ε-greedy exploration
```

The TB loss is the right choice here: it enforces exact flow balance at every state, giving stable gradients even with sparse rewards.

## Results

```
Mean IC:       0.148   (random baseline ≈ 0.05, 3× improvement)
Training time: < 5 min on Google Colab T4
```

Factor diversity: proportional sampling produces expressions with low inter-factor correlation — the mechanism works as intended.

## Known Limitations (Honest Assessment)

| Issue | Impact | Fix |
|---|---|---|
| No train/test split | IC likely 20-40% optimistically biased OOS | Time-series cross-validation |
| Sequence encoding | Loses expression tree structure | RGCN (AlphaSAGE approach) |
| Single objective (IC only) | Ignores turnover, capacity, risk | Multi-objective reward |
| Small dataset (5 stocks, 4yr) | Overfitting risk | Scale to Qlib universe |
| `eval()` for expression parsing | Security + correctness | Proper AST parser |

The limitation section exists because understanding *why* a result is limited is the precondition for improving it.

## Roadmap

- [ ] Time-series CV (ISO 3-fold forward chaining)
- [ ] RGCN encoder for structure-aware expression representation
- [ ] Multi-objective: `R = IC + λ·Sharpe - γ·turnover - δ·correlation`
- [ ] Distributional GFlowNet for risk-aware generation (Pan et al. 2025)
- [ ] Scale to full WorldQuant BRAIN expression universe

## References

- Bengio et al. (2021) — [GFlowNet Foundations](https://arxiv.org/abs/2111.09266)
- Pan et al. (2024) — Local Credit Assignment for GFlowNets
- Chen et al. (2025) — [AlphaSAGE](https://arxiv.org/abs/2509.25055): structure-aware alpha generation

## Quick Start

```bash
git clone https://github.com/Abstractor-bit/GFlowNet-Alpha-Mining.git
cd GFlowNet-Alpha-Mining
pip install torch numpy pandas yfinance jupyter
jupyter notebook notebooks/Alpha_GFN_v1_Baseline.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/github/Abstractor-bit/GFlowNet-Alpha-Mining/blob/main/notebooks/Alpha_GFN_v1_Baseline.ipynb).

---

*Part of a larger research program on world models in quantitative finance. See [Mathematical Framework for World Models](https://github.com/Abstractor-bit/mathmatical-framework-for-world-models-in-quant-finance) for the theoretical foundation.*
