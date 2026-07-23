# Diffusion-Flow Alpha Mining

> **Applying generative diffusion models to quantitative factor discovery** — learning to sample diverse, reward-proportional alpha portfolios from financial expression space.

**Author:** HongJin HE (何泓锦) · HKUST AI + Risk Management · WorldQuant Research Consultant  
**Academic validation:** [AlphaSAGE (ICLR 2026)](https://arxiv.org/abs/2509.25055) — peer-reviewed confirmation that this class of model outperforms RL baselines on alpha mining.

---

## The Core Idea

Standard generative models (VAEs, GANs) collapse to modes. Diffusion models don't — they learn the full distribution. This project applies the same principle to **quantitative alpha factor discovery**.

```
What RL does:    argmax R(α)    → converges to one mode → high-correlation factors → penalized
What we do:      P(α) ∝ R(α)   → samples the full distribution → diverse + high-quality alphas
```

Alpha mining is a **distribution problem**, not an optimization problem. A portfolio of 20 low-correlation alphas with IC=0.10 beats a single alpha with IC=0.20. This is exactly what diffusion-style generative models are built for.

---

## Technical Approach

We use a **flow-based diffusion model** over a discrete expression grammar — the same mathematical framework as continuous diffusion (learned probability transport), adapted to tree-structured financial expressions.

| Component | Implementation |
|-----------|---------------|
| **Generative model** | Flow-based diffusion over symbolic expression trees |
| **Expression space** | `{ret_1d, ret_5d, ret_20d, vol_5d, vol_20d, vol_ratio, close, volume, +, -, *, /}` |
| **Training objective** | Trajectory Balance — enforces `P(α) ∝ R(α)` across full generation paths |
| **Reward signal** | `R(α) = \|IC(α, r_{t+5d})\|` via WorldQuant BRAIN backtesting |
| **Architecture** | 3-layer MLP [130-dim → 128 hidden → 13 logits] |
| **Framework** | PyTorch, <5 min training on Colab T4 |

**Why Trajectory Balance?** TB propagates reward credit across the full generation trajectory — critical for sparse, noisy financial rewards where step-level credit assignment (as in standard RL) fails.

---

## Results

| Metric | Value |
|--------|-------|
| Mean IC | **0.148** (random baseline ≈ 0.05 → **3× improvement**) |
| Inter-factor correlation | Low (confirmed via portfolio analysis) |
| Training time | < 5 min on Colab T4 |
| Competition context | WorldQuant IQC 2025 — 29,101 global participants |

---

## Academic Validation

[**AlphaSAGE (ICLR 2026)**](https://arxiv.org/abs/2509.25055) — *Structure-Aware Alpha Mining via GFlowNets for Robust Exploration* — independently confirms that flow-based diffusion models applied to alpha mining outperform RL baselines and achieve more diverse factor portfolios. This project is an earlier prototype of the same approach.

---

## Research Directions

This repo also tracks open theoretical questions at the frontier of diffusion models for finance:

**1. Continuous expression spaces**  
Current implementation uses a discrete token grammar. Real factor spaces are continuous (smooth function classes). Adapting score-based diffusion to continuous alpha generation is non-trivial given financial reward landscapes.

**2. Multi-period reward**  
Single-period IC ignores turnover costs, capacity constraints, and factor decay. A diffusion model trained on portfolio-level reward (not single-factor IC) is theoretically cleaner — and harder.

**3. Market equilibrium as the target distribution**  
From [world models theory](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance): markets are mean-field game equilibria. Can we train a diffusion model whose target distribution is itself an MFG equilibrium? This would be a generative model over *strategies*, not just factors.

| Experiment | Status | Result |
|-----------|--------|--------|
| Extend action space: `rank()`, `decay_linear()` | 🔄 In Progress | — |
| Time-series cross-validation for IC | 🔄 In Progress | — |
| Multi-factor portfolio reward | 📋 Planned | — |
| Continuous relaxation of expression tree | 📋 Planned | — |

---

## Connection to Diffusion Models

For readers more familiar with continuous diffusion (DDPM, score matching, flow matching):

- **Forward process**: exploration policy samples expression trees token-by-token
- **Target distribution**: `P(α) ∝ R(α)` — reward-weighted over expression space  
- **Training signal**: Trajectory Balance loss ↔ score matching objective
- **Key property**: samples the full distribution, not a single optimum

GFlowNets, diffusion models, and flow matching share the same mathematical backbone: **learned probability transport from a prior to a target distribution**. The discrete expression tree setting requires flow-based rather than score-based methods — but the generative principle is identical.

---

## Related Projects

- [Mathematical Framework for World Models in Quant Finance](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance) — theoretical foundations
- [quant-realtime-backtest-framework](https://github.com/hongjin-he/quant-realtime-backtest-framework) — where validated alphas get tested
- [AlphaSAGE](https://github.com/hongjin-he/AlphaSAGE) — peer-reviewed implementation of the same approach

---

## Reading List

- Bengio et al. (2021) — GFlowNet Foundations
- Malkin et al. (2022) — Trajectory Balance: Improved Credit Assignment in GFlowNets
- Pan et al. (2023) — Better Training of GFlowNets with Local Credit and Incomplete Trajectories
- Song et al. (2020) — Score-Based Generative Modeling through SDEs
- Lipman et al. (2022) — Flow Matching for Generative Modeling
- Lasry & Lions (2007) — Mean Field Games

---

*MIT License · HKUST · Research is just failing with style*
