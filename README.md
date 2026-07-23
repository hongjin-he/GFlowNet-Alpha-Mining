<div align="center">

# GFlowNet Alpha Mining

**Applied GFlowNet-based alpha factor discovery to the WorldQuant International Quant Championship — because if a paper says "diverse sampling beats mode collapse," you should probably try it on a competition where diversity is literally scored.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IC](https://img.shields.io/badge/Mean_IC-0.148-brightgreen.svg)](#results)
[![Competition](https://img.shields.io/badge/WorldQuant-IQC-orange.svg)](https://www.worldquant.com/brain/iqc/)

*Author: HongJin HE (何泓锦) · HKUST AI + Risk Management · WorldQuant Research Consultant*

</div>

---

## Background: The Competition

This project was built for the **[WorldQuant International Quant Championship (IQC)](https://www.worldquant.com/brain/iqc/)** — a global competition where participants submit "alphas" (formulaic predictors of future stock returns) to the [WorldQuant BRAIN](https://www.worldquant.com/brain/) simulation platform.

**Scale of the competition:**
- 29,101 global participants in 2025
- Top 1.3% advance to Round 2 (~380 teams)
- Three elimination rounds, regional then international finals

**What you're actually optimizing:**

```
Fitness = sqrt( |Returns| / max(turnover, 0.125) ) × Sharpe
```

Not just Sharpe. Not just IC. **Fitness** — a compound metric that penalizes excessive turnover and rewards consistency. And critically: the leaderboard explicitly scores *baskets* of alphas, meaning **low-correlation, diverse factor portfolios are rewarded over any single high-IC alpha**.

This last point is why GFlowNets are the right tool.

---

## Why GFlowNets for Alpha Mining

Standard RL for alpha search has a fatal alignment problem with how IQC scores teams:

```
What RL does:    argmax R(α)      → converges to one mode → high-correlation factors → penalized
What IQC wants:  diverse(α)       → spread across factor space → low correlation → rewarded
What GFlowNets do: P(α) ∝ R(α)   → samples proportionally to reward → diverse + high-quality
```

GFlowNets learn a **flow** over the expression space — similar in spirit to diffusion models, which also model distributions rather than single optima. The network learns to distribute probability mass proportional to reward, which by construction prevents mode collapse and generates a portfolio of decorrelated factors.

This connection to flow-based generative models (normalizing flows, diffusion) is not coincidental: GFlowNets, diffusion models, and flow matching all share the same mathematical backbone of learned probability transport.

---

## How It Works

```mermaid
graph TD
    A[Start: Empty Expression Tree] --> B{Forward Policy π_F}
    B --> C[Operand tokens<br/>ret_1d · ret_5d · vol · close · volume]
    B --> D[Operator tokens<br/>+ · - · * · /]
    B --> E[STOP]
    C --> B
    D --> B
    E --> F[Complete Alpha Expression α]
    F --> G[Backtest on BRAIN universe<br/>compute IC vs r_t+5d]
    G --> H[Reward = |IC|]
    H --> I[Trajectory Balance Loss<br/>enforce flow conservation]
    I --> J[Update π_F and π_B]
    J --> K[P_T∝R enforced → diverse sampling]
    K --> B
```

### MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** | Partial alpha expression tree (token sequence) |
| **Actions** | `{ret_1d, ret_5d, ret_20d, vol_5d, vol_20d, vol_ratio, close, volume, +, -, *, /, STOP}` |
| **Termination** | STOP action or max depth reached |
| **Reward** | `|IC(α, r_{t+5d})|` — absolute Information Coefficient |

### Architecture

```python
Forward policy π_F:   3-layer MLP  [130-dim → 128 hidden → 13 logits]
Backward policy π_B:  learnable (not uniform)
Loss:                 Trajectory Balance (TB) — exact flow conservation guarantee
Optimizer:            Adam, lr=1e-3
Training:             500 episodes + ε-greedy exploration
```

**Why Trajectory Balance?** TB enforces `π_F(τ) / π_B(τ) = R(x) / Z` at every trajectory — guaranteeing the stationary distribution over complete expressions matches the reward. Compared to DB (Detailed Balance) loss, TB is more stable with sparse, noisy financial rewards because it propagates credit across the full trajectory rather than step-by-step.

---

## Results

```
Mean IC:       0.148   (random baseline ≈ 0.05 → 3× improvement)
Training:      < 5 min on Google Colab T4
Diversity:     low inter-factor correlation confirmed via portfolio analysis
```

> Reference: what a real diverse-alpha backtest portfolio looks like (Microsoft Qlib, open-source benchmark):

![Backtest Reference](https://raw.githubusercontent.com/microsoft/qlib/main/docs/_static/img/analysis/analysis_model_cumulative_return.png)

*The GFlowNet-mined factor portfolio targets this kind of cumulative curve — smooth, consistent, driven by decorrelated signals rather than a single over-fitted alpha.*

**Note on competition data:** The original IQC submission data is no longer available. The IC=0.148 figure comes from the research prototype on simulated price data. Competition-time performance on the BRAIN platform's actual universe was measured by the Fitness metric above, not raw IC.

---

## Academic Validation

If you're skeptical that this approach works at scale: **[AlphaSAGE (ICLR 2026)](https://arxiv.org/abs/2509.25055)** is a formal academic paper that independently developed the same core idea and validated it rigorously.

| | This Project | AlphaSAGE (ICLR 2026) |
|--|--|--|
| **Core mechanism** | GFlowNet over expression tree | GFlowNet over expression tree |
| **Encoder** | MLP on token sequence | RGCN (graph-aware, structure-sensitive) |
| **Reward** | IC | Multi-signal (IC + novelty + entropy) |
| **Diversity mechanism** | Proportional sampling via TB | Explicit novelty pressure |
| **Evaluation** | Simulated / IQC | CSI300, CSI500, S&P500 |
| **Status** | Research prototype | ICLR 2026 poster |

AlphaSAGE's three core innovations over a naive GFlowNet like this one:
1. **RGCN encoder** — treats the alpha as a *graph*, not a sequence, capturing mathematical structure (commutativity, associativity) that MLP misses
2. **Dense reward** — intermediate feedback at each token step, not just at STOP (fixes reward sparsity)
3. **Novelty pressure** — explicit penalty for re-discovering similar expressions

This project essentially proves the mechanism works. AlphaSAGE proves it works well.

---

## Limitations (Honest)

| Issue | Impact |
|-------|--------|
| No train/test split | IC likely 20-40% optimistically biased OOS |
| Simulated price data | BRAIN universe uses 85K+ real data types; this uses ~8 |
| Small action space | Misses `rank()`, `decay_linear()`, `neutralize()` — all standard IQC operators |
| 500 training episodes | BRAIN consultants test thousands of alphas manually |
| MLP encoder | Misses algebraic structure (e.g., `a+b = b+a` should map to same representation) |

For a production-grade version, see [AlphaSAGE](https://github.com/BerkinChen/AlphaSAGE). For the theory behind why this fits financial markets, see [my world models paper](https://github.com/hongjin-he/mathmatical-framework-for-world-models-in-quant-finance).

---

## Running It

```bash
git clone https://github.com/hongjin-he/GFlowNet-Alpha-Mining.git
cd GFlowNet-Alpha-Mining
pip install torch numpy pandas jupyter
jupyter notebook
```

Open the notebook and run all cells. IC improves noticeably over 500 episodes as the flow network learns which regions of expression space are worth sampling from.

---

## Related Work

- [AlphaSAGE](https://arxiv.org/abs/2509.25055) — ICLR 2026, the state-of-the-art version of this idea
- [alpha-gfn](https://github.com/nshen7/alpha-gfn) — another open-source GFlowNet alpha mining implementation
- [AlphaAgent](https://arxiv.org/html/2502.16789v2) — LLM-driven alpha mining (different approach, same problem)
- [GenFlowNet Deep Research](https://github.com/hongjin-he/GenFlowNet_Deep_Research) — where the open theoretical questions from this project live

---

## Citation

```bibtex
@misc{he2026gflownetalpha,
  author  = {He, Hongjin},
  title   = {GFlowNet Alpha Mining: Diverse Alpha Factor Discovery for WorldQuant IQC},
  year    = {2026},
  url     = {https://github.com/hongjin-he/GFlowNet-Alpha-Mining}
}
```

---

<div align="center">
<sub>MIT License · HKUST × WorldQuant · AlphaSAGE did it better, but we got here first</sub>
</div>
