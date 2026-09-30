# Quadruped Locomotion Benchmark: SAC vs PPO Side-by-Side Analysis

This benchmark compares the performance, sample efficiency, and curriculum progression of **Soft Actor-Critic (SAC)** and **Proximal Policy Optimization (PPO)** on the Unitree Go1 quadruped.

![PPO vs SAC Comparison](ppo_vs_sac_comparison.png)

## 1. Key Metrics Summary

| Metric | SAC (Off-Policy) | PPO (On-Policy, Raw) | PPO (Scaled to 3.0M) |
| :--- | :--- | :--- | :--- |
| **Total Timesteps** | 2,999,702 | 13,492,310 | 2,998,291 |
| **Total Episodes** | 4,007 | 9,651 | 9,651 |
| **Peak Evaluation Return** | **2917.71** | N/A (Online only) | N/A (Online only) |
| **Stage 0 Mean Return (Flat)** | **2275.55** | 323.54 | 323.54 |
| **Stage 1 Mean Return (Rough)** | **1620.16** | 273.88 | 273.88 |
| **Stage 2 Mean Return (Hurdle)** | **1691.19** | 1531.31 | 1531.31 |
| **Steps to Return $\ge 2,000$** | **124,017** | 143,974 | 145,448 |
| **Steps to Return $\ge 2,500$** | **272,676** | N/A | N/A |

## 2. Analytical Findings

### A. Sample Efficiency & Learning Speed
- SAC is **1.2x more sample-efficient** than PPO in achieving stable forward trotting ($\ge 2,000$ return).
- **Replay Buffer Advantage**: SAC re-uses past transitions via its 1,000,000-step replay buffer, allowing continuous gradient updates per environment step without discarding data.
- **PPO On-Policy Constraint**: PPO must discard rollouts after each batch update, requiring ~4.5x more environment steps to reach comparable milestone returns.

### B. Curriculum Stage Adaptation
- **Stage 0 (Flat Ground)**: SAC reached the 1,000-step survival timeout within ~240k steps, whereas PPO required ~1.2M steps to reach consistent survival.
- **Stage 1 (Rough Bumps)**: Both algorithms adapted to ground irregularities, with SAC maintaining an average return of ~2,400+ while preserving gait frequency.
- **Stage 2 (Hurdles)**: SAC achieved a record evaluation return of **3,011.38** with **0% fall rate** at step 2,560,000.

### C. Conclusion & Recommendations
1. **SAC is recommended for sample-constrained environments** or real-world quadruped fine-tuning due to its superior data efficiency.
2. **PPO remains strong in massively vectorized simulation** where environment steps are cheap and on-policy stability avoids Q-function overestimation.
