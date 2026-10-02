"""Side-by-Side Comparison: PPO vs SAC vs TD3 for Unitree Go1 Locomotion Curriculum.

Loads TensorBoard logs for all three algorithms, applies curriculum-aligned timestep
scaling to PPO, and generates a comprehensive 3-way comparison dashboard PNG.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUTPUT_DIR = os.path.join(REPO_ROOT, "src", "visualizations")
SAC_LOGDIR  = os.path.join(REPO_ROOT, "src", "sac_baseline", "logs", "sac_go1_curriculum")
TD3_LOGDIR  = os.path.join(REPO_ROOT, "src", "td3_baseline", "logs", "td3_go1_curriculum")
PPO_LOGDIR  = os.path.join(REPO_ROOT, "src", "visualizations", "tensorboard_logs", "ppo_raw")

PPO_MAX_STEPS = 13_500_000.0
SAC_MAX_STEPS =  3_000_000.0   # reference scale for alignment
TD3_MAX_STEPS = 10_500_000.0   # TD3 full curriculum budget


def moving_average(data, w=50):
    if len(data) < w:
        return np.array(data)
    return np.convolve(data, np.ones(w) / w, mode="valid")


def load_tb(logdir):
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = {
            "steps":  np.array([e.step  for e in events]),
            "values": np.array([e.value for e in events]),
        }
    return data


def extract_return(tb_data, scale=1.0):
    """Return (steps_scaled, return_vals) normalised to 1000-step horizon."""
    ret = tb_data.get("train/episode_return")
    lен = tb_data.get("train/episode_length")
    if ret is None:
        return np.array([]), np.array([])
    steps = ret["steps"] * scale
    vals  = ret["values"]
    if lен is not None:
        lens = lен["values"]
        vals = np.where(lens >= 1000.0, vals * (1000.0 / np.maximum(lens, 1.0)), vals)
    return steps, vals


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading TensorBoard logs …")
    sac_tb = load_tb(SAC_LOGDIR)
    td3_tb = load_tb(TD3_LOGDIR)
    ppo_tb = load_tb(PPO_LOGDIR)
    print(f"  SAC tags : {list(sac_tb.keys())}")
    print(f"  TD3 tags : {list(td3_tb.keys())}")
    print(f"  PPO tags : {list(ppo_tb.keys())}")

    # Scale factors so all curves share the SAC 3 M reference axis
    ppo_scale = SAC_MAX_STEPS / PPO_MAX_STEPS   # ≈ 0.222
    td3_scale = SAC_MAX_STEPS / TD3_MAX_STEPS   # ≈ 0.286

    sac_steps, sac_vals = extract_return(sac_tb, 1.0)
    td3_steps, td3_vals = extract_return(td3_tb, td3_scale)
    ppo_steps, ppo_vals = extract_return(ppo_tb, ppo_scale)

    W = 50  # smoothing window
    def smooth(steps, vals):
        sv = moving_average(vals, W)
        ss = steps[W - 1:] if len(steps) >= W else steps
        return ss, sv

    sac_ss, sac_sv = smooth(sac_steps, sac_vals)
    td3_ss, td3_sv = smooth(td3_steps, td3_vals)
    ppo_ss, ppo_sv = smooth(ppo_steps, ppo_vals)

    # ── colours ──────────────────────────────────────────────────────────────
    C_SAC = "#1f77b4"   # blue
    C_TD3 = "#2ca02c"   # green
    C_PPO = "#ff7f0e"   # orange

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1: Comprehensive 6-panel dashboard
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle(
        "Unitree Go1 Quadruped: PPO vs SAC vs TD3 — Three-Algorithm Curriculum Comparison",
        fontsize=16, fontweight="bold", y=0.998,
    )

    # Stage boundary lines (aligned 3 M scale)
    STAGE1 = 1_000_000
    STAGE2 = 2_000_000

    # ── Panel 1: Episode returns (aligned) ───────────────────────────────────
    ax = axes[0, 0]
    if len(sac_steps):
        ax.scatter(sac_steps, sac_vals, color=C_SAC, alpha=0.1, s=5)
        ax.plot(sac_ss, sac_sv,  color=C_SAC, lw=2.5, label=f"SAC (3.0 M steps)")
    if len(td3_steps):
        ax.scatter(td3_steps, td3_vals, color=C_TD3, alpha=0.1, s=5)
        ax.plot(td3_ss, td3_sv,  color=C_TD3, lw=2.5, label=f"TD3 (scaled from 10.5 M)")
    if len(ppo_steps):
        ax.scatter(ppo_steps, ppo_vals, color=C_PPO, alpha=0.1, s=5)
        ax.plot(ppo_ss, ppo_sv,  color=C_PPO, lw=2.5, label=f"PPO (scaled from 13.5 M)")
    ax.axvline(STAGE1, color="gray",   ls="--", lw=1.5, alpha=0.8, label="Stage 1 – Rough")
    ax.axvline(STAGE2, color="purple", ls="--", lw=1.5, alpha=0.8, label="Stage 2 – Hurdle")
    ax.set_title("1. Training Returns (Aligned Curriculum Steps)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Aligned Timesteps (3 M reference)")
    ax.set_ylabel("Episode Return")
    ax.set_ylim(-1500, 3500)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # ── Panel 2: Normalised progress (0–100 %) ───────────────────────────────
    ax = axes[0, 1]
    def to_pct(steps, max_steps):
        return steps / max_steps * 100.0

    if len(sac_steps):
        pct = to_pct(sac_ss, SAC_MAX_STEPS)
        ax.plot(pct, sac_sv, color=C_SAC, lw=2.5, label="SAC")
    if len(td3_steps):
        pct = to_pct(td3_ss * TD3_MAX_STEPS / SAC_MAX_STEPS, TD3_MAX_STEPS)  # un-scale
        ax.plot(pct, td3_sv, color=C_TD3, lw=2.5, label="TD3")
    if len(ppo_steps):
        raw_ppo_ss = ppo_ss * PPO_MAX_STEPS / SAC_MAX_STEPS
        pct = to_pct(raw_ppo_ss, PPO_MAX_STEPS)
        ax.plot(pct, ppo_sv, color=C_PPO, lw=2.5, label="PPO")
    ax.axvline(33.33, color="gray",   ls="--", lw=1.5, alpha=0.8, label="Stage 1 (33 %)")
    ax.axvline(66.66, color="purple", ls="--", lw=1.5, alpha=0.8, label="Stage 2 (67 %)")
    ax.set_title("2. Learning Curve vs Normalised Curriculum Progress", fontsize=12, fontweight="bold")
    ax.set_xlabel("Curriculum Progress (%)")
    ax.set_ylabel("Episode Return (MA-50)")
    ax.set_ylim(-500, 3200)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # ── Panel 3: Episode length / survival ──────────────────────────────────
    ax = axes[1, 0]
    for tag, tb, scale, col, label in [
        ("train/episode_length", sac_tb, 1.0,       C_SAC, "SAC Episode Length"),
        ("train/episode_length", td3_tb, td3_scale, C_TD3, "TD3 Episode Length (scaled)"),
        ("train/episode_length", ppo_tb, ppo_scale, C_PPO, "PPO Episode Length (scaled)"),
    ]:
        d = tb.get(tag)
        if d is not None:
            sv = moving_average(d["values"], W)
            ss = d["steps"][W - 1:] * scale if len(d["steps"]) >= W else d["steps"] * scale
            ax.plot(ss, sv, color=col, lw=2.5, label=label)
    ax.axhline(1000, color="green", ls=":", lw=1.5, alpha=0.7, label="Max episode limit (1000)")
    ax.axvline(STAGE1, color="gray",   ls="--", lw=1.2, alpha=0.6)
    ax.axvline(STAGE2, color="purple", ls="--", lw=1.2, alpha=0.6)
    ax.set_title("3. Survival / Episode Duration", fontsize=12, fontweight="bold")
    ax.set_xlabel("Aligned Timesteps")
    ax.set_ylabel("Steps Upright")
    ax.set_ylim(0, 1100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # ── Panel 4: Raw (actual) sample footprint ───────────────────────────────
    ax = axes[1, 1]
    if len(sac_steps):
        sac_raw_ss = sac_ss / 1e6
        ax.plot(sac_raw_ss, sac_sv, color=C_SAC, lw=2.5, label="SAC (3.0 M actual)")
    if len(td3_steps):
        td3_raw_ss = td3_ss * TD3_MAX_STEPS / SAC_MAX_STEPS / 1e6
        ax.plot(td3_raw_ss, td3_sv, color=C_TD3, lw=2.5, label="TD3 (10.5 M actual)")
    if len(ppo_steps):
        ppo_raw_ss = ppo_ss * PPO_MAX_STEPS / SAC_MAX_STEPS / 1e6
        ax.plot(ppo_raw_ss, ppo_sv, color=C_PPO, lw=2.5, label="PPO (13.5 M actual)")
    ax.set_title("4. Sample Footprint: Return vs Actual Env Steps", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Environment Steps (M)")
    ax.set_ylabel("Episode Return (MA-50)")
    ax.set_ylim(-500, 3200)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # ── Panel 5: Stage-by-stage box plots ────────────────────────────────────
    ax = axes[2, 0]
    stages = ["Stage 0\n(Flat)", "Stage 1\n(Rough)", "Stage 2\n(Hurdle)"]

    def bucket_by_stage(steps_raw, vals, boundaries):
        """boundaries = (s0_end, s1_end) in raw-step space."""
        s0e, s1e = boundaries
        buckets = [[], [], []]
        for s, v in zip(steps_raw, vals):
            if s < s0e:
                buckets[0].append(v)
            elif s < s1e:
                buckets[1].append(v)
            else:
                buckets[2].append(v)
        return buckets

    sac_buckets = bucket_by_stage(sac_steps, sac_vals, (1_000_000, 2_000_000))
    td3_buckets = bucket_by_stage(
        td3_steps * TD3_MAX_STEPS / SAC_MAX_STEPS, td3_vals,
        (3_500_000, 7_000_000),
    )
    ppo_ret = ppo_tb.get("train/episode_return")
    ppo_len = ppo_tb.get("train/episode_length")
    if ppo_ret is not None and ppo_len is not None:
        ppo_raw_vals = np.where(
            ppo_len["values"] >= 1000,
            ppo_ret["values"] * (1000.0 / np.maximum(ppo_len["values"], 1)),
            ppo_ret["values"],
        )
    else:
        ppo_raw_vals = np.array([])
    ppo_raw_steps = ppo_ret["steps"] if ppo_ret is not None else np.array([])
    ppo_buckets = bucket_by_stage(ppo_raw_steps, ppo_raw_vals, (4_500_000, 9_000_000))

    positions = {"sac": [1, 4, 7], "td3": [2, 5, 8], "ppo": [3, 6, 9]}
    colors    = {"sac": C_SAC,      "td3": C_TD3,      "ppo": C_PPO}
    labels    = {"sac": "SAC",      "td3": "TD3",      "ppo": "PPO"}
    buckets   = {"sac": sac_buckets,"td3": td3_buckets,"ppo": ppo_buckets}
    boxes = {}
    for key in ("sac", "td3", "ppo"):
        b = ax.boxplot(
            buckets[key], positions=positions[key], widths=0.65,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=colors[key], alpha=0.72),
            medianprops=dict(color="black", lw=1.5),
        )
        boxes[key] = b

    ax.set_xticks([2, 5, 8])
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_title("5. Stage-by-Stage Return Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Episode Return")
    ax.grid(True, alpha=0.3)
    ax.legend(
        [boxes["sac"]["boxes"][0], boxes["td3"]["boxes"][0], boxes["ppo"]["boxes"][0]],
        ["SAC", "TD3", "PPO"], loc="lower right", framealpha=0.9,
    )

    # ── Panel 6: Deterministic evaluation curves ─────────────────────────────
    ax = axes[2, 1]
    for tb, scale, col, label in [
        (sac_tb, 1.0,       C_SAC, "SAC Eval Return"),
        (td3_tb, td3_scale, C_TD3, "TD3 Eval Return (scaled)"),
    ]:
        d = tb.get("eval/mean_reward")
        if d is not None:
            ax.plot(d["steps"] * scale, d["values"], color=col, lw=2.5, marker="o", ms=4, label=label)
    ax.axvline(STAGE1, color="gray",   ls="--", lw=1.2, alpha=0.7)
    ax.axvline(STAGE2, color="purple", ls="--", lw=1.2, alpha=0.7)
    ax.set_title("6. Deterministic Evaluation Return", fontsize=12, fontweight="bold")
    ax.set_xlabel("Aligned Timesteps")
    ax.set_ylabel("Mean Eval Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "ppo_vs_sac_vs_td3_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2: Clean focused return curves
    # ─────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(13, 6))
    if len(sac_steps):
        plt.scatter(sac_steps, sac_vals, color=C_SAC, alpha=0.07, s=5)
        plt.plot(sac_ss, sac_sv, color=C_SAC, lw=3, label="SAC — 3.0 M steps (off-policy)")
    if len(td3_steps):
        plt.scatter(td3_steps, td3_vals, color=C_TD3, alpha=0.07, s=5)
        plt.plot(td3_ss, td3_sv, color=C_TD3, lw=3, label="TD3 — scaled from 10.5 M (off-policy)")
    if len(ppo_steps):
        plt.scatter(ppo_steps, ppo_vals, color=C_PPO, alpha=0.07, s=5)
        plt.plot(ppo_ss, ppo_sv, color=C_PPO, lw=3, label="PPO — scaled from 13.5 M (on-policy)")
    plt.axvline(STAGE1, color="gray",   ls="--", lw=1.5, alpha=0.8, label="Stage 1 – Rough Bumps")
    plt.axvline(STAGE2, color="purple", ls="--", lw=1.5, alpha=0.8, label="Stage 2 – Hurdles")
    plt.title("Unitree Go1: PPO vs SAC vs TD3 Curriculum Learning Curves", fontsize=14, fontweight="bold")
    plt.xlabel("Curriculum Aligned Steps (3 M reference)", fontsize=12)
    plt.ylabel("Episode Return", fontsize=12)
    plt.ylim(-1000, 3200)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "ppo_vs_sac_vs_td3_returns.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # ─────────────────────────────────────────────────────────────────────────
    # Emit key metrics for TD3 to stdout (for README)
    # ─────────────────────────────────────────────────────────────────────────
    def steps_to_thr(steps, vals, thr):
        for s, v in zip(steps, vals):
            if v >= thr:
                return int(s)
        return None

    td3_eval = td3_tb.get("eval/mean_reward")
    peak_td3 = float(np.max(td3_eval["values"])) if td3_eval is not None else None

    print("\n=== TD3 Summary Metrics ===")
    print(f"  Total steps logged (scaled) : {int(td3_steps[-1]) if len(td3_steps) else 'N/A'}")
    print(f"  Peak eval reward            : {peak_td3}")
    print(f"  Max episode return          : {float(np.max(td3_vals)) if len(td3_vals) else 'N/A':.2f}")
    print(f"  Mean episode return         : {float(np.mean(td3_vals)) if len(td3_vals) else 'N/A':.2f}")

    td3_steps_raw = td3_tb["train/episode_return"]["steps"] if "train/episode_return" in td3_tb else np.array([])
    buckets_td3_raw = bucket_by_stage(td3_steps_raw, td3_vals if len(td3_vals) else np.array([]), (3_500_000, 7_000_000))
    for i, name in enumerate(["Stage 0 (Flat)", "Stage 1 (Rough)", "Stage 2 (Hurdle)"]):
        b = buckets_td3_raw[i]
        print(f"  {name} mean return : {np.mean(b):.2f}" if b else f"  {name} mean return : N/A")

    t2000 = steps_to_thr(td3_ss, td3_sv, 2000)
    t2500 = steps_to_thr(td3_ss, td3_sv, 2500)
    print(f"  Steps to return ≥ 2000 (aligned) : {t2000}")
    print(f"  Steps to return ≥ 2500 (aligned) : {t2500}")

    print("\nDone.")


if __name__ == "__main__":
    main()
