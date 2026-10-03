"""Side-by-Side Comparison: PPO vs SAC vs TD3 for Unitree Go1 Locomotion Curriculum.

Loads TensorBoard logs for all three algorithms, applies curriculum-aligned timestep
scaling to PPO and TD3, and generates polished comparison PNGs.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing import event_accumulator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUTPUT_DIR = os.path.join(REPO_ROOT, "src", "visualizations")
SAC_LOGDIR = os.path.join(REPO_ROOT, "src", "sac_baseline", "logs", "sac_go1_curriculum")
TD3_LOGDIR = os.path.join(REPO_ROOT, "src", "td3_baseline", "logs", "td3_go1_curriculum")
PPO_LOGDIR = os.path.join(REPO_ROOT, "src", "visualizations", "tensorboard_logs", "ppo_raw")

PPO_MAX_STEPS = 13_500_000.0
SAC_MAX_STEPS =  3_000_000.0   # reference scale for alignment
TD3_MAX_STEPS = 10_500_000.0

# ── colours & style ───────────────────────────────────────────────────────────
C_SAC  = "#1f77b4"   # Matplotlib blue
C_TD3  = "#2ca02c"   # Matplotlib green
C_PPO  = "#d62728"   # Matplotlib red
C_S1   = "#6b6b6b"   # Stage 1 divider — dark grey
C_S2   = "#7b2d8b"   # Stage 2 divider — purple

STAGE1 = 1_000_000
STAGE2 = 2_000_000

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.framealpha": 0.92,
    "legend.fontsize": 8.5,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def moving_average(data, w=60):
    if len(data) < w:
        return np.array(data)
    return np.convolve(data, np.ones(w) / w, mode="valid")


def load_tb(logdir):
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps":  np.array([e.step  for e in events]),
            "values": np.array([e.value for e in events]),
        }
    return data


def extract_return(tb_data, scale=1.0):
    """Scaled steps + horizon-normalised return values."""
    ret = tb_data.get("train/episode_return")
    length = tb_data.get("train/episode_length")
    if ret is None:
        return np.array([]), np.array([])
    steps = ret["steps"] * scale
    vals = ret["values"].copy()
    if length is not None:
        lens = length["values"]
        vals = np.where(lens >= 1000.0, vals * (1000.0 / np.maximum(lens, 1.0)), vals)
    return steps, vals


def smooth_pair(steps, vals, w=60):
    sv = moving_average(vals, w)
    ss = steps[w - 1:] if len(steps) >= w else steps
    return ss, sv


def fmt_m(x, _pos=None):
    """Format axis ticks as '0', '0.5M', '1M', etc."""
    if x == 0:
        return "0"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M".rstrip("0").rstrip(".")
    if x >= 1_000:
        return f"{x / 1_000:.0f}k"
    return str(int(x))


def add_stage_lines(ax, x1=STAGE1, x2=STAGE2, label1="Stage 1\n(Rough)", label2="Stage 2\n(Hurdle)"):
    ax.axvline(x1, color=C_S1, ls="--", lw=1.3, alpha=0.75)
    ax.axvline(x2, color=C_S2, ls="--", lw=1.3, alpha=0.75)
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    ax.text(x1 + 0.01 * (x2 - x1), ymax - 0.04 * span, label1,
            color=C_S1, fontsize=7.5, va="top", ha="left")
    ax.text(x2 + 0.01 * (x2 - x1), ymax - 0.04 * span, label2,
            color=C_S2, fontsize=7.5, va="top", ha="left")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading TensorBoard logs …")
    sac_tb = load_tb(SAC_LOGDIR)
    td3_tb = load_tb(TD3_LOGDIR)
    ppo_tb = load_tb(PPO_LOGDIR)
    print(f"  SAC tags: {list(sac_tb.keys())}")
    print(f"  TD3 tags: {list(td3_tb.keys())}")
    print(f"  PPO tags: {list(ppo_tb.keys())}")

    ppo_scale = SAC_MAX_STEPS / PPO_MAX_STEPS   # ≈ 0.222
    td3_scale = SAC_MAX_STEPS / TD3_MAX_STEPS   # ≈ 0.286

    sac_steps, sac_vals = extract_return(sac_tb, 1.0)
    td3_steps, td3_vals = extract_return(td3_tb, td3_scale)
    ppo_steps, ppo_vals = extract_return(ppo_tb, ppo_scale)

    W = 60
    sac_ss, sac_sv = smooth_pair(sac_steps, sac_vals, W)
    td3_ss, td3_sv = smooth_pair(td3_steps, td3_vals, W)
    ppo_ss, ppo_sv = smooth_pair(ppo_steps, ppo_vals, W)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1: 6-panel dashboard
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle(
        "Unitree Go1 Quadruped  ·  PPO vs SAC vs TD3  —  Three-Algorithm Curriculum Benchmark",
        fontsize=14, fontweight="bold", y=1.002,
    )

    # shared legend handles
    legend_elements = [
        Line2D([0], [0], color=C_SAC, lw=2.5, label="SAC  (3.0 M steps, off-policy)"),
        Line2D([0], [0], color=C_TD3, lw=2.5, label="TD3  (10.5 M steps, off-policy, aligned)"),
        Line2D([0], [0], color=C_PPO, lw=2.5, label="PPO  (13.5 M steps, on-policy, aligned)"),
        Line2D([0], [0], color=C_S1,  lw=1.5, ls="--", label="Stage 1 – Rough Bumps"),
        Line2D([0], [0], color=C_S2,  lw=1.5, ls="--", label="Stage 2 – Hurdles"),
    ]

    # ── Panel 1: Returns (aligned) ────────────────────────────────────────────
    ax = axes[0, 0]
    if len(sac_steps):
        ax.scatter(sac_steps, sac_vals, color=C_SAC, alpha=0.06, s=4, rasterized=True)
        ax.plot(sac_ss, sac_sv, color=C_SAC, lw=2.2)
    if len(td3_steps):
        ax.scatter(td3_steps, td3_vals, color=C_TD3, alpha=0.06, s=4, rasterized=True)
        ax.plot(td3_ss, td3_sv, color=C_TD3, lw=2.2)
    if len(ppo_steps):
        ax.scatter(ppo_steps, ppo_vals, color=C_PPO, alpha=0.06, s=4, rasterized=True)
        ax.plot(ppo_ss, ppo_sv, color=C_PPO, lw=2.2)
    ax.set_xlim(0, SAC_MAX_STEPS)
    ax.set_ylim(-1200, 3400)
    add_stage_lines(ax)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_m))
    ax.set_title("1 · Training Episode Returns  (Aligned Curriculum Steps)")
    ax.set_xlabel("Aligned Timesteps  (3 M reference)")
    ax.set_ylabel("Episode Return")
    ax.legend(handles=legend_elements, loc="upper left", ncol=1)

    # ── Panel 2: Normalised progress (0–100 %) ───────────────────────────────
    ax = axes[0, 1]
    if len(sac_ss):
        ax.plot(sac_ss / SAC_MAX_STEPS * 100, sac_sv, color=C_SAC, lw=2.2, label="SAC")
    if len(td3_ss):
        raw_td3 = td3_ss * TD3_MAX_STEPS / SAC_MAX_STEPS
        ax.plot(raw_td3 / TD3_MAX_STEPS * 100, td3_sv, color=C_TD3, lw=2.2, label="TD3")
    if len(ppo_ss):
        raw_ppo = ppo_ss * PPO_MAX_STEPS / SAC_MAX_STEPS
        ax.plot(raw_ppo / PPO_MAX_STEPS * 100, ppo_sv, color=C_PPO, lw=2.2, label="PPO")
    ax.axvline(33.33, color=C_S1, ls="--", lw=1.3, alpha=0.75)
    ax.axvline(66.66, color=C_S2, ls="--", lw=1.3, alpha=0.75)
    ax.set_xlim(0, 100)
    ax.set_ylim(-400, 3200)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_title("2 · Learning Curve vs Normalised Curriculum Progress")
    ax.set_xlabel("Curriculum Progress  (%)")
    ax.set_ylabel("Episode Return  (MA-60)")
    ax.legend(loc="upper left")

    # ── Panel 3: Survival / episode length ──────────────────────────────────
    ax = axes[1, 0]
    for tb, scale, col, label in [
        (sac_tb, 1.0,       C_SAC, "SAC"),
        (td3_tb, td3_scale, C_TD3, "TD3 (aligned)"),
        (ppo_tb, ppo_scale, C_PPO, "PPO (aligned)"),
    ]:
        d = tb.get("train/episode_length")
        if d is not None:
            sv = moving_average(d["values"], W)
            ss = d["steps"][W - 1:] * scale if len(d["steps"]) >= W else d["steps"] * scale
            ax.plot(ss, sv, color=col, lw=2.2, label=label)
    ax.axhline(1000, color="#27ae60", ls=":", lw=1.6, alpha=0.8, label="Max limit (1000)")
    ax.set_xlim(0, SAC_MAX_STEPS)
    ax.set_ylim(0, 1100)
    add_stage_lines(ax)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_m))
    ax.set_title("3 · Survival Capability  /  Episode Duration")
    ax.set_xlabel("Aligned Timesteps")
    ax.set_ylabel("Steps Upright")
    ax.legend(loc="lower right")

    # ── Panel 4: Actual sample footprint ─────────────────────────────────────
    ax = axes[1, 1]
    if len(sac_ss):
        ax.plot(sac_ss / 1e6, sac_sv, color=C_SAC, lw=2.2, label="SAC  (3.0 M)")
    if len(td3_ss):
        ax.plot(td3_ss * TD3_MAX_STEPS / SAC_MAX_STEPS / 1e6, td3_sv,
                color=C_TD3, lw=2.2, label="TD3  (10.5 M)")
    if len(ppo_ss):
        ax.plot(ppo_ss * PPO_MAX_STEPS / SAC_MAX_STEPS / 1e6, ppo_sv,
                color=C_PPO, lw=2.2, label="PPO  (13.5 M)")
    ax.set_ylim(-400, 3200)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} M" if x > 0 else "0"))
    ax.set_title("4 · Sample Footprint:  Return vs Actual Env Interactions")
    ax.set_xlabel("Actual Environment Steps  (Millions)")
    ax.set_ylabel("Episode Return  (MA-60)")
    ax.legend(loc="lower right")

    # ── Panel 5: Stage-by-stage box plots ────────────────────────────────────
    ax = axes[2, 0]

    def bucket_by_stage(steps_raw, vals, boundaries):
        s0e, s1e = boundaries
        b = [[], [], []]
        for s, v in zip(steps_raw, vals):
            if s < s0e:
                b[0].append(v)
            elif s < s1e:
                b[1].append(v)
            else:
                b[2].append(v)
        return b

    sac_bkts = bucket_by_stage(sac_steps, sac_vals, (1_000_000, 2_000_000))
    td3_bkts = bucket_by_stage(
        td3_steps * TD3_MAX_STEPS / SAC_MAX_STEPS, td3_vals, (3_500_000, 7_000_000))

    ppo_ret = ppo_tb.get("train/episode_return")
    ppo_len = ppo_tb.get("train/episode_length")
    if ppo_ret is not None and ppo_len is not None:
        ppo_rv = np.where(ppo_len["values"] >= 1000,
                          ppo_ret["values"] * (1000.0 / np.maximum(ppo_len["values"], 1)),
                          ppo_ret["values"])
    else:
        ppo_rv = np.array([])
    ppo_rs = ppo_ret["steps"] if ppo_ret is not None else np.array([])
    ppo_bkts = bucket_by_stage(ppo_rs, ppo_rv, (4_500_000, 9_000_000))

    bp_kw = dict(widths=0.55, patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", lw=1.8),
                 whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))

    for key, bkts, pos, col in [
        ("sac", sac_bkts, [1, 4, 7], C_SAC),
        ("td3", td3_bkts, [2, 5, 8], C_TD3),
        ("ppo", ppo_bkts, [3, 6, 9], C_PPO),
    ]:
        b = ax.boxplot(bkts, positions=pos, boxprops=dict(facecolor=col, alpha=0.70), **bp_kw)
        if key == "sac":
            h_sac = b["boxes"][0]
        elif key == "td3":
            h_td3 = b["boxes"][0]
        else:
            h_ppo = b["boxes"][0]

    ax.set_xticks([2, 5, 8])
    ax.set_xticklabels(["Stage 0\n(Flat)", "Stage 1\n(Rough)", "Stage 2\n(Hurdle)"], fontsize=10)
    ax.set_title("5 · Stage-by-Stage Return Distribution  (Median & IQR)")
    ax.set_ylabel("Episode Return")
    ax.legend([h_sac, h_td3, h_ppo], ["SAC", "TD3", "PPO"], loc="upper right")

    # ── Panel 6: Deterministic eval curves ───────────────────────────────────
    ax = axes[2, 1]
    for tb, scale, col, label in [
        (sac_tb, 1.0,       C_SAC, "SAC  Eval Return"),
        (td3_tb, td3_scale, C_TD3, "TD3  Eval Return (aligned)"),
    ]:
        d = tb.get("eval/mean_reward")
        if d is not None:
            ax.plot(d["steps"] * scale, d["values"],
                    color=col, lw=2.2, marker="o", ms=3.5, markevery=3, label=label)
    ax.set_xlim(0, SAC_MAX_STEPS)
    ax.set_ylim(0, 3200)
    add_stage_lines(ax)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_m))
    ax.set_title("6 · Deterministic Evaluation Return  (greedy policy)")
    ax.set_xlabel("Aligned Timesteps")
    ax.set_ylabel("Mean Eval Return")
    ax.legend(loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 1])
    out1 = os.path.join(OUTPUT_DIR, "ppo_vs_sac_vs_td3_comparison.png")
    fig.savefig(out1)
    plt.close(fig)
    print(f"Saved: {out1}")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2: Clean focused return curves
    # ─────────────────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(13, 6))

    if len(sac_steps):
        ax2.scatter(sac_steps, sac_vals, color=C_SAC, alpha=0.05, s=4, rasterized=True)
        ax2.plot(sac_ss, sac_sv, color=C_SAC, lw=3,
                 label="SAC  —  3.0 M steps  (off-policy)")
    if len(td3_steps):
        ax2.scatter(td3_steps, td3_vals, color=C_TD3, alpha=0.05, s=4, rasterized=True)
        ax2.plot(td3_ss, td3_sv, color=C_TD3, lw=3,
                 label="TD3  —  10.5 M steps  (off-policy, aligned)")
    if len(ppo_steps):
        ax2.scatter(ppo_steps, ppo_vals, color=C_PPO, alpha=0.05, s=4, rasterized=True)
        ax2.plot(ppo_ss, ppo_sv, color=C_PPO, lw=3,
                 label="PPO  —  13.5 M steps  (on-policy, aligned)")

    ax2.axvline(STAGE1, color=C_S1, ls="--", lw=1.5, alpha=0.8)
    ax2.axvline(STAGE2, color=C_S2, ls="--", lw=1.5, alpha=0.8)
    # annotate stage dividers
    for x, col, label in [(STAGE1, C_S1, "Stage 1 – Rough Bumps"),
                           (STAGE2, C_S2, "Stage 2 – Hurdles")]:
        ax2.text(x + SAC_MAX_STEPS * 0.012, 3050, label,
                 color=col, fontsize=9, va="center")

    # annotate peak values
    for ss, sv, col, fmt in [
        (sac_ss, sac_sv, C_SAC, "SAC peak\n{:.0f}"),
        (td3_ss, td3_sv, C_TD3, "TD3 peak\n{:.0f}"),
    ]:
        if len(sv):
            peak_idx = np.argmax(sv)
            ax2.annotate(
                fmt.format(sv[peak_idx]),
                xy=(ss[peak_idx], sv[peak_idx]),
                xytext=(ss[peak_idx] - SAC_MAX_STEPS * 0.12, sv[peak_idx] - 200),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2),
                color=col, fontsize=8.5, fontweight="bold",
            )

    ax2.set_xlim(0, SAC_MAX_STEPS)
    ax2.set_ylim(-800, 3400)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_m))
    ax2.set_title(
        "Unitree Go1  ·  PPO vs SAC vs TD3  —  Curriculum Learning Curves  (MA-60)",
        fontsize=13, fontweight="bold",
    )
    ax2.set_xlabel("Curriculum Aligned Steps  (3 M reference)", fontsize=11)
    ax2.set_ylabel("Episode Return", fontsize=11)
    ax2.legend(fontsize=10.5, loc="upper left", framealpha=0.92)

    fig2.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "ppo_vs_sac_vs_td3_returns.png")
    fig2.savefig(out2)
    plt.close(fig2)
    print(f"Saved: {out2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
