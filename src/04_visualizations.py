"""
Step 4: Create visualizations for the Artificial Outsiders analysis.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

SEED = 42
np.random.seed(SEED)

# Load merged data
merged = pd.read_csv("results/merged_dataset.csv")
print(f"Loaded {len(merged)} stories")

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
palette = sns.color_palette("Set2")

# ─── Figure 1: LLM Consensus vs Human Interestingness scatter ────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1a: Scatter with regression
ax = axes[0]
sns.regplot(data=merged, x="llm_mean", y="human_interestingness",
            scatter_kws={"alpha": 0.3, "s": 15}, line_kws={"color": "red"},
            ax=ax)
ax.set_xlabel("LLM Consensus Score (mean of 3 judges)")
ax.set_ylabel("Human Interestingness\n(mean of Surprise + Engagement)")
ax.set_title("A) LLM Consensus vs Human Interestingness\n(Spearman ρ = 0.48)")

# 1b: By source model type (human vs LLM)
ax = axes[1]
for is_llm, label, color in [(False, "Human-written", palette[0]), (True, "LLM-generated", palette[1])]:
    subset = merged[merged["is_llm"] == is_llm]
    ax.scatter(subset["llm_mean"], subset["human_interestingness"],
               alpha=0.4, s=15, label=label, color=color)
ax.set_xlabel("LLM Consensus Score")
ax.set_ylabel("Human Interestingness")
ax.set_title("B) By Source Type")
ax.legend()

# 1c: Bin analysis
ax = axes[2]
bins = merged.groupby("llm_bin").agg(
    mean=("human_interestingness", "mean"),
    sem=("human_interestingness", "sem"),
).reset_index()
bars = ax.bar(range(4), bins["mean"], yerr=bins["sem"]*1.96,
              capsize=5, color=[palette[3], palette[4], palette[5], palette[2]])
ax.set_xticks(range(4))
ax.set_xticklabels(bins["llm_bin"])
ax.set_xlabel("LLM Consensus Quartile")
ax.set_ylabel("Human Interestingness (mean ± 95% CI)")
ax.set_title("C) Human Interest by LLM Consensus Quartile")

plt.tight_layout()
plt.savefig("figures/fig1_consensus_vs_interest.png", dpi=150, bbox_inches="tight")
print("Saved figures/fig1_consensus_vs_interest.png")

# ─── Figure 2: LLM Disagreement analysis ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 2a: LLM disagreement vs human interest
ax = axes[0]
sns.regplot(data=merged, x="llm_std", y="human_interestingness",
            scatter_kws={"alpha": 0.3, "s": 15}, line_kws={"color": "red"}, ax=ax)
ax.set_xlabel("LLM Disagreement (std across 3 judges)")
ax.set_ylabel("Human Interestingness")
ax.set_title("A) LLM Disagreement vs Human Interest\n(Spearman ρ = 0.15)")

# 2b: Heatmap of correlations
ax = axes[1]
corr_cols = ["llm_gpt_4o", "llm_gpt_4o_mini", "llm_gpt_4_1_mini",
             "Surprise", "Engagement", "human_interestingness"]
corr_labels = ["GPT-4o", "GPT-4o-mini", "GPT-4.1-mini",
               "Human\nSurprise", "Human\nEngagement", "Human\nInterest"]
corr_matrix = merged[corr_cols].corr(method="spearman")
corr_matrix.columns = corr_labels
corr_matrix.index = corr_labels
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-0.5, vmax=1.0, ax=ax, square=True)
ax.set_title("B) Spearman Correlations")

# 2c: Distribution of LLM scores by human interest quartile
ax = axes[2]
merged["human_interest_q"] = pd.qcut(merged["human_interestingness"], q=4,
                                       labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
sns.boxplot(data=merged, x="human_interest_q", y="llm_mean", ax=ax, palette="Set2")
ax.set_xlabel("Human Interestingness Quartile")
ax.set_ylabel("LLM Consensus Score")
ax.set_title("C) LLM Scores by Human Interest Level")

plt.tight_layout()
plt.savefig("figures/fig2_disagreement_analysis.png", dpi=150, bbox_inches="tight")
print("Saved figures/fig2_disagreement_analysis.png")

# ─── Figure 3: Per-model analysis ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3a: LLM consensus correlation by source model
ax = axes[0]
from scipy import stats
model_corrs = []
for model in sorted(merged["Model"].unique()):
    subset = merged[merged["Model"] == model]
    r, p = stats.spearmanr(subset["llm_mean"], subset["human_interestingness"])
    model_corrs.append({"model": model, "rho": r, "p": p, "n": len(subset)})

mdf = pd.DataFrame(model_corrs).sort_values("rho")
colors = ["#e74c3c" if r < 0 else "#2ecc71" for r in mdf["rho"]]
ax.barh(range(len(mdf)), mdf["rho"], color=colors, alpha=0.8)
ax.set_yticks(range(len(mdf)))
ax.set_yticklabels(mdf["model"])
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("Spearman ρ (LLM consensus vs Human interest)")
ax.set_title("A) Correlation by Source Model")

# 3b: LLM vs Human score distributions
ax = axes[1]
ax.hist(merged["llm_mean"], bins=20, alpha=0.6, label="LLM Consensus", color=palette[0], density=True)
ax.hist(merged["human_interestingness"], bins=20, alpha=0.6, label="Human Interest", color=palette[1], density=True)
ax.set_xlabel("Score")
ax.set_ylabel("Density")
ax.set_title("B) Score Distributions")
ax.legend()

plt.tight_layout()
plt.savefig("figures/fig3_model_analysis.png", dpi=150, bbox_inches="tight")
print("Saved figures/fig3_model_analysis.png")

# ─── Figure 4: The "quality confound" ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 4a: LLM consensus vs human quality
ax = axes[0]
sns.regplot(data=merged, x="llm_mean", y="human_quality",
            scatter_kws={"alpha": 0.3, "s": 15}, line_kws={"color": "red"}, ax=ax)
r, p = stats.spearmanr(merged["llm_mean"], merged["human_quality"])
ax.set_xlabel("LLM Consensus Score")
ax.set_ylabel("Human Overall Quality")
ax.set_title(f"A) LLM Consensus vs Human Quality\n(ρ = {r:.3f})")

# 4b: After controlling for quality
ax = axes[1]
quality_resid = merged["human_interestingness"] - merged["human_quality"]
sns.regplot(x=merged["llm_mean"], y=quality_resid,
            scatter_kws={"alpha": 0.3, "s": 15}, line_kws={"color": "red"}, ax=ax)
r, p = stats.spearmanr(merged["llm_mean"], quality_resid)
ax.set_xlabel("LLM Consensus Score")
ax.set_ylabel("Interest - Quality (residual)")
ax.set_title(f"B) After Controlling for Quality\n(ρ = {r:.3f}, p = {p:.3f})")

plt.tight_layout()
plt.savefig("figures/fig4_quality_confound.png", dpi=150, bbox_inches="tight")
print("Saved figures/fig4_quality_confound.png")

print("\nAll figures saved successfully.")
