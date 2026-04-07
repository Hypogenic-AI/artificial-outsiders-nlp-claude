"""
Step 3: Core statistical analysis for the Artificial Outsiders hypothesis.
Tests whether LLM judge consensus inversely correlates with human interestingness.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ─── Load data ───────────────────────────────────────────────────────────────
with open("results/hanna_stories_prepared.json") as f:
    stories = pd.DataFrame(json.load(f))

with open("results/llm_ratings.json") as f:
    ratings = pd.DataFrame(json.load(f))

print(f"Stories: {len(stories)}, LLM ratings: {len(ratings)}")

# ─── Pivot LLM ratings to wide format ────────────────────────────────────────
llm_wide = ratings.pivot(index="story_id", columns="model", values="score")
llm_wide.columns = [f"llm_{c.replace('-', '_').replace('.', '_')}" for c in llm_wide.columns]

# Compute LLM consensus and disagreement
llm_cols = llm_wide.columns.tolist()
llm_wide["llm_mean"] = llm_wide[llm_cols].mean(axis=1)
llm_wide["llm_std"] = llm_wide[llm_cols].std(axis=1)
llm_wide["llm_min"] = llm_wide[llm_cols].min(axis=1)
llm_wide["llm_max"] = llm_wide[llm_cols].max(axis=1)
llm_wide["llm_range"] = llm_wide["llm_max"] - llm_wide["llm_min"]

# Merge with human data
merged = stories.merge(llm_wide, left_on="Story_ID", right_index=True, how="inner")
print(f"Merged: {len(merged)} stories with both human and LLM ratings")

# ─── Descriptive Statistics ──────────────────────────────────────────────────
print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

print("\nHuman ratings:")
for col in ["Surprise", "Engagement", "human_interestingness", "human_quality"]:
    print(f"  {col}: mean={merged[col].mean():.3f}, std={merged[col].std():.3f}")

print("\nLLM ratings:")
print(f"  LLM mean score: mean={merged['llm_mean'].mean():.3f}, std={merged['llm_mean'].std():.3f}")
print(f"  LLM std (disagreement): mean={merged['llm_std'].mean():.3f}, std={merged['llm_std'].std():.3f}")

for col in llm_cols:
    print(f"  {col}: mean={merged[col].mean():.3f}, std={merged[col].std():.3f}")

# ─── Core Hypothesis Tests ───────────────────────────────────────────────────
print("\n" + "="*70)
print("HYPOTHESIS TESTS")
print("="*70)

alpha = 0.05
bonferroni_alpha = alpha / 4  # 4 primary tests

results = {}

# H1: LLM consensus (mean) negatively correlates with human Surprise
rho, p = stats.spearmanr(merged["llm_mean"], merged["Surprise"])
results["H1_consensus_vs_surprise"] = {"rho": rho, "p": p}
sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < bonferroni_alpha else "ns"
print(f"\nH1: LLM consensus vs Human Surprise")
print(f"  Spearman ρ = {rho:.4f}, p = {p:.2e} [{sig}]")
print(f"  Direction: {'SUPPORTS' if rho < 0 else 'REFUTES'} hypothesis (expected negative)")

# H2: LLM disagreement (std) positively correlates with human Engagement
rho2, p2 = stats.spearmanr(merged["llm_std"], merged["Engagement"])
results["H2_disagreement_vs_engagement"] = {"rho": rho2, "p": p2}
sig = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < bonferroni_alpha else "ns"
print(f"\nH2: LLM disagreement vs Human Engagement")
print(f"  Spearman ρ = {rho2:.4f}, p = {p2:.2e} [{sig}]")
print(f"  Direction: {'SUPPORTS' if rho2 > 0 else 'REFUTES'} hypothesis (expected positive)")

# H3: Bottom quartile LLM consensus has higher human interestingness than top quartile
q25 = merged["llm_mean"].quantile(0.25)
q75 = merged["llm_mean"].quantile(0.75)
low_consensus = merged[merged["llm_mean"] <= q25]["human_interestingness"]
high_consensus = merged[merged["llm_mean"] >= q75]["human_interestingness"]
u_stat, p3 = stats.mannwhitneyu(low_consensus, high_consensus, alternative="greater")
cohens_d = (low_consensus.mean() - high_consensus.mean()) / np.sqrt(
    (low_consensus.std()**2 + high_consensus.std()**2) / 2)
results["H3_quartile_comparison"] = {
    "low_mean": low_consensus.mean(), "high_mean": high_consensus.mean(),
    "u_stat": u_stat, "p": p3, "cohens_d": cohens_d,
    "n_low": len(low_consensus), "n_high": len(high_consensus)
}
sig = "***" if p3 < 0.001 else "**" if p3 < 0.01 else "*" if p3 < bonferroni_alpha else "ns"
print(f"\nH3: LLM consensus quartile comparison")
print(f"  Bottom 25% LLM consensus: human interest = {low_consensus.mean():.3f} (n={len(low_consensus)})")
print(f"  Top 25% LLM consensus: human interest = {high_consensus.mean():.3f} (n={len(high_consensus)})")
print(f"  Mann-Whitney U = {u_stat:.0f}, p = {p3:.2e} [{sig}]")
print(f"  Cohen's d = {cohens_d:.3f}")
print(f"  Direction: {'SUPPORTS' if low_consensus.mean() > high_consensus.mean() else 'REFUTES'} hypothesis")

# H4: Effect holds specifically for LLM-generated stories
llm_stories = merged[merged["is_llm"] == True]
rho4, p4 = stats.spearmanr(llm_stories["llm_mean"], llm_stories["human_interestingness"])
results["H4_llm_only"] = {"rho": rho4, "p": p4, "n": len(llm_stories)}
sig = "***" if p4 < 0.001 else "**" if p4 < 0.01 else "*" if p4 < bonferroni_alpha else "ns"
print(f"\nH4: LLM consensus vs Human interestingness (LLM-generated stories only)")
print(f"  n = {len(llm_stories)}")
print(f"  Spearman ρ = {rho4:.4f}, p = {p4:.2e} [{sig}]")
print(f"  Direction: {'SUPPORTS' if rho4 < 0 else 'REFUTES'} hypothesis (expected negative)")

# ─── Additional Analysis ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("ADDITIONAL ANALYSES")
print("="*70)

# Correlation with human interestingness composite
rho_int, p_int = stats.spearmanr(merged["llm_mean"], merged["human_interestingness"])
print(f"\nLLM consensus vs Human interestingness (all stories):")
print(f"  Spearman ρ = {rho_int:.4f}, p = {p_int:.2e}")

# LLM disagreement vs human interestingness
rho_dis, p_dis = stats.spearmanr(merged["llm_std"], merged["human_interestingness"])
print(f"\nLLM disagreement vs Human interestingness (all stories):")
print(f"  Spearman ρ = {rho_dis:.4f}, p = {p_dis:.2e}")

# Per-model correlation with human interest
print("\nPer-model correlation with human interestingness:")
for col in llm_cols:
    r, p = stats.spearmanr(merged[col], merged["human_interestingness"])
    print(f"  {col}: ρ = {r:.4f}, p = {p:.2e}")

# LLM consensus vs each human dimension
print("\nLLM consensus vs each human dimension:")
for dim in ["Relevance", "Coherence", "Empathy", "Surprise", "Engagement", "Complexity"]:
    r, p = stats.spearmanr(merged["llm_mean"], merged[dim])
    print(f"  {dim}: ρ = {r:.4f}, p = {p:.2e}")

# ─── Control: Is this just about quality? ────────────────────────────────────
print("\n" + "="*70)
print("CONTROL: QUALITY vs INTERESTINGNESS")
print("="*70)

# Partial correlation: LLM consensus vs human interest, controlling for quality
from scipy.stats import spearmanr

# Residualize human interestingness from quality
quality_resid = merged["human_interestingness"] - merged["human_quality"]
rho_resid, p_resid = spearmanr(merged["llm_mean"], quality_resid)
print(f"\nLLM consensus vs (Interest - Quality) residual:")
print(f"  Spearman ρ = {rho_resid:.4f}, p = {p_resid:.2e}")

# ─── Analysis by source model ────────────────────────────────────────────────
print("\n" + "="*70)
print("ANALYSIS BY SOURCE MODEL")
print("="*70)

for model in sorted(merged["Model"].unique()):
    subset = merged[merged["Model"] == model]
    r, p = stats.spearmanr(subset["llm_mean"], subset["human_interestingness"])
    print(f"  {model:20s}: ρ = {r:+.4f}, p = {p:.3e}, n = {len(subset)}")

# ─── Bin analysis ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BIN ANALYSIS: LLM CONSENSUS BINS vs HUMAN RATINGS")
print("="*70)

merged["llm_bin"] = pd.qcut(merged["llm_mean"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
bin_stats = merged.groupby("llm_bin").agg(
    n=("human_interestingness", "count"),
    human_interest_mean=("human_interestingness", "mean"),
    human_interest_std=("human_interestingness", "std"),
    human_surprise_mean=("Surprise", "mean"),
    human_engagement_mean=("Engagement", "mean"),
    human_quality_mean=("human_quality", "mean"),
    llm_disagree_mean=("llm_std", "mean"),
).round(3)
print(bin_stats.to_string())

# ─── Bootstrap confidence intervals ─────────────────────────────────────────
print("\n" + "="*70)
print("BOOTSTRAP CONFIDENCE INTERVALS (n=10000)")
print("="*70)

n_boot = 10000
boot_rhos = []
for _ in range(n_boot):
    idx = np.random.choice(len(merged), size=len(merged), replace=True)
    r, _ = stats.spearmanr(merged.iloc[idx]["llm_mean"], merged.iloc[idx]["human_interestingness"])
    boot_rhos.append(r)

ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
print(f"LLM consensus vs Human interestingness:")
print(f"  ρ = {rho_int:.4f}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")

# ─── Inter-LLM agreement ────────────────────────────────────────────────────
print("\n" + "="*70)
print("INTER-LLM AGREEMENT")
print("="*70)

for c1, c2 in combinations(llm_cols, 2):
    r, p = stats.spearmanr(merged[c1], merged[c2])
    print(f"  {c1} vs {c2}: ρ = {r:.4f}")

# ─── Save all results ────────────────────────────────────────────────────────
# Convert to serializable
for k, v in results.items():
    for kk, vv in v.items():
        if isinstance(vv, (np.floating, np.integer)):
            results[k][kk] = float(vv)

with open("results/analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

merged.to_csv("results/merged_dataset.csv", index=False)
print(f"\nSaved analysis results and merged dataset.")
