"""
Step 6: Quality-controlled analysis.
The HANNA dataset has old models with wildly varying quality.
Test the hypothesis on subsets where quality is controlled:
1. Only high-quality stories (top 50% human quality)
2. Within each source model (controls for model quality)
3. After partialing out quality via regression
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

SEED = 42
np.random.seed(SEED)

merged = pd.read_csv("results/merged_dataset.csv")
print(f"Total stories: {len(merged)}")

# ─── Analysis 1: High-quality stories only ────────────────────────────────────
print("\n" + "="*70)
print("ANALYSIS 1: HIGH-QUALITY STORIES ONLY (top 50% by human quality)")
print("="*70)

quality_median = merged["human_quality"].median()
high_q = merged[merged["human_quality"] >= quality_median]
print(f"High-quality stories: {len(high_q)} (quality >= {quality_median:.2f})")

rho, p = stats.spearmanr(high_q["llm_mean"], high_q["human_interestingness"])
print(f"LLM consensus vs Human interest: ρ = {rho:.4f}, p = {p:.2e}")

rho2, p2 = stats.spearmanr(high_q["llm_std"], high_q["human_interestingness"])
print(f"LLM disagreement vs Human interest: ρ = {rho2:.4f}, p = {p2:.2e}")

# Quartile comparison within high-quality
q25 = high_q["llm_mean"].quantile(0.25)
q75 = high_q["llm_mean"].quantile(0.75)
low_c = high_q[high_q["llm_mean"] <= q25]["human_interestingness"]
high_c = high_q[high_q["llm_mean"] >= q75]["human_interestingness"]
u, p3 = stats.mannwhitneyu(low_c, high_c, alternative="greater")
print(f"Low-LLM-consensus ({len(low_c)}): human interest = {low_c.mean():.3f}")
print(f"High-LLM-consensus ({len(high_c)}): human interest = {high_c.mean():.3f}")
print(f"Mann-Whitney p = {p3:.2e} (one-sided: low > high)")

# ─── Analysis 2: Partial correlation ─────────────────────────────────────────
print("\n" + "="*70)
print("ANALYSIS 2: PARTIAL CORRELATION (controlling for overall quality)")
print("="*70)

# Regress out quality from both LLM consensus and human interest
X_quality = merged[["human_quality"]].values

# Residualize LLM consensus
lr1 = LinearRegression().fit(X_quality, merged["llm_mean"])
llm_resid = merged["llm_mean"] - lr1.predict(X_quality)

# Residualize human interestingness
lr2 = LinearRegression().fit(X_quality, merged["human_interestingness"])
interest_resid = merged["human_interestingness"] - lr2.predict(X_quality)

rho_partial, p_partial = stats.spearmanr(llm_resid, interest_resid)
print(f"Partial correlation (controlling for quality): ρ = {rho_partial:.4f}, p = {p_partial:.2e}")
print(f"(Negative value would support the outsider hypothesis)")

# ─── Analysis 3: Within-model analysis ───────────────────────────────────────
print("\n" + "="*70)
print("ANALYSIS 3: WITHIN-MODEL CORRELATIONS")
print("="*70)
print("(Controls for model-level quality differences)")

within_results = []
for model in sorted(merged["Model"].unique()):
    subset = merged[merged["Model"] == model]
    if len(subset) < 20:
        continue
    r, p = stats.spearmanr(subset["llm_mean"], subset["human_interestingness"])
    within_results.append({"model": model, "rho": r, "p": p, "n": len(subset)})
    direction = "supports" if r < 0 else "refutes"
    sig = "*" if p < 0.05 else "ns"
    print(f"  {model:20s}: ρ = {r:+.4f}, p = {p:.3e} [{sig}] - {direction} hypothesis")

# Meta-analysis: average within-model correlation
avg_within = np.mean([r["rho"] for r in within_results])
print(f"\nAverage within-model ρ: {avg_within:.4f}")

# ─── Analysis 4: LLM-only stories, quality controlled ────────────────────────
print("\n" + "="*70)
print("ANALYSIS 4: LLM-ONLY STORIES, PARTIAL CORRELATION")
print("="*70)

llm_only = merged[merged["is_llm"] == True]
X_q = llm_only[["human_quality"]].values
lr_a = LinearRegression().fit(X_q, llm_only["llm_mean"])
lr_b = LinearRegression().fit(X_q, llm_only["human_interestingness"])
resid_a = llm_only["llm_mean"] - lr_a.predict(X_q)
resid_b = llm_only["human_interestingness"] - lr_b.predict(X_q)
rho_llm, p_llm = stats.spearmanr(resid_a, resid_b)
print(f"LLM-only partial correlation: ρ = {rho_llm:.4f}, p = {p_llm:.2e}")

# ─── Analysis 5: Focus on Surprise dimension ─────────────────────────────────
print("\n" + "="*70)
print("ANALYSIS 5: LLM CONSENSUS vs SURPRISE (quality-controlled)")
print("="*70)

lr_s = LinearRegression().fit(X_quality, merged["Surprise"])
surprise_resid = merged["Surprise"] - lr_s.predict(X_quality)
rho_s, p_s = stats.spearmanr(llm_resid, surprise_resid)
print(f"Partial correlation (LLM consensus vs Surprise | quality): ρ = {rho_s:.4f}, p = {p_s:.2e}")
print(f"(Negative would mean: LLM-disliked stories are more surprising to humans, controlling for quality)")

# ─── Analysis 6: Extreme outsiders ───────────────────────────────────────────
print("\n" + "="*70)
print("ANALYSIS 6: EXTREME OUTSIDERS")
print("="*70)

# Stories where ALL 3 LLMs rate low (<=2) vs ALL rate high (>=4)
all_low = merged[(merged["llm_gpt_4o"] <= 2) & (merged["llm_gpt_4o_mini"] <= 2) & (merged["llm_gpt_4_1_mini"] <= 2)]
all_high = merged[(merged["llm_gpt_4o"] >= 4) & (merged["llm_gpt_4o_mini"] >= 4) & (merged["llm_gpt_4_1_mini"] >= 4)]
print(f"All LLMs rate <=2: {len(all_low)} stories, human interest = {all_low['human_interestingness'].mean():.3f}")
print(f"All LLMs rate >=4: {len(all_high)} stories, human interest = {all_high['human_interestingness'].mean():.3f}")
if len(all_low) > 0 and len(all_high) > 0:
    u, p = stats.mannwhitneyu(all_low["human_interestingness"], all_high["human_interestingness"], alternative="greater")
    print(f"Mann-Whitney (low > high): p = {p:.2e}")

# Stories where LLMs DISAGREE (high std) but rate moderately
high_disagree = merged[merged["llm_std"] > merged["llm_std"].quantile(0.75)]
low_disagree = merged[merged["llm_std"] < merged["llm_std"].quantile(0.25)]
print(f"\nHigh LLM disagreement stories: n={len(high_disagree)}, human interest = {high_disagree['human_interestingness'].mean():.3f}")
print(f"Low LLM disagreement stories: n={len(low_disagree)}, human interest = {low_disagree['human_interestingness'].mean():.3f}")
u2, p2 = stats.mannwhitneyu(high_disagree["human_interestingness"], low_disagree["human_interestingness"], alternative="greater")
print(f"Mann-Whitney (high disagree > low disagree): p = {p2:.2e}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY OF ALL QUALITY-CONTROLLED TESTS")
print("="*70)
print(f"""
Raw correlation (LLM consensus vs human interest): ρ = +0.48 (REFUTES hypothesis)
Partial correlation (controlling for quality):      ρ = {rho_partial:+.4f} {'(SUPPORTS)' if rho_partial < 0 else '(REFUTES)'}
Within-model average:                               ρ = {avg_within:+.4f} {'(SUPPORTS)' if avg_within < 0 else '(REFUTES)'}
LLM-only partial:                                   ρ = {rho_llm:+.4f} {'(SUPPORTS)' if rho_llm < 0 else '(REFUTES)'}
Surprise partial:                                   ρ = {rho_s:+.4f} {'(SUPPORTS)' if rho_s < 0 else '(REFUTES)'}
High-disagree > Low-disagree human interest:        p = {p2:.2e}
""")

# Save
results = {
    "high_quality_rho": float(rho), "high_quality_p": float(p),
    "partial_rho": float(rho_partial), "partial_p": float(p_partial),
    "within_model_avg_rho": float(avg_within),
    "llm_only_partial_rho": float(rho_llm), "llm_only_partial_p": float(p_llm),
    "surprise_partial_rho": float(rho_s), "surprise_partial_p": float(p_s),
}
with open("results/quality_controlled_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to results/quality_controlled_results.json")
