"""
Step 5: Complementary analysis using Arena Human Preference 55K dataset.
Tests whether LLM judge ratings predict which response humans chose in head-to-head battles.
"""
import json
import os
import time
import random
import numpy as np
import pandas as pd
from scipy import stats
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load Arena data
from datasets import load_from_disk
ds = load_from_disk("datasets/arena_human_preference")
df = pd.DataFrame(ds["train"])
print(f"Arena dataset: {len(df)} battles")
print(f"Columns: {df.columns.tolist()}")

# Derive winner column from the three boolean columns
df["winner"] = "tie"
df.loc[df["winner_model_a"] == 1, "winner"] = "model_a"
df.loc[df["winner_model_b"] == 1, "winner"] = "model_b"
print(f"\nWinner distribution:\n{df['winner'].value_counts()}")
print(f"\nSample models: {df['model_a'].value_counts().head(10)}")

# Sample strategy: Take battles where there's a clear winner (model_a or model_b)
clear_winners = df[df["winner"].isin(["model_a", "model_b"])].copy()
print(f"\nClear winner battles: {len(clear_winners)}")

# Sample 200 battles for LLM evaluation (cost-effective)
sample = clear_winners.sample(n=min(200, len(clear_winners)), random_state=SEED)
print(f"Sampled {len(sample)} battles for LLM evaluation")

RATING_PROMPT = """You are evaluating a response for how INTERESTING and ENGAGING it is.

Rate the following response on a scale of 1-5 for "interestingness":
1 = Extremely boring, generic, predictable
2 = Somewhat dull, mostly boilerplate
3 = Moderately interesting, some engaging elements
4 = Quite interesting, engaging and thoughtful
5 = Extremely interesting, captivating, insightful

IMPORTANT: Respond with ONLY a JSON object with keys "score" (integer 1-5) and "reason" (one sentence).

USER QUESTION:
"""

MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"]

def rate_response(text, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": RATING_PROMPT + text[:4000]}],
                temperature=0.0,
                max_tokens=100,
                seed=SEED,
            )
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result = json.loads(result_text)
            return int(result["score"])
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None

# For each battle, rate both responses with all 3 LLM judges
results = []
results_file = "results/arena_llm_ratings.json"

if os.path.exists(results_file):
    with open(results_file) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} existing arena results")
else:
    print("Rating Arena responses...")

    def process_battle(row):
        idx = row.name
        prompt = row.get("prompt", row.get("conversation_a", ""))
        if isinstance(prompt, list):
            prompt = " ".join([str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in prompt[:2]])

        response_a = row.get("response_a", row.get("conversation_a", ""))
        response_b = row.get("response_b", row.get("conversation_b", ""))
        if isinstance(response_a, list):
            response_a = " ".join([str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in response_a])
        if isinstance(response_b, list):
            response_b = " ".join([str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in response_b])

        # Build text for rating
        text_a = f"{str(prompt)[:500]}\n\nRESPONSE:\n{str(response_a)[:3000]}"
        text_b = f"{str(prompt)[:500]}\n\nRESPONSE:\n{str(response_b)[:3000]}"

        battle_result = {
            "idx": int(idx),
            "winner": row["winner"],
            "model_a": row.get("model_a", "unknown"),
            "model_b": row.get("model_b", "unknown"),
        }

        for model in MODELS:
            score_a = rate_response(text_a, model)
            score_b = rate_response(text_b, model)
            battle_result[f"{model}_score_a"] = score_a
            battle_result[f"{model}_score_b"] = score_b

        return battle_result

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_battle, row): i
                   for i, (_, row) in enumerate(sample.iterrows())}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Arena battles"):
            result = future.result()
            results.append(result)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} arena results")

# ─── Analysis ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("ARENA ANALYSIS")
print("="*70)

rdf = pd.DataFrame(results)

# For each LLM judge, compute agreement with human preference
print("\nLLM judge agreement with human preference:")
for model in MODELS:
    col_a = f"{model}_score_a"
    col_b = f"{model}_score_b"

    valid = rdf.dropna(subset=[col_a, col_b]).copy()

    # LLM prefers A if score_a > score_b
    valid["llm_prefers_a"] = valid[col_a] > valid[col_b]
    valid["llm_prefers_b"] = valid[col_b] > valid[col_a]
    valid["llm_tie"] = valid[col_a] == valid[col_b]
    valid["human_prefers_a"] = valid["winner"] == "model_a"

    # Agreement rate (excluding ties)
    non_tie = valid[~valid["llm_tie"]]
    if len(non_tie) > 0:
        agreement = ((non_tie["llm_prefers_a"] & non_tie["human_prefers_a"]) |
                     (non_tie["llm_prefers_b"] & ~non_tie["human_prefers_a"])).mean()
        print(f"  {model}: {agreement:.3f} agreement ({len(non_tie)} non-tie battles, "
              f"{valid['llm_tie'].sum()} ties)")

# Compute LLM consensus for each response
for side in ["a", "b"]:
    score_cols = [f"{m}_score_{side}" for m in MODELS]
    rdf[f"llm_mean_{side}"] = rdf[score_cols].mean(axis=1)
    rdf[f"llm_std_{side}"] = rdf[score_cols].std(axis=1)

# Key test: Does the human-preferred response have LOWER LLM consensus?
rdf["winner_llm_mean"] = np.where(rdf["winner"] == "model_a", rdf["llm_mean_a"], rdf["llm_mean_b"])
rdf["loser_llm_mean"] = np.where(rdf["winner"] == "model_a", rdf["llm_mean_b"], rdf["llm_mean_a"])
rdf["winner_llm_std"] = np.where(rdf["winner"] == "model_a", rdf["llm_std_a"], rdf["llm_std_b"])
rdf["loser_llm_std"] = np.where(rdf["winner"] == "model_a", rdf["llm_std_b"], rdf["llm_std_a"])

valid_rdf = rdf.dropna(subset=["winner_llm_mean", "loser_llm_mean"])

print(f"\nHuman-preferred response LLM mean: {valid_rdf['winner_llm_mean'].mean():.3f}")
print(f"Human-rejected response LLM mean: {valid_rdf['loser_llm_mean'].mean():.3f}")
t_stat, p_val = stats.ttest_rel(valid_rdf["winner_llm_mean"], valid_rdf["loser_llm_mean"])
print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.2e}")
if valid_rdf["winner_llm_mean"].mean() > valid_rdf["loser_llm_mean"].mean():
    print("→ LLMs rate the human-preferred response HIGHER (REFUTES outsider hypothesis)")
else:
    print("→ LLMs rate the human-preferred response LOWER (SUPPORTS outsider hypothesis)")

# Disagreement analysis
print(f"\nHuman-preferred response LLM disagreement: {valid_rdf['winner_llm_std'].mean():.3f}")
print(f"Human-rejected response LLM disagreement: {valid_rdf['loser_llm_std'].mean():.3f}")
t2, p2 = stats.ttest_rel(valid_rdf["winner_llm_std"], valid_rdf["loser_llm_std"])
print(f"Paired t-test: t = {t2:.3f}, p = {p2:.2e}")

# Compute fraction where human picked the "outsider" (lower LLM score)
rdf["human_picked_lower_llm"] = (
    ((rdf["winner"] == "model_a") & (rdf["llm_mean_a"] < rdf["llm_mean_b"])) |
    ((rdf["winner"] == "model_b") & (rdf["llm_mean_b"] < rdf["llm_mean_a"]))
)
rdf["llm_scores_equal"] = abs(rdf["llm_mean_a"] - rdf["llm_mean_b"]) < 0.01
non_equal = rdf[~rdf["llm_scores_equal"]]
outsider_rate = non_equal["human_picked_lower_llm"].mean()
print(f"\nFraction where human preferred the lower-LLM-rated response: {outsider_rate:.3f}")
print(f"(Expected 0.50 under null, >0.50 supports outsider hypothesis)")

# Binomial test
from scipy.stats import binomtest
n_success = int(non_equal["human_picked_lower_llm"].sum())
n_total = len(non_equal)
p_binom = binomtest(n_success, n_total, 0.5).pvalue
print(f"Binomial test: {n_success}/{n_total} = {n_success/n_total:.3f}, p = {p_binom:.3f}")

# Save
arena_results = {
    "n_battles": len(rdf),
    "winner_llm_mean": float(valid_rdf["winner_llm_mean"].mean()),
    "loser_llm_mean": float(valid_rdf["loser_llm_mean"].mean()),
    "paired_t": float(t_stat),
    "paired_p": float(p_val),
    "outsider_rate": float(outsider_rate),
    "binom_p": float(p_binom),
}
with open("results/arena_analysis_results.json", "w") as f:
    json.dump(arena_results, f, indent=2)

print("\nArena analysis complete.")
