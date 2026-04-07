"""
Step 1: Prepare HANNA dataset for the Artificial Outsiders experiment.
Aggregates human ratings per story and saves clean dataset.
"""
import json
import pandas as pd
import numpy as np
from datasets import load_from_disk

SEED = 42
np.random.seed(SEED)

# Load HANNA
ds = load_from_disk("datasets/hanna")
df = pd.DataFrame(ds["train"])

print(f"Raw HANNA: {len(df)} annotations for {df['Story_ID'].nunique()} stories")
print(f"Models: {sorted(df['Model'].unique())}")
print(f"Human dimensions: Relevance, Coherence, Empathy, Surprise, Engagement, Complexity")

# Aggregate human ratings per story (mean of 3 annotators)
dims = ["Relevance", "Coherence", "Empathy", "Surprise", "Engagement", "Complexity"]
agg = df.groupby(["Story_ID", "Model", "Prompt", "Story"])[dims].mean().reset_index()

# Compute human "interestingness" = mean(Surprise, Engagement)
agg["human_interestingness"] = (agg["Surprise"] + agg["Engagement"]) / 2

# Compute overall human quality
agg["human_quality"] = agg[dims].mean(axis=1)

print(f"\nAggregated: {len(agg)} unique stories")
print(f"\nHuman interestingness distribution:")
print(agg["human_interestingness"].describe())
print(f"\nBy model type:")
print(agg.groupby("Model")["human_interestingness"].mean().sort_values(ascending=False))

# Flag LLM-generated vs human-written
agg["is_llm"] = agg["Model"] != "Human"

# Save
agg.to_json("results/hanna_stories_prepared.json", orient="records", indent=2)
print(f"\nSaved {len(agg)} stories to results/hanna_stories_prepared.json")

# Print story length stats
agg["story_length"] = agg["Story"].str.len()
print(f"\nStory length (chars): mean={agg['story_length'].mean():.0f}, "
      f"median={agg['story_length'].median():.0f}, "
      f"max={agg['story_length'].max():.0f}")
