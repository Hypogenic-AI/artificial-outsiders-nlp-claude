"""
Step 2: Have multiple LLM judges rate each HANNA story on interestingness.
Uses 3 OpenAI models: gpt-4o, gpt-4o-mini, gpt-4.1-mini.
"""
import json
import os
import time
import random
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"]

RATING_PROMPT = """You are evaluating a short story for how INTERESTING it is to read.

Rate the following story on a scale of 1-5 for "interestingness":
1 = Extremely boring, predictable, nothing noteworthy
2 = Somewhat dull, mostly predictable
3 = Moderately interesting, some engaging elements
4 = Quite interesting, engaging and somewhat surprising
5 = Extremely interesting, captivating, surprising, memorable

Consider: Does the story grab your attention? Is it surprising or unexpected? Would you want to read more? Does it have original ideas or an unusual perspective?

IMPORTANT: Respond with ONLY a JSON object with keys "score" (integer 1-5) and "reason" (one sentence).

STORY:
"""


def build_prompt(story_text):
    return RATING_PROMPT + story_text[:3000]


def rate_story(story_id, story_text, model, max_retries=3):
    """Rate a single story with a single model."""
    prompt = build_prompt(story_text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                seed=SEED,
            )
            text = response.choices[0].message.content.strip()
            # Parse JSON response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            score = int(result["score"])
            if 1 <= score <= 5:
                return {
                    "story_id": story_id,
                    "model": model,
                    "score": score,
                    "reason": result.get("reason", ""),
                    "tokens_used": response.usage.total_tokens,
                }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed story {story_id} with {model}: {e}")
                return {
                    "story_id": story_id,
                    "model": model,
                    "score": None,
                    "reason": f"ERROR: {str(e)[:100]}",
                    "tokens_used": 0,
                }
    return None


def main():
    # Load prepared stories
    with open("results/hanna_stories_prepared.json") as f:
        stories = json.load(f)

    print(f"Loaded {len(stories)} stories")
    print(f"Models to evaluate: {MODELS}")

    # Check for existing results (resume capability)
    results_file = "results/llm_ratings.json"
    existing_results = []
    done_keys = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            existing_results = json.load(f)
        done_keys = {(r["story_id"], r["model"]) for r in existing_results}
        print(f"Resuming: {len(existing_results)} existing ratings found")

    # Build task list
    tasks = []
    for s in stories:
        for model in MODELS:
            key = (s["Story_ID"], model)
            if key not in done_keys:
                tasks.append((s["Story_ID"], s["Story"], model))

    print(f"Tasks remaining: {len(tasks)}")
    if not tasks:
        print("All done!")
        return

    # Rate stories with thread pool (parallel API calls)
    results = list(existing_results)
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {}
        for story_id, story_text, model in tasks:
            future = executor.submit(rate_story, story_id, story_text, model)
            futures[future] = (story_id, model)

        pbar = tqdm(total=len(futures), desc="Rating stories")
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
                total_tokens += result.get("tokens_used", 0)
            pbar.update(1)

            # Save checkpoint every 200 ratings
            if len(results) % 200 == 0:
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
        pbar.close()

    # Final save
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\nCompleted: {len(results)} total ratings")
    print(f"Total tokens: {total_tokens:,}")

    # Quick stats
    valid = [r for r in results if r["score"] is not None]
    for model in MODELS:
        model_scores = [r["score"] for r in valid if r["model"] == model]
        if model_scores:
            print(f"  {model}: mean={np.mean(model_scores):.2f}, "
                  f"std={np.std(model_scores):.2f}, n={len(model_scores)}")


if __name__ == "__main__":
    main()
