# Artificial Outsiders

Testing whether LLM-written text that other LLMs don't find interesting is paradoxically more interesting to humans.

## Key Findings

- **Hypothesis refuted**: LLM consensus *positively* correlates with human interestingness (Spearman ρ = +0.48, p < 10⁻⁶²). Stories all LLMs dislike are also boring to humans.
- **Quality explains everything**: After controlling for overall text quality, the LLM-interestingness correlation drops to ρ ≈ 0.00. LLMs detect quality, not "interestingness" per se.
- **Arena confirms**: In 200 Chatbot Arena battles, humans chose the LLM-preferred response 77% of the time (binomial p < 10⁻¹¹).
- **LLM judges agree**: Inter-judge correlation ρ = 0.71–0.77 across three OpenAI models (gpt-4o, gpt-4o-mini, gpt-4.1-mini).
- **Caveat**: Dataset uses GPT-2-era generators. The hypothesis may hold for modern high-quality LLM outputs where quality variance is smaller.

## Datasets

- **HANNA**: 1,056 stories, 11 generators, 6-dimension human ratings (3 annotators each)
- **Arena Human Preference 55K**: 200-battle subsample with LLM judge evaluations

## Reproduce

```bash
# Environment
uv venv && source .venv/bin/activate
uv pip install openai numpy pandas scipy matplotlib seaborn tqdm datasets scikit-learn

# Run pipeline (requires OPENAI_API_KEY)
python src/01_prepare_data.py          # Prepare HANNA dataset
python src/02_llm_evaluation.py        # Rate stories with 3 LLM judges (~5 min)
python src/03_analysis.py              # Core statistical analysis
python src/04_visualizations.py        # Generate figures
python src/05_arena_analysis.py        # Arena complementary analysis (~5 min)
python src/06_quality_controlled_analysis.py  # Quality-controlled tests
```

## File Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Research plan and methodology
├── src/
│   ├── 01_prepare_data.py     # HANNA data preparation
│   ├── 02_llm_evaluation.py   # Multi-LLM judge evaluation pipeline
│   ├── 03_analysis.py         # Core hypothesis tests
│   ├── 04_visualizations.py   # Figure generation
│   ├── 05_arena_analysis.py   # Arena dataset analysis
│   └── 06_quality_controlled_analysis.py  # Quality confound analysis
├── results/                   # Raw results and analysis outputs
├── figures/                   # Generated visualizations
├── datasets/                  # HANNA, Arena, MT-Bench, Story Writing Benchmark
├── papers/                    # 25 related research papers
└── code/                      # 4 baseline code repositories
```

See [REPORT.md](REPORT.md) for full methodology, results, and discussion.
