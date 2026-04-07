# Resources Catalog

## Summary
This document catalogs all resources gathered for the "Artificial Outsiders" research project.

**Hypothesis**: LLM-written documents that other LLMs do not find as interesting are more likely to be interesting to humans than LLM-written documents that all LLMs like.

### Papers
Total papers downloaded: 25

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | LLM Evaluators Recognize and Favor Their Own Generations | Panickssery et al. | 2024 | papers/panickssery2024_llm_self_preference.pdf | Self-recognition drives self-preference; 411 citations |
| 2 | Judging LLM-as-a-judge with MT-Bench and Chatbot Arena | Zheng et al. | 2023 | papers/zheng2023_mt_bench_chatbot_arena.pdf | Foundational LLM-as-judge paper; 7691 citations |
| 3 | Self-Preference Bias in LLM-as-a-Judge | Wataoka et al. | 2024 | papers/wataoka2024_self_preference_bias.pdf | Perplexity drives self-preference; 154 citations |
| 4 | CoBBLEr: Cognitive Biases in LLM Evaluators | Koo et al. | 2023 | papers/koo2023_cobbler_cognitive_biases.pdf | 6 biases, 44% human-machine alignment; 147 citations |
| 5 | Inconsistent and Biased Evaluators | Stureborg et al. | 2024 | papers/stureborg2024_inconsistent_biased.pdf | Familiarity bias confirmed; 114 citations |
| 6 | Not Fair Evaluators | Wang et al. | 2023 | papers/wang2023_not_fair_evaluators.pdf | Position bias + calibration; 892 citations |
| 7 | Justice or Prejudice? | Ye et al. | 2024 | papers/ye2024_justice_prejudice.pdf | 12 bias types framework; 276 citations |
| 8 | Style Over Substance | Wu, Aji | 2023 | papers/wu2023_style_over_substance.pdf | Fluency > factuality in evaluation; 66 citations |
| 9 | Humans or LLMs as Judge | Bavaresco et al. | 2024 | papers/bavaresco2024_llms_instead_human_judges.pdf | Human vs LLM bias comparison; 263 citations |
| 10 | Narcissistic Evaluators | Liu et al. | 2023 | papers/liu2023_narcissistic_evaluators.pdf | Self-model favoritism; 86 citations |
| 11 | Play Favorites | Spiliopoulou et al. | 2025 | papers/play_favorites_self_bias.pdf | Statistical self-bias framework; 17 citations |
| 12 | OffsetBias | Park et al. | 2024 | papers/offsetbias_debiased_evaluators.pdf | De-biasing evaluators; 90 citations |
| 13 | FLAMe Autoraters | Vu et al. | 2024 | papers/foundational_autoraters.pdf | Trained less-biased evaluators; 86 citations |
| 14 | No Free Labels | Krumdick et al. | 2025 | papers/krumdick2025_no_free_labels.pdf | LLM judges need human grounding; 39 citations |
| 15 | Diverging Preferences | 2024 | 2024 | papers/diverging_preferences.pdf | Annotator disagreement prediction; 40 citations |
| 16 | Generative AI: Creativity vs Diversity | Doshi, Hauser | 2024 | papers/doshi2024_generative_ai_creativity_diversity.pdf | AI boosts individual quality, reduces diversity; 436 citations |
| 17 | Echoes in AI: Plot Diversity | 2024 | 2024 | papers/echoes_ai_plot_diversity.pdf | LLMs converge on similar narratives; 34 citations |
| 18 | Art or Artifice? | Chakrabarty et al. | 2023 | papers/chakrabarty2023_art_artifice.pdf | LLM creativity analysis; 231 citations |
| 19 | Confederacy of Models | Gomez-Rodriguez, Williams | 2023 | papers/gomez2023_confederacy_creative_writing.pdf | Multi-model creative writing evaluation; 134 citations |
| 20 | Novel Research Ideas? | Si et al. | 2024 | papers/si2024_novel_research_ideas.pdf | LLM ideas rated novel but less feasible; 320 citations |
| 21 | AI as Salieri | Agarwal et al. | 2024 | papers/agarwal2024_salieri.pdf | Linguistic creativity quantification; 36 citations |
| 22 | Can AI Writing Be Salvaged? | 2024 | 2024 | papers/can_ai_writing_salvaged.pdf | Human-AI alignment through editing; 50 citations |
| 23 | Temperature and Creativity | 2024 | 2024 | papers/is_temperature_creativity.pdf | Temperature effects on diversity; 131 citations |
| 24 | Creativity in LLMs | Chen et al. | 2024 | papers/chen2024_creativity_llm.pdf | Creativity evaluation framework; 52 citations |
| 25 | AI Poetry Rated Favorably | Porter et al. | 2024 | papers/porter2024_ai_poetry_rated_favorably.pdf | Humans prefer AI poetry blindly; 99 citations |

See papers/README.md for detailed descriptions.

### Datasets
Total datasets downloaded: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| HANNA | HuggingFace llm-aes/hanna | 3,168 annotations | Story evaluation | datasets/hanna/ | 1,056 stories, 6-dim human ratings |
| MT-Bench Human Judgments | HuggingFace lmsys/mt_bench_human_judgments | 5,755 judgments | QA evaluation | datasets/mt_bench_human_judgments/ | Human + GPT-4 evaluations |
| Story Writing Benchmark | HuggingFace lars1234/story_writing_benchmark | 8,520 stories | Creative writing | datasets/story_writing_benchmark/ | 15 LLMs, multi-LLM grading |
| Arena Human Preference 55K | HuggingFace lmarena-ai/arena-human-preference-55k | 57,477 battles | Open-ended chat | datasets/arena_human_preference/ | 64+ models, human votes |

See datasets/README.md for detailed descriptions and download instructions.

### Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| CoBBLEr | github.com/minnesotanlp/cobbler | LLM evaluator bias benchmark | code/cobbler/ | 6 bias types |
| Confederacy of Models | github.com/komoku/confederacy-of-models | Creative writing evaluation | code/confederacy-of-models/ | 65 stories, human ratings |
| Lechmazur Writing | github.com/lechmazur/writing | Multi-LLM story benchmark | code/lechmazur-writing/ | 7 LLM graders |
| Self-Preference | github.com/zhiyuanc2001/self-preference | Self-preference bias measurement | code/self-preference/ | EMNLP 2025 |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
- Used paper-finder with diligent mode across 5 different query sets covering LLM evaluation bias, self-preference, creativity/diversity, and human preferences
- Aggregated 130 unique relevant papers from Semantic Scholar
- Selected 25 papers spanning the three core research pillars
- Searched HuggingFace, GitHub, and academic data repositories for datasets

### Selection Criteria
- Papers: Prioritized (1) self-preference/familiarity bias, (2) LLM output diversity, (3) human-AI preference divergence, (4) creative writing evaluation
- Datasets: Required either (a) multi-LLM generated text with human ratings, or (b) human preference data for LLM outputs
- Code: Tools for running multi-LLM evaluation and measuring bias

### Challenges Encountered
- Most datasets have either human OR LLM evaluations, rarely both. HANNA and MT-Bench are exceptions.
- No existing dataset directly tests the "outsider" hypothesis---experiment will need to generate multi-LLM evaluations on existing human-rated creative text.
- Some arXiv IDs from Semantic Scholar were incorrect and needed manual resolution.

### Gaps and Workarounds
- **No direct "interestingness" dataset**: Use HANNA's Surprise + Engagement dimensions as proxy
- **Story Writing Benchmark lacks human ratings**: Plan to add human evaluation on a subsample
- **No cross-model disagreement metrics**: Will need to compute from multi-LLM evaluation data

---

## Recommendations for Experiment Design

### Primary Dataset: HANNA
1. Load HANNA's 1,056 stories (from 10 generators including human)
2. Run 3-5 diverse LLM judges (GPT-4, Claude, Llama-3, Mistral, Gemma) on each story
3. Compute inter-LLM disagreement (variance, entropy of scores)
4. Correlate LLM disagreement with human Surprise and Engagement ratings
5. Test: Do stories with HIGH LLM disagreement have HIGHER human interest ratings?

### Complementary Analysis: Story Writing Benchmark
1. Use 8,520 stories from 15 LLMs with existing multi-LLM grading
2. Identify "outlier" stories (low consensus among LLM graders)
3. Run human evaluation on a stratified sample (high-consensus vs low-consensus)
4. Compare human ratings

### Baseline Methods
1. **Null hypothesis**: LLM consensus predicts human preference (higher LLM agreement = higher human rating)
2. **Single best judge**: GPT-4 alone predicts human interest
3. **Perplexity ranking**: Lower perplexity = more interesting (expect this to be wrong for our hypothesis)

### Evaluation Metrics
- Spearman/Kendall correlation between LLM disagreement and human interest
- ROC-AUC for predicting "interesting to humans" from LLM disagreement
- Conditional comparison: mean human rating for high-consensus vs low-consensus LLM groups

### Code to Adapt/Reuse
- **CoBBLEr**: Framework for running multi-LLM evaluations
- **Self-Preference**: Scripts for measuring cross-model preferences
- **Lechmazur Writing**: Multi-judge evaluation pipeline
- **FairEval**: Position debiasing (pip install from GitHub)
