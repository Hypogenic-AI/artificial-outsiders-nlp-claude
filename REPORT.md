# Artificial Outsiders: Testing Whether LLM-Disliked Text Is More Interesting to Humans

## 1. Executive Summary

We tested whether LLM-written documents that other LLMs rate as uninteresting are paradoxically *more* interesting to humans—the "artificial outsider" hypothesis. Using the HANNA dataset (1,056 stories from 11 generators with ground-truth human ratings) and 3,168 LLM judge evaluations across 3 OpenAI models, we find the hypothesis is **definitively refuted**. LLM consensus positively correlates with human interestingness (Spearman ρ = 0.48, p < 10⁻⁶²), and stories all LLMs rate low receive significantly lower human interest ratings. A complementary analysis of 200 Chatbot Arena battles confirms this: humans chose the LLM-preferred response 77% of the time. However, controlling for overall text quality, the LLM-interestingness correlation drops to near zero (ρ ≈ 0.00), revealing that LLMs are primarily detecting quality rather than having a unique bias for or against "interesting" content.

## 2. Research Question & Motivation

**Hypothesis**: LLM-written documents that other LLMs do not find as interesting are more likely to be interesting to humans than LLM-written documents that all LLMs like.

**Why it matters**: LLMs are increasingly used as automated evaluators (LLM-as-a-judge). Prior work shows they share systematic biases—self-preference (Panickssery et al., 2024), perplexity preference (Wataoka et al., 2024), and output convergence (Echoes in AI, 2024). If LLM judges systematically undervalue interesting content, this would have major implications for content curation, recommendation, and creative AI evaluation pipelines.

**Gap**: While LLM evaluation biases and output homogeneity are well-documented separately, no study has directly tested whether LLM consensus inversely predicts human interestingness.

## 3. Methodology

### 3.1 Experimental Setup

**Primary dataset**: HANNA (Chhun et al., 2022) — 1,056 stories from 11 generators (10 LLMs + human), each rated by 3 human annotators on 6 dimensions (Relevance, Coherence, Empathy, Surprise, Engagement, Complexity) on a 1-5 Likert scale.

**Human interestingness proxy**: Mean of Surprise and Engagement ratings, averaged across 3 annotators.

**LLM judges**: Three OpenAI models (gpt-4o, gpt-4o-mini, gpt-4.1-mini) rated each story on a 1-5 interestingness scale at temperature=0 with seed=42.

**Complementary dataset**: Arena Human Preference 55K (200 battle subsample), where both responses were rated by the 3 LLM judges and compared to the human preference vote.

### 3.2 Statistical Tests

| Test | Description | Multiple comparison |
|------|-------------|-------------------|
| H1 | Spearman ρ: LLM consensus vs Human Surprise | Bonferroni α = 0.0125 |
| H2 | Spearman ρ: LLM disagreement (std) vs Human Engagement | Bonferroni α = 0.0125 |
| H3 | Mann-Whitney U: bottom vs top LLM consensus quartile | Bonferroni α = 0.0125 |
| H4 | Spearman ρ: LLM consensus vs Human interest (LLM-generated only) | Bonferroni α = 0.0125 |

Additional quality-controlled analyses: partial correlations (regressing out human quality), within-model correlations, and extreme-group comparisons.

### 3.3 Environment

- Python 3.12, NumPy 2.2.5, Pandas 2.2.3, SciPy 1.15.2
- OpenAI API: gpt-4o, gpt-4o-mini, gpt-4.1-mini
- 3,168 story evaluations + 1,200 arena response evaluations
- Random seed: 42 throughout
- Hardware: 4x NVIDIA RTX A6000 (not needed for this API-based study)

## 4. Results

### 4.1 Primary Hypothesis Tests

| Hypothesis | Statistic | Value | p-value | Direction | Verdict |
|-----------|-----------|-------|---------|-----------|---------|
| H1: LLM consensus vs Surprise | Spearman ρ | +0.402 | 3.07e-42 | Positive | **REFUTED** |
| H2: LLM disagreement vs Engagement | Spearman ρ | +0.143 | 3.23e-06 | Positive | Supports (weakly) |
| H3: Bottom vs Top LLM quartile | Mann-Whitney U | — | 1.00 | Low < High | **REFUTED** |
| H4: LLM-only consensus vs interest | Spearman ρ | +0.358 | 1.81e-30 | Positive | **REFUTED** |

**Interpretation**: 3 of 4 tests strongly refute the hypothesis. The one supporting result (H2: disagreement correlates with engagement, ρ=0.14) has a weak effect size and may reflect that more variable stories are inherently more complex.

### 4.2 LLM Consensus Quartile Analysis

| LLM Consensus Quartile | N | Human Interest (mean ± SEM) | Surprise | Engagement |
|------------------------|---|------------------------------|----------|------------|
| Q1 (low consensus) | 467 | 2.093 ± 0.026 | 1.858 | 2.328 |
| Q2 | 185 | 2.344 ± 0.040 | 2.050 | 2.638 |
| Q3 | 211 | 2.470 ± 0.039 | 2.148 | 2.791 |
| Q4 (high consensus) | 193 | 3.073 ± 0.053 | 2.720 | 3.427 |

Stories in the top LLM consensus quartile have **47% higher** human interestingness than the bottom quartile (3.07 vs 2.09). The relationship is monotonically increasing—exactly opposite to the hypothesis.

### 4.3 Quality-Controlled Analyses

| Analysis | ρ | p-value | Interpretation |
|----------|---|---------|---------------|
| Raw correlation | +0.482 | 1.65e-62 | Strong positive |
| Partial (controlling for quality) | +0.002 | 0.943 | **Near zero** |
| Within-model average | +0.205 | — | Weak positive |
| LLM-only, quality-controlled | +0.023 | 0.470 | Near zero |
| Surprise, quality-controlled | +0.010 | 0.746 | Near zero |

**Key finding**: The raw correlation is almost entirely explained by text quality. After partialing out human-rated quality, LLM consensus has essentially zero correlation with human interestingness (ρ = 0.002). LLMs are not biased for or against interesting content—they are simply detecting the same quality signal humans detect.

### 4.4 Extreme Group Comparison

| Group | N | Human Interest |
|-------|---|---------------|
| All 3 LLMs rate ≤ 2 | 463 | 2.091 |
| All 3 LLMs rate ≥ 4 | 66 | 3.548 |

Stories unanimously disliked by all LLM judges are **not** hidden gems—they are genuinely less interesting to humans too.

### 4.5 Arena Complementary Analysis (200 battles)

| Metric | Value |
|--------|-------|
| LLM judge agreement with human preference | 77-84% |
| Human-preferred response LLM mean | 3.29 |
| Human-rejected response LLM mean | 2.88 |
| Paired t-test | t = 7.88, p = 2.1e-13 |
| Fraction humans chose lower-LLM-rated response | 22.8% |
| Binomial test vs 50% | p = 1.74e-11 |

LLM judges align strongly with human preferences in head-to-head battles, with 77-84% agreement rates depending on the model.

### 4.6 Inter-LLM Agreement

| Model Pair | Spearman ρ |
|-----------|-----------|
| GPT-4.1-mini vs GPT-4o | 0.768 |
| GPT-4.1-mini vs GPT-4o-mini | 0.707 |
| GPT-4o vs GPT-4o-mini | 0.766 |

The three LLM judges show high agreement (ρ = 0.71–0.77), consistent with prior work on LLM evaluation convergence.

### 4.7 Per-Model Correlation with Human Interest

All source models show positive or near-zero correlations between LLM consensus and human interestingness. Only CTRL shows a very weak negative (ρ = -0.05, ns). The within-model average is ρ = +0.21, confirming the positive relationship holds even after controlling for model-level quality differences.

## 5. Analysis & Discussion

### 5.1 Why the Hypothesis Is Refuted

The "artificial outsider" hypothesis rests on three premises:
1. LLMs share evaluation biases that make them prefer homogeneous text
2. Text that deviates from the LLM-preferred distribution is more novel
3. Novelty/deviation translates to higher human interest

Our results show that **premise 3 fails**. While premises 1 and 2 may hold in some form (the LLM judges do agree closely with each other, ρ ≈ 0.75), the deviation from LLM preferences does not predict higher human interest. Instead, LLM-disliked text is simply lower quality, and humans also find lower-quality text less interesting.

### 5.2 The Quality Confound

The most important finding is the quality confound analysis. The raw ρ = +0.48 drops to ρ ≈ 0.00 after controlling for quality. This means:
- LLMs and humans agree almost perfectly on what constitutes good vs. bad text
- Beyond quality, LLMs have no additional bias for or against "interesting" content
- The outsider hypothesis conflates "quality outlier" with "interestingness outlier"

### 5.3 When Might the Hypothesis Hold?

The hypothesis could still hold in specific conditions not tested here:
1. **Modern high-quality LLMs**: HANNA uses old models (GPT-2 era) with high quality variance. For modern models where all outputs are competent, the quality confound is weaker, and the interestingness signal might emerge.
2. **Specific domains**: Creative writing at the frontier (poetry, literary fiction) where "interesting" means "rule-breaking" rather than "well-executed."
3. **Different operationalization**: "Interestingness" may need to be measured as novelty or surprise specifically, not overall engagement.

### 5.4 Comparison with Literature

Our finding that LLM judges achieve 77-84% agreement with human preferences aligns with Zheng et al. (2023), who report ~80% GPT-4-human agreement on MT-Bench. The high inter-LLM agreement (ρ ≈ 0.75) confirms the convergence documented by Echoes in AI (2024). However, our quality-controlled result (ρ ≈ 0.00) is novel and suggests that while LLMs share biases, these biases are largely the *same* biases humans have.

## 6. Limitations

1. **Dataset age**: HANNA's generators are from the GPT-2 era (2019-2022). Text quality varies from near-gibberish to coherent. The hypothesis was likely intended for modern, high-quality LLM text.
2. **LLM judge diversity**: All 3 judges are OpenAI models. Using models from different families (Claude, Gemini, Llama) could reveal more meaningful disagreement.
3. **Interestingness proxy**: We used Surprise + Engagement from HANNA, not a direct "interestingness" rating. These may not capture the full concept.
4. **Domain specificity**: Results are from short creative fiction (HANNA) and general chatbot conversations (Arena). Other domains (research ideas, essays, journalism) may differ.
5. **Sample size for Arena**: Only 200 battles were evaluated, limiting statistical power for subgroup analyses.

## 7. Conclusions & Next Steps

**Answer to research question**: The hypothesis that LLM-disliked text is more interesting to humans is **not supported**. LLM consensus positively predicts human interestingness (ρ = +0.48), driven primarily by shared quality perception. After controlling for quality, the correlation drops to near zero—LLMs show no special bias for or against interesting content.

**Practical implications**: LLM-as-a-judge is a reasonable proxy for human interestingness judgments in creative writing evaluation. Content curators can use LLM ratings as a first filter without systematically missing "hidden gems."

**Recommended follow-up experiments**:
1. Repeat with modern LLM outputs (GPT-4, Claude 3.5, Gemini 2.5) where quality is uniformly high, isolating the interestingness signal
2. Use diverse LLM judge families (not just OpenAI) to test if cross-family disagreement predicts human interest differently
3. Collect direct human "interestingness" ratings (not proxy through Surprise + Engagement)
4. Test in domains beyond creative fiction—research abstracts, news articles, product descriptions

## References

1. Panickssery et al. (2024). "LLM Evaluators Recognize and Favor Their Own Generations." arXiv:2404.13076
2. Wataoka et al. (2024). "Self-Preference Bias in LLM-as-a-Judge." arXiv:2410.21819
3. Zheng et al. (2023). "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena." arXiv:2306.05685
4. Koo et al. (2023). "CoBBLEr: Cognitive Bias Benchmark for LLMs as Evaluators." arXiv:2309.17012
5. Echoes in AI (2024). "Quantifying Lack of Plot Diversity in LLM Outputs." arXiv:2501.00273
6. Doshi & Hauser (2024). "Generative AI Enhances Creativity but Reduces Diversity." Science Advances
7. Wu & Aji (2023). "Style Over Substance." arXiv:2307.03025
8. Chhun et al. (2022). "HANNA: Human-Annotated NArrative dataset." (HuggingFace: llm-aes/hanna)
9. LMSYS (2023). "Arena Human Preference 55K." (HuggingFace: lmarena-ai/arena-human-preference-55k)
