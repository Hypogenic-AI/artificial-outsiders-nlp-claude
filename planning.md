# Research Plan: Artificial Outsiders

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used to evaluate text quality (LLM-as-a-judge), but they share systematic biases — preferring lower-perplexity, more familiar text (Wataoka et al., 2024) and exhibiting self-preference (Panickssery et al., 2024). If LLM judges converge on preferring homogeneous text, they may systematically undervalue the most interesting and novel content — exactly what matters most in creative writing, ideation, and content discovery. Understanding this bias is critical for anyone using LLM evaluation in production.

### Gap in Existing Work
Prior work documents (1) LLM self-preference bias, (2) LLM output convergence/low diversity, and (3) human-LLM preference misalignment — but **no study directly tests whether LLM-judged "outsiders" are more interesting to humans**. The literature treats these as separate phenomena. We unify them into one testable prediction.

### Our Novel Contribution
We test whether inter-LLM consensus on text quality is **inversely** correlated with human-judged interestingness. Specifically: LLM-written stories that multiple LLMs rate low or disagree about may be more surprising and engaging to human readers than stories all LLMs rate highly.

### Experiment Justification
- **Experiment 1 (HANNA dataset)**: Core hypothesis test. HANNA has 1,056 stories with ground-truth human ratings on Surprise and Engagement. We add multi-LLM judge ratings and test the correlation. This is needed because no existing dataset has both multi-LLM judge scores and human interestingness ratings for the same texts.
- **Experiment 2 (Arena dataset)**: Robustness check on conversational data. Arena has 55K human preference votes. We sample responses, get LLM ratings, and check whether human-preferred responses are ones LLMs disagree about.
- **Experiment 3 (Controlled generation)**: We generate stories at different creativity levels (temperature), have LLMs judge them, and see which temperature settings produce "outsiders" that score differently with LLMs vs. would with humans (using HANNA-calibrated patterns).

## Research Question
Do LLM-written texts that receive low or divergent ratings from multiple LLM judges receive higher human ratings on interestingness (surprise + engagement) compared to texts that all LLM judges rate highly?

## Hypothesis Decomposition
- **H1**: LLM judge consensus (mean rating across judges) is negatively correlated with human Surprise ratings.
- **H2**: LLM judge disagreement (variance across judges) is positively correlated with human Engagement ratings.
- **H3**: Stories in the bottom quartile of LLM consensus have higher human interestingness than stories in the top quartile.
- **H4**: The effect holds specifically for LLM-generated stories (not just human-written ones which LLMs might rate low for other reasons).

## Proposed Methodology

### Approach
Use the HANNA dataset as primary testbed. It has 1,056 stories from 11 generators (10 LLMs + human) with 3 human annotators each rating Surprise, Engagement, and 4 other dimensions. We add ratings from 3 diverse OpenAI LLM judges (GPT-4o, GPT-4o-mini, GPT-4.1) and test correlations.

### Experimental Steps
1. Load and preprocess HANNA: aggregate human ratings per story (mean of 3 annotators)
2. Design interestingness rating prompt for LLM judges
3. Call 3 LLM models to rate each of 1,056 stories on a 1-5 interestingness scale
4. Compute per-story: LLM consensus (mean), LLM disagreement (std dev), human interestingness (mean of Surprise + Engagement)
5. Statistical analysis: correlations, group comparisons, regression
6. Complementary analysis on Arena Human Preference 55K subset

### Baselines
- **Null hypothesis**: LLM consensus positively predicts human interest (higher LLM mean → higher human interest)
- **Single-judge baseline**: GPT-4o alone predicts human interest
- **Quality baseline**: Overall quality (mean of all 6 human dimensions) vs. interestingness-specific signal

### Evaluation Metrics
- Spearman's rank correlation between LLM consensus and human interestingness
- Spearman's rank correlation between LLM disagreement and human interestingness
- Mann-Whitney U test comparing human ratings of top vs bottom LLM consensus quartiles
- Effect sizes (Cohen's d)

### Statistical Analysis Plan
- Primary: Spearman correlations with 95% bootstrap CIs
- Group comparison: Mann-Whitney U (non-parametric, no normality assumption)
- Multiple comparison correction: Bonferroni for the 4 hypothesis tests
- Significance level: α = 0.05 (Bonferroni-corrected: 0.0125)
- Robustness: repeat analysis separately for LLM-generated vs. human-written stories

## Expected Outcomes
- **Supporting H**: Negative correlation between LLM consensus and human Surprise/Engagement, especially for LLM-generated text
- **Refuting H**: Positive correlation or no correlation — LLM judges and humans agree on interestingness
- **Nuanced**: Effect may depend on story source model — newer/better LLMs may produce "outliers" that are genuinely bad rather than interestingly different

## Timeline
1. Environment & data prep: 15 min
2. LLM evaluation pipeline: 30 min (implementation + API calls)
3. Statistical analysis: 30 min
4. Complementary experiments: 30 min
5. Documentation: 30 min

## Potential Challenges
- API rate limits → use exponential backoff
- HANNA's LLM models are old (GPT-2 era) → findings may not generalize to modern LLMs → address with Arena analysis
- 3 LLM judges may not capture enough variance → mitigate by using architecturally diverse models
- "Interestingness" is subjective → use established Surprise + Engagement dimensions from HANNA

## Success Criteria
- Clear statistical test of all 4 sub-hypotheses with proper correction
- At least one significant result (positive or negative) with adequate effect size
- Honest reporting regardless of direction
