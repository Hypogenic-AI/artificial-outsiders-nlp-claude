# Literature Review: Artificial Outsiders

## Research Hypothesis
LLM-written documents that other LLMs do not find as interesting are more likely to be interesting to humans than LLM-written documents that all LLMs like.

---

## Research Area Overview

This hypothesis sits at the intersection of three active research areas: (1) **LLM-as-a-Judge** evaluation methodology and its biases, (2) **human vs. AI preferences** for generated text, and (3) **diversity and creativity** in LLM outputs. The core insight is that LLM evaluators share systematic biases (self-preference, perplexity preference, style-over-substance) that cause them to converge on preferring certain types of text. Text that breaks these patterns---"artificial outsiders"---may actually be more interesting to humans precisely because it deviates from the homogeneous distribution LLMs both produce and prefer.

---

## Key Papers

### 1. LLM Self-Preference and Recognition

**Panickssery et al. (2024) - "LLM Evaluators Recognize and Favor Their Own Generations"** [arXiv:2404.13076]
- **Key Finding**: LLMs (GPT-4, GPT-3.5, Llama 2) exhibit significant self-preference bias, rating their own outputs higher than competitors'. Crucially, they can recognize their own outputs with non-trivial accuracy (GPT-4: 73.5%).
- **Methodology**: Fine-tuning experiments on XSUM and CNN/DailyMail show a linear correlation between self-recognition capability and self-preference strength.
- **Implication for our research**: If LLMs prefer their own outputs, then text that multiple LLMs dislike may be text that deviates from the shared LLM "style"---potentially more novel/interesting to humans.
- **Code**: https://bit.ly/llm_self_recognition

**Wataoka et al. (2024) - "Self-Preference Bias in LLM-as-a-Judge"** [arXiv:2410.21819]
- **Key Finding**: Self-preference is driven by **perplexity preference**---LLMs assign higher scores to lower-perplexity (more familiar) text, regardless of authorship. GPT-4 shows self-preference bias of 0.52 on their metric.
- **Methodology**: Novel Equal Opportunity-based metric on Chatbot Arena data (33K dialogues, 8 LLMs). Analyzed winning judgment rates conditioned on perplexity differences.
- **Critical insight**: The bias is about familiarity, not identity. LLMs prefer text that looks like what they would generate. This directly supports our hypothesis: "outsider" text (high perplexity to multiple LLMs) deviates from the shared distribution.
- **Datasets**: Chatbot Arena

**Koo et al. (2023) - "CoBBLEr: Cognitive Bias Benchmark for LLMs as Evaluators"** [arXiv:2309.17012]
- **Key Finding**: LLMs exhibit 6 cognitive biases as evaluators, with egocentric bias (self-preference) being prominent. Average RBO between human and machine preferences is only 0.44, indicating substantial misalignment.
- **Methodology**: 16 LLMs evaluated pairwise on 50 QA instructions from ELI5 and BigBench, creating 630K comparison samples.
- **Implication**: The 56% misalignment between humans and LLMs suggests systematic divergence in what each values.
- **Code**: https://github.com/minnesotanlp/cobbler

### 2. LLM Evaluation Biases

**Zheng et al. (2023) - "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena"** [arXiv:2306.05685]
- **Key Finding**: Strong LLM judges (GPT-4) achieve ~80% agreement with humans, but exhibit position, verbosity, and self-enhancement biases. This is the same agreement level as human-human.
- **Datasets**: MT-bench (80 multi-turn questions), Chatbot Arena (30K conversations). Both publicly available.
- **Code**: https://github.com/lm-sys/FastChat

**Wu & Aji (2023) - "Style Over Substance"** [arXiv:2307.03025]
- **Key Finding**: Both human and LLM evaluators prefer longer, grammatically polished text over shorter or less fluent text---even when the polished text contains factual errors. Factually incorrect answers are rated higher than grammatically imperfect ones.
- **Methodology**: GPT-4-generated answers with controlled flaws (factual errors, grammar errors, short length) evaluated by crowd workers, experts, GPT-4, and Claude-1 using Elo ratings.
- **Implication**: LLMs and humans share some biases (length, fluency) but diverge on others (factual accuracy). The "style over substance" bias means LLMs may universally prefer polished-but-bland text over rough-but-interesting text.

**Wang et al. (2023) - "Large Language Models are not Fair Evaluators"** [arXiv:2310.01432]
- **Key Finding**: LLM evaluation rankings can be easily manipulated by altering response order. Proposes calibration strategies.
- **Code**: https://github.com/i-Eval/FairEval

**Stureborg et al. (2024) - "Large Language Models are Inconsistent and Biased Evaluators"** [arXiv:2405.01724]
- **Key Finding**: LLMs show familiarity bias (preference for lower-perplexity text), skewed rating distributions, and anchoring effects. Low inter-sample agreement shows inconsistency.
- **Datasets**: SummEval, RoSE

### 3. LLM Output Diversity and Creativity

**Echoes in AI (2024) - "Quantifying Lack of Plot Diversity in LLM Outputs"** [arXiv:2501.00273]
- **Key Finding**: LLM outputs are dramatically less diverse than human writing. They introduce the **Sui Generis score** measuring plot-level uniqueness. Human stories score significantly higher. Even across different LLMs, outputs converge on similar narrative solutions.
- **Methodology**: 100 stories from WritingPrompts and Wikipedia TV plots. Generated 20 alternative continuations at each position. GPT-4 judges semantic entailment.
- **Critical insight**: Cross-model echo rates show LLaMA-3 outputs appear in GPT-4 continuations almost as frequently as in their own---LLMs share a common "boring" distribution. Text outside this distribution (low echo rate) is genuinely more unique.
- **Implication**: Directly supports our hypothesis. The "artificial outsider" is text in the tail of the shared LLM distribution.

**Doshi & Hauser (2024) - "Generative AI Enhances Creativity but Reduces Diversity"** [arXiv:2312.00506, Science Advances]
- **Key Finding**: GenAI access makes individual stories better-written and more enjoyable (especially for less creative writers), but GenAI-enabled stories are more similar to each other. Individual creativity up, collective diversity down.
- **Methodology**: Pre-registered experiment with 293 writers (3 conditions: no AI, 1 AI idea, 5 AI ideas) evaluated by 600 evaluators on novelty, usefulness, and emotional characteristics.
- **Implication**: The creativity-diversity tradeoff is central to our hypothesis. GenAI homogenizes content, and the rare outliers that break this pattern may be disproportionately interesting.

**Chakrabarty et al. (2023) - "Art or Artifice?"** [arXiv:2309.14556]
- **Key Finding**: LLMs generate text rated as creative by automated metrics but lacking in genuine novelty and surprise when closely analyzed by experts.
- **Evaluation**: Comprehensive multi-model evaluation on creative writing tasks.

**Gomez-Rodriguez & Williams (2023) - "A Confederacy of Models"** [arXiv:2310.08433]
- **Key Finding**: Blind human evaluation of 65 stories from 13 LLMs found significant quality variation across models, with human-written stories often preferred for originality and humor.
- **Datasets**: 65 stories with human ratings (fluency, coherence, originality, humor, style).
- **Code**: https://github.com/komoku/confederacy-of-models

### 4. Human vs. AI Preferences

**Porter et al. (2024) - "AI-generated Poetry Rated More Favorably"** [Scientific Reports]
- **Key Finding**: Humans rated AI-generated poems higher than human-written poems in a blind study, possibly because AI poems are more "accessible" and less challenging.
- **Implication**: Suggests humans may prefer "easier" AI text in some contexts, but this may be about surface appeal vs. depth.

**Si et al. (2024) - "Can LLMs Generate Novel Research Ideas?"** [arXiv:2409.04109]
- **Key Finding**: LLM-generated research ideas were rated as more novel than expert-written ones but less feasible. Experts judged LLM ideas as sometimes "surprisingly creative."

**Diverging Preferences (2024)** [arXiv:2410.14632]
- **Key Finding**: Examines when annotators disagree and whether models can predict disagreement. Relevant to understanding when humans diverge from LLM consensus.

---

## Common Methodologies

1. **Pairwise comparison with Elo/Bradley-Terry**: Used in Chatbot Arena, MT-Bench, CoBBLEr. Standard for ranking LLM outputs.
2. **Multi-dimensional Likert rating**: Used in HANNA (6 dimensions), MERS (3 dimensions). Better for measuring specific quality aspects.
3. **Perplexity-based analysis**: Wataoka et al., Stureborg et al. show perplexity as a proxy for LLM familiarity.
4. **Controlled perturbation**: Wu & Aji, OffsetBias introduce controlled flaws to measure bias sensitivity.
5. **Cross-model evaluation**: CoBBLEr, Echoes in AI evaluate how different LLMs judge each other's outputs.

## Standard Baselines

- **Random baseline**: Expected agreement if judges were random
- **GPT-4 as judge**: De facto standard LLM judge (Zheng et al., 2023)
- **Human inter-annotator agreement**: Typically 60-80% pairwise agreement
- **Perplexity-based ranking**: Rank by model perplexity as a null hypothesis

## Evaluation Metrics

- **Inter-judge agreement** (Cohen's kappa, Fleiss' kappa, RBO): Measures consensus among LLM judges
- **Human-LLM correlation** (Spearman's rho, Kendall's tau): Measures alignment between human and LLM preferences
- **Self-preference bias** (Wataoka metric, Panickssery fine-tuning correlation): Quantifies egocentric evaluation
- **Sui Generis score**: Measures plot-level uniqueness/surprise in narratives
- **Semantic similarity** (embedding cosine distance): Measures inter-story diversity

## Datasets in the Literature

| Dataset | Used In | Task | Human Evals | LLM Evals |
|---------|---------|------|-------------|-----------|
| HANNA | Chhun et al. (2022) | Story generation | 6-dim Likert | Available |
| Chatbot Arena | Zheng et al. (2023) | Open-ended chat | Pairwise votes | Available |
| MT-Bench | Zheng et al. (2023) | Multi-turn QA | Expert pairwise | GPT-4 judge |
| WritingPrompts | Echoes in AI (2024) | Creative stories | Human surprise | LLM continuations |
| Story Writing Benchmark | lars76 | Creative stories | None | Multi-LLM grading |
| ELI5 + BigBench | Koo et al. (2023) | QA | Human ranking | 16 LLM rankings |

---

## Gaps and Opportunities

1. **No study directly tests our hypothesis**: While self-preference, familiarity bias, and diversity reduction are well-documented separately, no paper asks whether LLM-judged "outliers" are more interesting to humans.

2. **Cross-LLM disagreement is understudied**: Most papers measure single-LLM bias. The signal from *inter-LLM disagreement* (what happens when LLMs disagree about text quality) is unexplored.

3. **"Interestingness" is underspecified**: Papers measure quality, coherence, creativity, but "interestingness" as a dimension is rarely directly evaluated. The HANNA dataset's "Surprise" and "Engagement" dimensions are closest.

4. **The outlier hypothesis is novel**: Combining the self-preference bias finding (LLMs prefer familiar text) with the diversity finding (LLMs converge on similar outputs) yields the testable prediction that inter-LLM consensus inversely correlates with human interest.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **HANNA** (primary): 1,056 stories from 10 generators + human with 6-dim Likert ratings. Add multi-LLM evaluations.
2. **Story Writing Benchmark** (complementary): 8,520 stories from 15 LLMs with multi-LLM grading. Add human evaluations on a subsample.
3. **Arena Human Preference 55K**: Largest human preference dataset. Add multi-LLM evaluations on a subsample.
4. **MT-Bench Human Judgments**: Already has both human and GPT-4 evaluations.

### Recommended Baselines
1. **LLM consensus = human preference** (null hypothesis): Average LLM judge score predicts human rating
2. **Single best LLM judge**: GPT-4 alone predicts human preference
3. **Perplexity ranking**: Lower perplexity = higher quality

### Recommended Metrics
1. **LLM disagreement score**: Variance or entropy across multiple LLM judges' ratings
2. **Human interestingness rating**: Direct Likert rating on engagement/surprise/interest
3. **Correlation between LLM disagreement and human interest**: The core test of our hypothesis
4. **Conditional analysis**: For texts where LLMs agree it's bad vs. where they disagree, compare human ratings

### Methodological Considerations
- Use 3-5 diverse LLM judges (e.g., GPT-4, Claude, Llama, Mistral, Gemma) to capture meaningful disagreement
- Control for text quality: distinguish "LLMs disagree because it's genuinely unusual" from "LLMs disagree because it's low quality"
- Use both pairwise and absolute scoring to triangulate
- Apply position and length debiasing (FairEval, OffsetBias methods)
- Focus on creative/open-ended text where "interestingness" is meaningful (not factual QA)
