# Cloned Repositories

## 1. CoBBLEr (Minnesota NLP)
- **URL**: https://github.com/minnesotanlp/cobbler
- **Purpose**: Benchmark for measuring 6 cognitive biases in LLM evaluators (egocentric, order, salience, bandwagon, compassion fade, attentional)
- **Location**: code/cobbler/
- **Key files**: `evaluations/evaluate.py` (evaluation scripts), `n15_responses/` (model responses)
- **How to use**: Framework for measuring inter-LLM evaluation bias. Can adapt to measure disagreement patterns across LLM judges.

## 2. Confederacy of Models
- **URL**: https://github.com/komoku/confederacy-of-models
- **Purpose**: 65 stories from 13 LLMs + human, with expert human ratings on fluency, coherence, originality, humor, style
- **Location**: code/confederacy-of-models/
- **Key files**: Excel files with ratings, story texts
- **How to use**: Small but rich dataset of creative writing with human evaluation. Can add multi-LLM evaluations.

## 3. Lechmazur Writing Benchmark
- **URL**: https://github.com/lechmazur/writing
- **Purpose**: Stories from many LLMs scored by 7 independent LLM graders on 18-question rubric
- **Location**: code/lechmazur-writing/
- **Key files**: Stories and evaluations in structured format
- **How to use**: Multi-LLM judge scores on creative writing. Can compute inter-judge disagreement and correlate with human ratings.

## 4. Self-Preference (Chen et al.)
- **URL**: https://github.com/zhiyuanc2001/self-preference
- **Purpose**: Measures self-preference bias across helpfulness, truthfulness, and translation tasks
- **Location**: code/self-preference/
- **Key files**: `gen_preference.py`, `gen_answering.py`, `average_preference.py`
- **How to use**: Scripts for running self-preference experiments with multiple LLMs.

## Recommended Additional Repos (not cloned, can be installed)
- **lm-sys/FastChat**: MT-Bench judge framework (pip install fschat)
- **prometheus-eval/prometheus-eval**: Open-source evaluator LLMs (pip install prometheus-eval)
- **i-Eval/FairEval**: Position bias calibration
- **allenai/WildBench**: Real-world evaluation benchmark
