# Downloaded Datasets

This directory contains datasets for the "Artificial Outsiders" research project.
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: HANNA (Human-ANnotated NArratives)

### Overview
- **Source**: HuggingFace `llm-aes/hanna`
- **Size**: 3,168 annotated story evaluations (1,056 stories x 3 annotators)
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Story generation evaluation
- **Features**: Story_ID, Prompt, Human, Story, Model, Relevance, Coherence, Empathy, Surprise, Engagement, Complexity
- **License**: Research use

### Why Selected
Creative stories from 10 different generators (including human), each rated by humans on 6 quality dimensions including Surprise and Engagement---the closest existing proxies for "interestingness." This is the primary dataset for our hypothesis.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("llm-aes/hanna")
dataset.save_to_disk("datasets/hanna")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/hanna")
```

---

## Dataset 2: MT-Bench Human Judgments

### Overview
- **Source**: HuggingFace `lmsys/mt_bench_human_judgments`
- **Size**: 3,355 human judgments + 2,400 GPT-4 pair judgments
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Multi-turn question answering evaluation
- **Splits**: human (3,355), gpt4_pair (2,400)
- **Features**: question_id, model_a, model_b, winner, judge, conversation_a, conversation_b, turn

### Why Selected
One of the few datasets with BOTH human expert and LLM (GPT-4) evaluations on the same response pairs. Enables direct comparison of human vs LLM preferences.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("lmsys/mt_bench_human_judgments")
dataset.save_to_disk("datasets/mt_bench_human_judgments")
```

---

## Dataset 3: Story Writing Benchmark

### Overview
- **Source**: HuggingFace `lars1234/story_writing_benchmark`
- **Size**: 8,520 stories from 15 different LLMs
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Creative story generation with multi-LLM evaluation
- **Features**: prompt_id, prompt, story_text, model_name, 15 quality scores (q1-q15), overall_score
- **Models**: 15 LLMs including GPT-4, Claude, Llama, Mistral variants

### Why Selected
Largest creative writing dataset with stories from multiple LLMs, each scored by multiple LLM graders on 15 quality dimensions. Ideal for computing inter-LLM disagreement. Needs human evaluation to complete.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("lars1234/story_writing_benchmark")
dataset.save_to_disk("datasets/story_writing_benchmark")
```

---

## Dataset 4: Arena Human Preference 55K

### Overview
- **Source**: HuggingFace `lmarena-ai/arena-human-preference-55k`
- **Size**: 57,477 pairwise battles
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Open-ended LLM comparison
- **Features**: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie
- **Models**: 64+ unique LLMs
- **License**: Apache 2.0

### Why Selected
Largest public human preference dataset with responses from 64+ models. Can filter for creative/open-ended prompts and add multi-LLM evaluations.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("lmarena-ai/arena-human-preference-55k")
dataset.save_to_disk("datasets/arena_human_preference")
```

---

## Notes

- All datasets are downloaded locally for experiment use
- Large data files are excluded from git via `.gitignore`
- Sample files for documentation are in `*_samples/` subdirectories
- The experiment runner should use `load_from_disk()` for locally saved datasets
