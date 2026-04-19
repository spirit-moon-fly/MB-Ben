~~~markdown
# NLI Evaluation Experiments

This repository contains code and data for MB-Ben under different prompting and reasoning settings, including:

- Vanilla zero-shot / few-shot probability extraction
- Thinking mode vs. No-thinking mode evaluation through API inference

## Project Structure

```bash
.
├── Vanilla_result/
│   ├── zero-shot-orig_prob.py
│   ├── few-shot-orig_prob.py
│   └── ... results
├── eval_thinking_modes/
│   ├── eval_think.py
│   ├── eval-no-think.py
│   └── ... results
~~~

## Scripts

### Vanilla_result

- `zero-shot-orig_prob.py`
  Vanilla zero-shot probability extraction with a local causal LM.
- `few-shot-orig_prob.py`
  Vanilla few-shot probability extraction with manually designed demonstrations.

### eval_thinking_modes

- `eval_think.py`
  API-based evaluation with `reasoning_effort="high"`.
- `eval-no-think.py`
  API-based evaluation with `reasoning_effort="none"`.

These scripts output prediction logs and final accuracy summaries.

## Datasets

The code is written for JSONL datasets containing fields such as:

- `premise`
- `hypothesis`
- `label`

Example datasets used in the scripts include:

- `mnli.jsonl`
- `hans.jsonl`
- `5bias.jsonl`
- `3bias.jsonl`

## Requirements

Install the main dependencies:

```bash
pip install torch transformers accelerate aiohttp tqdm
```

## Usage

Run the corresponding scripts inside each folder.

### Vanilla

```bash
cd Vanilla_result
python zero-shot-orig_prob.py
python few-shot-orig_prob.py
```

### Thinking modes

```bash
cd eval_thinking_modes
python eval_think.py
python eval-no-think.py
```

## Notes

- Local inference scripts use a local Llama-style model path and compute probabilities over the three NLI labels from final-token logits.
- API evaluation scripts require manual configuration of `API_KEY`, `BASE_URL`, and `MODEL_NAME`.
- Few-shot scripts include demonstration examples before the test instance.