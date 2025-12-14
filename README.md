# Medical Term-Aware Machine Translation Pipeline

---

## Overview

This repository implements a complete workflow for improving medical machine translation quality by enforcing **explicit medical term mappings** during training.

The pipeline consists of four stages:

1. **Prompting**  
   Generate term-controlled synthetic parallel data (*MeddictGem*) using a fixed medical translation prompt.
2. **Fine-tuning**  
   Perform lightweight domain adaptation on a pretrained MT model.
3. **Inference**  
   Translate an independent medical test set using baseline and fine-tuned models.
4. **Evaluation**  
   Compare translation quality using **SacreBLEU**.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Configuration

### Environment Variables

```bash
# LLM API (Step A: Prompting)
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o-mini"
export LLM_PROVIDER="openai"

# MT model (Step B & C)
export BASE_MODEL="vinai/vinai-translate-en2vi-v2"
export MODEL_NAME="vinai/vinai-translate-en2vi-v2"

# Evaluation data (Step C & D)
export TEST_SRC="mede_test.src"
export TEST_REF="mede_test.ref"
```

---

## Usage

### Run the full pipeline

```bash
python run_pipeline.py
```

### Run individual stages

---

### Step A — Prompting (Synthetic Data Generation)

```bash
# Process full dataset
python step_a_prompting.py

# Process first N rows (for testing)
python step_a_prompting.py 10
```

**Input**  
CSV file with columns:
- `index` (optional)
- `English` (English medical term)
- `Vietnamese` (Vietnamese medical term)
- `POS` (optional)
- `English Sentence` (must contain English term)
- `Vietnamese Sentence` (used as Vietnamese example context)

**Output**
- `meddictgem.csv` — synthetic EN–VI training data  
- `failed_rows.csv` — invalid or failed samples  
- `prompt_logs.jsonl` — raw prompt/response logs  

---

### Step B — Fine-tuning

```bash
python step_b_finetune.py meddictgem.csv
```

**Output**
- `checkpoints/ft_model/` — fine-tuned model checkpoint  
- `checkpoints/train_config.json` — training configuration  
- `data/train/train.src`, `data/train/train.tgt` — training files  

---

### Step C — Inference

```bash
python step_c_inference.py mede_test.src
```

**Output**
- `pred_baseline.txt` — baseline model translations  
- `pred_ft.txt` — fine-tuned model translations  

---

### Step D — Evaluation

```bash
python step_d_evaluation.py mede_test.ref
```

**Output**
- Console BLEU scores  
- `evaluation_results.txt` — detailed evaluation report  

---

### Skip pipeline stages

```bash
# Skip fine-tuning
python run_pipeline.py --skip-step b

# Skip multiple stages
python run_pipeline.py --skip-step b,c

# Limit rows for quick testing
python run_pipeline.py --max-rows 10
```

## Data Validation Rules

- `English Sentence` **must contain** the English term (case-insensitive).
- Generated Vietnamese translations **must contain** the mapped Vietnamese term.
- Invalid samples are skipped and logged.

---

## Evaluation

- **Metric:** SacreBLEU  
- **Test set:** MedEV test split (independent, non-synthetic)

---

## Supported Models

### Recommended
```bash
vinai/vinai-translate-en2vi-v2
```

### Alternatives
- `facebook/mbart-large-50-many-to-many-mmt`
- `Helsinki-NLP/opus-mt-en-vi`

---

## Notes

- Each pipeline stage is checkpointed and can be resumed independently.
- Prompting may be slow due to LLM rate limits.
- Fine-tuning requires GPU for practical runtimes.
- This pipeline focuses on **domain adaptation**, not training MT models from scratch.
