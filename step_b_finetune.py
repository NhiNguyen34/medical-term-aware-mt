"""
Step B: Fine-tuning Medical MT Model
- Load MeddictGem dataset
- Prepare training data (src: en_sentence, tgt: vi_sentence_gen)
- Fine-tune pretrained model (VinAI Translate or equivalent)
- Save checkpoint
"""

import csv
import os
import json
from typing import List, Tuple
from pathlib import Path

# Model configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'vinai/vinai-translate-en2vi-v2')  # VinAI Translate
BASE_MODEL = os.getenv('BASE_MODEL', MODEL_NAME)
CHECKPOINT_DIR = 'checkpoints/ft_model'
TRAIN_CONFIG_FILE = 'checkpoints/train_config.json'

def load_meddictgem(meddictgem_file: str) -> Tuple[List[str], List[str]]:
    """Load MeddictGem dataset and return (src_sentences, tgt_sentences)"""
    src_sentences = []
    tgt_sentences = []
    
    with open(meddictgem_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            en_sentence = row.get('en_sentence', '').strip()
            vi_sentence = row.get('vi_sentence_gen', '').strip()
            
            if en_sentence and vi_sentence:
                src_sentences.append(en_sentence)
                tgt_sentences.append(vi_sentence)
    
    print(f"Loaded {len(src_sentences)} sentence pairs from {meddictgem_file}")
    return src_sentences, tgt_sentences

def save_training_data(src_sentences: List[str], tgt_sentences: List[str], 
                      output_dir: str = 'data/train'):
    """Save training data as separate source and target files"""
    os.makedirs(output_dir, exist_ok=True)
    
    src_file = os.path.join(output_dir, 'train.src')
    tgt_file = os.path.join(output_dir, 'train.tgt')
    
    with open(src_file, 'w', encoding='utf-8') as f:
        for sent in src_sentences:
            f.write(sent + '\n')
    
    with open(tgt_file, 'w', encoding='utf-8') as f:
        for sent in tgt_sentences:
            f.write(sent + '\n')
    
    print(f"Saved training data to {src_file} and {tgt_file}")
    return src_file, tgt_file

def fine_tune_with_transformers(src_file: str, tgt_file: str, 
                                base_model: str, output_dir: str):
    """Fine-tune using HuggingFace Transformers"""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSeq2SeqLM,
            Seq2SeqTrainingArguments, Seq2SeqTrainer,
            DataCollatorForSeq2Seq
        )
        from datasets import Dataset
        import torch
    except ImportError:
        print("ERROR: transformers and datasets not installed")
        print("Install with: pip install transformers datasets torch")
        return False
    
    print(f"\nLoading model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    
    # Load training data
    print("Loading training data...")
    src_sentences = []
    tgt_sentences = []
    
    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f if line.strip()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_sentences = [line.strip() for line in f if line.strip()]
    
    if len(src_sentences) != len(tgt_sentences):
        print(f"ERROR: Mismatch in data lengths: {len(src_sentences)} vs {len(tgt_sentences)}")
        return False
    
    print(f"Training on {len(src_sentences)} sentence pairs")
    
    # Tokenize
    def tokenize_function(examples):
        model_inputs = tokenizer(examples['src'], max_length=512, truncation=True, padding='max_length')
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['tgt'], max_length=512, truncation=True, padding='max_length')
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    dataset = Dataset.from_dict({
        'src': src_sentences,
        'tgt': tgt_sentences
    })
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("\nStarting fine-tuning...")
    trainer.train()
    
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return True

def fine_tune_with_fairseq(src_file: str, tgt_file: str, 
                          base_model: str, output_dir: str):
    """Fine-tune using fairseq (alternative method)"""
    print("Fairseq fine-tuning not implemented in this script")
    print("Please use fairseq-train command directly or implement separately")
    return False

def save_training_config(base_model: str, checkpoint_dir: str, 
                        num_samples: int, seed: int = 42):
    """Save training configuration for reproducibility"""
    config = {
        'base_model': base_model,
        'checkpoint_dir': checkpoint_dir,
        'num_training_samples': num_samples,
        'seed': seed,
        'framework': 'transformers'
    }
    
    os.makedirs(os.path.dirname(TRAIN_CONFIG_FILE), exist_ok=True)
    with open(TRAIN_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Saved training config to {TRAIN_CONFIG_FILE}")

if __name__ == '__main__':
    import sys
    import random
    import numpy as np
    
    meddictgem_file = 'meddictgem.csv'
    framework = os.getenv('FINE_TUNE_FRAMEWORK', 'transformers')  # 'transformers' or 'fairseq'
    
    if len(sys.argv) > 1:
        meddictgem_file = sys.argv[1]
    
    print("=" * 60)
    print("Step B: Fine-tuning Medical MT Model")
    print("=" * 60)
    print(f"MeddictGem file: {meddictgem_file}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Framework: {framework}")
    print("=" * 60)
    
    # Load data
    src_sentences, tgt_sentences = load_meddictgem(meddictgem_file)
    
    if not src_sentences:
        print("ERROR: No training data found!")
        sys.exit(1)
    
    # Save training data
    src_file, tgt_file = save_training_data(src_sentences, tgt_sentences)
    
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(seed)
    
    # Fine-tune
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if framework == 'transformers':
        success = fine_tune_with_transformers(
            src_file, tgt_file, BASE_MODEL, CHECKPOINT_DIR
        )
    elif framework == 'fairseq':
        success = fine_tune_with_fairseq(
            src_file, tgt_file, BASE_MODEL, CHECKPOINT_DIR
        )
    else:
        print(f"ERROR: Unknown framework '{framework}'")
        success = False
    
    if success:
        save_training_config(BASE_MODEL, CHECKPOINT_DIR, len(src_sentences), seed)
        print("\n✓ Fine-tuning completed successfully!")
    else:
        print("\n✗ Fine-tuning failed!")
        sys.exit(1)

