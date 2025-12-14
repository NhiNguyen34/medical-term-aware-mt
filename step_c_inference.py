"""
Step C: Inference on MedEV test set
- Load baseline model (pretrained)
- Load fine-tuned model
- Run inference on mede_test.src
- Save predictions: pred_baseline.txt, pred_ft.txt
"""

import os
import sys
from pathlib import Path

# Model configuration
BASE_MODEL = os.getenv('MODEL_NAME', 'vinai/vinai-translate-en2vi-v2')
FT_MODEL = os.getenv('FT_MODEL', 'checkpoints/ft_model')
TEST_SRC = os.getenv('TEST_SRC', 'mede_test.src')
PRED_BASELINE = 'pred_baseline.txt'
PRED_FT = 'pred_ft.txt'

def inference_with_transformers(model_path: str, src_file: str, output_file: str):
    """Run inference using HuggingFace Transformers"""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError:
        print("ERROR: transformers not installed")
        print("Install with: pip install transformers torch")
        return False
    
    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    else:
        print("Using CPU")
    
    # Load source sentences
    print(f"Loading source sentences from {src_file}")
    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Translating {len(src_sentences)} sentences...")
    
    # Translate in batches
    batch_size = 8
    translations = []
    
    for i in range(0, len(src_sentences), batch_size):
        batch = src_sentences[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode
        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(batch_translations)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{len(src_sentences)} sentences")
    
    # Save predictions
    print(f"\nSaving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for trans in translations:
            f.write(trans + '\n')
    
    print(f"✓ Inference completed: {len(translations)} translations saved")
    return True

def inference_with_fairseq(model_path: str, src_file: str, output_file: str):
    """Run inference using fairseq (alternative method)"""
    print("Fairseq inference not implemented in this script")
    print("Please use fairseq-interactive command directly")
    return False

def run_baseline_inference(src_file: str, output_file: str):
    """Run inference with baseline (pretrained) model"""
    print("\n" + "=" * 60)
    print("Baseline Model Inference")
    print("=" * 60)
    return inference_with_transformers(BASE_MODEL, src_file, output_file)

def run_ft_inference(src_file: str, output_file: str):
    """Run inference with fine-tuned model"""
    print("\n" + "=" * 60)
    print("Fine-tuned Model Inference")
    print("=" * 60)
    
    if not os.path.exists(FT_MODEL):
        print(f"ERROR: Fine-tuned model not found at {FT_MODEL}")
        print("Please run Step B (fine-tuning) first")
        return False
    
    return inference_with_transformers(FT_MODEL, src_file, output_file)

if __name__ == '__main__':
    framework = os.getenv('INFERENCE_FRAMEWORK', 'transformers')
    
    if len(sys.argv) > 1:
        TEST_SRC = sys.argv[1]
    
    print("=" * 60)
    print("Step C: Inference on MedEV Test Set")
    print("=" * 60)
    print(f"Test source file: {TEST_SRC}")
    print(f"Baseline model: {BASE_MODEL}")
    print(f"Fine-tuned model: {FT_MODEL}")
    print(f"Output files: {PRED_BASELINE}, {PRED_FT}")
    print("=" * 60)
    
    if not os.path.exists(TEST_SRC):
        print(f"ERROR: Test source file not found: {TEST_SRC}")
        sys.exit(1)
    
    # Run baseline inference
    success_baseline = run_baseline_inference(TEST_SRC, PRED_BASELINE)
    
    # Run fine-tuned inference
    success_ft = run_ft_inference(TEST_SRC, PRED_FT)
    
    if success_baseline and success_ft:
        print("\n" + "=" * 60)
        print("✓ All inference completed successfully!")
        print(f"  Baseline predictions: {PRED_BASELINE}")
        print(f"  Fine-tuned predictions: {PRED_FT}")
        print("=" * 60)
    else:
        print("\n✗ Inference failed!")
        sys.exit(1)

