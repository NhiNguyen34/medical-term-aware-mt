"""
Step D: Evaluation with SacreBLEU
- Compute SacreBLEU for baseline and fine-tuned models
- Report baseline BLEU, fine-tuned BLEU, and delta BLEU
- Optionally compute term accuracy if annotations available
"""

import os
import sys
import subprocess
from pathlib import Path

# File paths
TEST_REF = os.getenv('TEST_REF', 'mede_test.ref')
PRED_BASELINE = 'pred_baseline.txt'
PRED_FT = 'pred_ft.txt'

def run_sacrebleu(ref_file: str, pred_file: str) -> dict:
    """Run sacrebleu and return results"""
    try:
        result = subprocess.run(
            ['sacrebleu', ref_file, '-i', pred_file, '--format', 'text'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output
        output = result.stdout.strip()
        lines = output.split('\n')
        
        # Extract BLEU score (first line usually contains the score)
        bleu_score = None
        for line in lines:
            if 'BLEU' in line or 'score' in line.lower():
                # Try to extract number
                import re
                numbers = re.findall(r'\d+\.\d+', line)
                if numbers:
                    bleu_score = float(numbers[0])
                    break
        
        # If not found, try to parse the standard sacrebleu output
        if bleu_score is None:
            # Standard format: "BLEU = XX.XX, ..."
            import re
            match = re.search(r'BLEU\s*=\s*(\d+\.\d+)', output)
            if match:
                bleu_score = float(match.group(1))
        
        return {
            'bleu': bleu_score,
            'raw_output': output
        }
    
    except subprocess.CalledProcessError as e:
        print(f"ERROR running sacrebleu: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("ERROR: sacrebleu not found. Install with: pip install sacrebleu")
        return None

def compute_term_accuracy(ref_file: str, pred_file: str, term_file: str = None):
    """Compute term accuracy if term annotations are available"""
    if not term_file or not os.path.exists(term_file):
        return None
    
    # This would require term annotations in MedEV test set
    # Implementation depends on annotation format
    print("Term accuracy computation not implemented (requires term annotations)")
    return None

def print_evaluation_report(baseline_result: dict, ft_result: dict):
    """Print evaluation report"""
    print("\n" + "=" * 60)
    print("Evaluation Report")
    print("=" * 60)
    
    if baseline_result and 'bleu' in baseline_result:
        baseline_bleu = baseline_result['bleu']
        print(f"\nBaseline BLEU: {baseline_bleu:.2f}")
    else:
        print("\nBaseline BLEU: ERROR - Could not compute")
        baseline_bleu = None
    
    if ft_result and 'bleu' in ft_result:
        ft_bleu = ft_result['bleu']
        print(f"Fine-tuned BLEU: {ft_bleu:.2f}")
    else:
        print("Fine-tuned BLEU: ERROR - Could not compute")
        ft_bleu = None
    
    if baseline_bleu is not None and ft_bleu is not None:
        delta_bleu = ft_bleu - baseline_bleu
        print(f"Delta BLEU: {delta_bleu:+.2f}")
        
        if delta_bleu > 0:
            print(f"\n✓ Fine-tuned model improved by {delta_bleu:.2f} BLEU points")
        elif delta_bleu < 0:
            print(f"\n✗ Fine-tuned model decreased by {abs(delta_bleu):.2f} BLEU points")
        else:
            print(f"\n- No change in BLEU score")
    
    print("\n" + "=" * 60)
    
    # Save detailed results
    save_detailed_results(baseline_result, ft_result)

def save_detailed_results(baseline_result: dict, ft_result: dict):
    """Save detailed evaluation results to file"""
    results_file = 'evaluation_results.txt'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Medical MT Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Baseline Model:\n")
        if baseline_result:
            f.write(f"  BLEU: {baseline_result.get('bleu', 'N/A')}\n")
            f.write(f"  Raw output:\n{baseline_result.get('raw_output', '')}\n")
        else:
            f.write("  ERROR: Could not compute\n")
        
        f.write("\nFine-tuned Model:\n")
        if ft_result:
            f.write(f"  BLEU: {ft_result.get('bleu', 'N/A')}\n")
            f.write(f"  Raw output:\n{ft_result.get('raw_output', '')}\n")
        else:
            f.write("  ERROR: Could not compute\n")
        
        if baseline_result and ft_result:
            baseline_bleu = baseline_result.get('bleu')
            ft_bleu = ft_result.get('bleu')
            if baseline_bleu is not None and ft_bleu is not None:
                delta = ft_bleu - baseline_bleu
                f.write(f"\nDelta BLEU: {delta:+.2f}\n")
    
    print(f"\nDetailed results saved to {results_file}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        TEST_REF = sys.argv[1]
    
    print("=" * 60)
    print("Step D: Evaluation with SacreBLEU")
    print("=" * 60)
    print(f"Reference file: {TEST_REF}")
    print(f"Baseline predictions: {PRED_BASELINE}")
    print(f"Fine-tuned predictions: {PRED_FT}")
    print("=" * 60)
    
    # Check files exist
    if not os.path.exists(TEST_REF):
        print(f"ERROR: Reference file not found: {TEST_REF}")
        sys.exit(1)
    
    if not os.path.exists(PRED_BASELINE):
        print(f"ERROR: Baseline predictions not found: {PRED_BASELINE}")
        print("Please run Step C (inference) first")
        sys.exit(1)
    
    if not os.path.exists(PRED_FT):
        print(f"ERROR: Fine-tuned predictions not found: {PRED_FT}")
        print("Please run Step C (inference) first")
        sys.exit(1)
    
    # Compute BLEU scores
    print("\nComputing baseline BLEU...")
    baseline_result = run_sacrebleu(TEST_REF, PRED_BASELINE)
    
    print("\nComputing fine-tuned BLEU...")
    ft_result = run_sacrebleu(TEST_REF, PRED_FT)
    
    # Print report
    print_evaluation_report(baseline_result, ft_result)

