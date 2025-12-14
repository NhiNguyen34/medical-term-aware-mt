"""
Master script to run the complete Medical MT Pipeline:
Step A: Prompting → Step B: Fine-tuning → Step C: Inference → Step D: Evaluation
"""

import os
import sys
import subprocess
from pathlib import Path

def run_step(step_name: str, script_path: str, args: list = None):
    """Run a pipeline step"""
    print("\n" + "=" * 80)
    print(f"Running {step_name}")
    print("=" * 80)
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {step_name} interrupted by user")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'transformers',
        'torch',
        'sacrebleu',
        'openai'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run the complete pipeline"""
    print("=" * 80)
    print("Medical MT Pipeline")
    print("=" * 80)
    print("\nThis pipeline will execute:")
    print("  Step A: Prompting to create MeddictGem")
    print("  Step B: Fine-tuning on MeddictGem")
    print("  Step C: Inference on MedEV test set")
    print("  Step D: Evaluation with SacreBLEU")
    print("=" * 80)
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ All dependencies installed")
    
    # Configuration
    input_csv = os.getenv('INPUT_CSV', 'MeddictGem01_5000samples(Sheet1).csv')
    test_src = os.getenv('TEST_SRC', 'mede_test.src')
    test_ref = os.getenv('TEST_REF', 'mede_test.ref')
    max_rows = os.getenv('MAX_ROWS', None)  # Set to limit rows for testing
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("\nUsage:")
            print("  python run_pipeline.py [--skip-step STEP] [--max-rows N]")
            print("\nOptions:")
            print("  --skip-step STEP    Skip a specific step (a, b, c, d)")
            print("  --max-rows N        Limit number of rows for Step A (testing)")
            sys.exit(0)
    
    skip_steps = []
    if '--skip-step' in sys.argv:
        idx = sys.argv.index('--skip-step')
        if idx + 1 < len(sys.argv):
            skip_steps = sys.argv[idx + 1].split(',')
    
    if '--max-rows' in sys.argv:
        idx = sys.argv.index('--max-rows')
        if idx + 1 < len(sys.argv):
            max_rows = int(sys.argv[idx + 1])
    
    # Step A: Prompting
    if 'a' not in skip_steps:
        args = []
        if max_rows:
            args = [str(max_rows)]
        if not run_step("Step A: Prompting", "step_a_prompting.py", args):
            print("\nPipeline stopped at Step A")
            sys.exit(1)
    else:
        print("\n⚠ Skipping Step A: Prompting")
    
    # Step B: Fine-tuning
    if 'b' not in skip_steps:
        if not os.path.exists('meddictgem.csv'):
            print("\nERROR: meddictgem.csv not found. Run Step A first.")
            sys.exit(1)
        
        if not run_step("Step B: Fine-tuning", "step_b_finetune.py"):
            print("\nPipeline stopped at Step B")
            sys.exit(1)
    else:
        print("\n⚠ Skipping Step B: Fine-tuning")
    
    # Step C: Inference
    if 'c' not in skip_steps:
        if not os.path.exists('checkpoints/ft_model'):
            print("\nERROR: Fine-tuned model not found. Run Step B first.")
            sys.exit(1)
        
        if not os.path.exists(test_src):
            print(f"\nERROR: Test source file not found: {test_src}")
            sys.exit(1)
        
        if not run_step("Step C: Inference", "step_c_inference.py", [test_src]):
            print("\nPipeline stopped at Step C")
            sys.exit(1)
    else:
        print("\n⚠ Skipping Step C: Inference")
    
    # Step D: Evaluation
    if 'd' not in skip_steps:
        if not os.path.exists('pred_baseline.txt') or not os.path.exists('pred_ft.txt'):
            print("\nERROR: Prediction files not found. Run Step C first.")
            sys.exit(1)
        
        if not os.path.exists(test_ref):
            print(f"\nERROR: Test reference file not found: {test_ref}")
            sys.exit(1)
        
        if not run_step("Step D: Evaluation", "step_d_evaluation.py", [test_ref]):
            print("\nPipeline stopped at Step D")
            sys.exit(1)
    else:
        print("\n⚠ Skipping Step D: Evaluation")
    
    # Summary
    print("\n" + "=" * 80)
    print("Pipeline Completed Successfully!")
    print("=" * 80)
    print("\nOutput files:")
    print("  - meddictgem.csv: Synthetic training data")
    print("  - checkpoints/ft_model/: Fine-tuned model")
    print("  - pred_baseline.txt: Baseline predictions")
    print("  - pred_ft.txt: Fine-tuned predictions")
    print("  - evaluation_results.txt: Evaluation report")
    print("=" * 80)

if __name__ == '__main__':
    main()

