"""
Step A: Prompting to create Synthetic Data 1 (MeddictGem)
- Read CSV input
- Validate data
- Create prompts using template (ii)
- Call LLM API
- Parse outputs
- Generate MeddictGem dataset
"""

import csv
import json
import re
import os
from typing import Dict, List, Tuple, Optional
import openai
from openai import OpenAI

# LLM Configuration
# Support OpenAI API or local LLM
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')  # 'openai' or 'local'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# Prompt template (ii) - MUST USE VERBATIM
PROMPT_TEMPLATE = """You are an experienced doctor and a professional translator specializing in medical and healthcare texts. Translate the given <English sentence> (A) to Vietnamese in a medical context where the "<EN term>" has a "<VI term>" meaning in Vietnamese and the "<VI term>" also used in several contexts in these Vietnamese sentence examples below:
<Example VI sentences> (B)

Input format
<index> <English term> <Vietnamese term> <English sentence> (A) <Example VI sentences> (B)

Desired output format
<index> | <English sentence containing the English term searched from Internet> | <Vietnamese sentence containing the Vietnamese term is the translation of (A)>"""

def clean_sentence(sentence: str) -> str:
    """Strip whitespace and quotes from sentences"""
    if not sentence:
        return ""
    sentence = sentence.strip()
    # Remove surrounding quotes if present
    if (sentence.startswith('"') and sentence.endswith('"')) or \
       (sentence.startswith("'") and sentence.endswith("'")):
        sentence = sentence[1:-1].strip()
    return sentence

def validate_row(row: Dict, index: int) -> Tuple[bool, str]:
    """Validate a CSV row"""
    en_term = row.get('English', '').strip()
    vi_term = row.get('Vietnamese', '').strip()
    en_sentence = clean_sentence(row.get('English Sentence', ''))
    vi_sentence = clean_sentence(row.get('Vietnamese Sentence', ''))
    
    if not en_term:
        return False, "Missing English term"
    if not vi_term:
        return False, "Missing Vietnamese term"
    if not en_sentence:
        return False, "Missing English Sentence"
    if not vi_sentence:
        return False, "Missing Vietnamese Sentence"
    
    # Validate: English Sentence contains English term (case-insensitive)
    if en_term.lower() not in en_sentence.lower():
        return False, f"English Sentence does not contain English term '{en_term}'"
    
    return True, ""

def prepare_vi_examples(vi_sentence: str) -> str:
    """Prepare Vietnamese example sentences (B)
    If only one sentence, duplicate it or split if possible
    Minimum 2 sentences preferred
    """
    vi_sentence = clean_sentence(vi_sentence)
    if not vi_sentence:
        return ""
    
    # Try to split by sentence delimiters
    sentences = re.split(r'[.!?]\s+', vi_sentence)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) >= 2:
        # Use first 2 sentences
        return "\n".join(sentences[:2])
    elif len(sentences) == 1:
        # Duplicate the sentence
        return f"{sentences[0]}\n{sentences[0]}"
    else:
        # Fallback: use original sentence twice
        return f"{vi_sentence}\n{vi_sentence}"

def create_prompt_input(index: int, en_term: str, vi_term: str, 
                       en_sentence: str, vi_examples: str) -> str:
    """Create prompt input string exactly formatted as required"""
    return f"{index} {en_term} {vi_term} {en_sentence} {vi_examples}"

def call_llm(prompt_input: str, vi_term: str, retry_count: int = 0) -> Tuple[Optional[str], str]:
    """Call LLM API with the prompt"""
    full_prompt = f"{PROMPT_TEMPLATE}\n\n{prompt_input}"
    
    try:
        if LLM_PROVIDER == 'openai':
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a medical translator. Follow the format exactly."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            output = response.choices[0].message.content.strip()
        else:
            # For local LLM, you can add other providers here
            raise ValueError(f"LLM provider '{LLM_PROVIDER}' not implemented")
        
        return output, "success"
    
    except Exception as e:
        error_msg = f"LLM API error: {str(e)}"
        return None, error_msg

def parse_llm_output(output: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Parse LLM output into: index, English sentence, Vietnamese sentence"""
    try:
        # Expected format: <index> | <English sentence> | <Vietnamese sentence>
        parts = output.split('|')
        if len(parts) < 3:
            return None, None, None
        
        index = int(parts[0].strip())
        en_sentence = parts[1].strip()
        vi_sentence = parts[2].strip()
        
        return index, en_sentence, vi_sentence
    
    except Exception as e:
        return None, None, None

def validate_output(vi_sentence: str, vi_term: str) -> bool:
    """Check if generated Vietnamese sentence contains VI term"""
    if not vi_sentence or not vi_term:
        return False
    return vi_term in vi_sentence

def process_csv(input_file: str, output_file: str, failed_file: str, 
                prompt_logs_file: str, max_rows: Optional[int] = None):
    """Process CSV and generate MeddictGem dataset"""
    
    results = []
    failed_rows = []
    prompt_logs = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if max_rows:
            rows = rows[:max_rows]
        
        total = len(rows)
        auto_index = 1
        
        for row_idx, row in enumerate(rows, start=1):
            # Auto-generate index if missing
            index = row.get('index', '').strip()
            if not index:
                try:
                    index = int(index)
                except:
                    index = auto_index
                    auto_index += 1
            else:
                try:
                    index = int(index)
                except:
                    index = auto_index
                    auto_index += 1
            
            # Validate row
            is_valid, error_msg = validate_row(row, index)
            if not is_valid:
                failed_rows.append({
                    'index': index,
                    'en_term': row.get('English', ''),
                    'vi_term': row.get('Vietnamese', ''),
                    'reason': error_msg
                })
                print(f"[{row_idx}/{total}] Row {index}: FAILED - {error_msg}")
                continue
            
            en_term = row.get('English', '').strip()
            vi_term = row.get('Vietnamese', '').strip()
            en_sentence = clean_sentence(row.get('English Sentence', ''))
            vi_sentence = clean_sentence(row.get('Vietnamese Sentence', ''))
            
            print(f"\n[{row_idx}/{total}] Processing index {index}: {en_term} / {vi_term}")
            
            # Prepare Vietnamese examples
            vi_examples = prepare_vi_examples(vi_sentence)
            
            # Create prompt input
            prompt_input = create_prompt_input(index, en_term, vi_term, en_sentence, vi_examples)
            
            # Call LLM (with retry logic)
            vi_sentence_gen = None
            status = "failed"
            retry_count = 0
            max_retries = 2
            
            while retry_count <= max_retries:
                if retry_count > 0:
                    # Add instruction for retry
                    prompt_input_retry = f"{prompt_input}\n\nEnsure the Vietnamese translation contains the term: {vi_term}."
                    output, error = call_llm(prompt_input_retry, vi_term, retry_count)
                else:
                    output, error = call_llm(prompt_input, vi_term, retry_count)
                
                if output:
                    parsed_index, en_sent_echoed, vi_sent_gen = parse_llm_output(output)
                    
                    if vi_sent_gen and validate_output(vi_sent_gen, vi_term):
                        vi_sentence_gen = vi_sent_gen
                        status = "success"
                        break
                    elif retry_count < max_retries:
                        retry_count += 1
                        print(f"  Retry {retry_count}: Output missing VI term")
                        continue
                    else:
                        status = f"failed_validation: {error}"
                        break
                else:
                    if retry_count < max_retries:
                        retry_count += 1
                        print(f"  Retry {retry_count}: {error}")
                        continue
                    else:
                        status = f"failed_api: {error}"
                        break
            
            # Log prompt and response
            prompt_logs.append({
                'index': index,
                'prompt_input': prompt_input,
                'raw_response': output if output else "",
                'status': status
            })
            
            if vi_sentence_gen:
                results.append({
                    'index': index,
                    'en_term': en_term,
                    'vi_term': vi_term,
                    'en_sentence': en_sentence,
                    'vi_sentence_gen': vi_sentence_gen
                })
                print(f"  ✓ Success!")
            else:
                failed_rows.append({
                    'index': index,
                    'en_term': en_term,
                    'vi_term': vi_term,
                    'reason': status
                })
                print(f"  ✗ Failed: {status}")
            
            # Save progress periodically
            if row_idx % 10 == 0:
                save_results(results, output_file)
                save_failed(failed_rows, failed_file)
                save_prompt_logs(prompt_logs, prompt_logs_file)
                print(f"\n  Progress saved at {row_idx} rows")
    
    # Final save
    save_results(results, output_file)
    save_failed(failed_rows, failed_file)
    save_prompt_logs(prompt_logs, prompt_logs_file)
    
    print(f"\n\nCompleted!")
    print(f"Success: {len(results)} rows")
    print(f"Failed: {len(failed_rows)} rows")

def save_results(results: List[Dict], output_file: str):
    """Save results to meddictgem.csv"""
    if not results:
        return
    
    fieldnames = ['index', 'en_term', 'vi_term', 'en_sentence', 'vi_sentence_gen']
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def save_failed(failed_rows: List[Dict], failed_file: str):
    """Save failed rows to failed_rows.csv"""
    if not failed_rows:
        return
    
    fieldnames = ['index', 'en_term', 'vi_term', 'reason']
    
    with open(failed_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)

def save_prompt_logs(prompt_logs: List[Dict], prompt_logs_file: str):
    """Save prompt logs to prompt_logs.jsonl"""
    with open(prompt_logs_file, 'w', encoding='utf-8') as f:
        for log in prompt_logs:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    import sys
    
    input_file = 'MeddictGem01_5000samples(Sheet1).csv'
    output_file = 'meddictgem.csv'
    failed_file = 'failed_rows.csv'
    prompt_logs_file = 'prompt_logs.jsonl'
    
    # For testing, limit rows
    max_rows = None
    if len(sys.argv) > 1:
        try:
            max_rows = int(sys.argv[1])
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}. Using default.")
    
    print("=" * 60)
    print("Step A: Prompting to create MeddictGem")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Failed rows file: {failed_file}")
    print(f"Prompt logs file: {prompt_logs_file}")
    if max_rows:
        print(f"Processing first {max_rows} rows (testing mode)")
    else:
        print("Processing ALL rows")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print("=" * 60)
    
    if LLM_PROVIDER == 'openai' and not OPENAI_API_KEY:
        print("\nERROR: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    process_csv(input_file, output_file, failed_file, prompt_logs_file, max_rows)

