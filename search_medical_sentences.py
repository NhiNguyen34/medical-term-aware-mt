"""
Script to search web for medical sentences containing English and Vietnamese terms.
For each (English term, Vietnamese term) pair:
- Find 1 English sentence containing the English term
- Find 2 Vietnamese sentences containing the Vietnamese term
"""

import csv
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
import json
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("Warning: duckduckgo-search not installed. Install with: pip install duckduckgo-search")

# Medical websites to prioritize
EN_MEDICAL_SITES = [
    'wikipedia.org',
    'pubmed.ncbi.nlm.nih.gov',
    'mayoclinic.org',
    'nhs.uk',
    'webmd.com',
    'medlineplus.gov',
    'healthline.com'
]

VI_MEDICAL_SITES = [
    'vi.wikipedia.org',
    'vinmec.com',
    'medlatec.vn',
    'youmed.vn',
    'suckhoedoisong.vn',
    'vnexpress.net',
    'tuoitre.vn'
]

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text)
    return text.strip()

def extract_sentences(text, term, min_length=20, is_vietnamese=False):
    """Extract sentences containing the term"""
    if not text or not term:
        return []
    
    # Split by sentence delimiters (handle both English and Vietnamese)
    # Vietnamese might not have space after punctuation
    sentences = re.split(r'[.!?]\s*', text)
    results = []
    
    for sentence in sentences:
        sentence = clean_text(sentence)
        if not sentence:
            continue
            
        # Check if sentence contains the term
        if len(term) > 1:  # Only check if term is meaningful
            # For Vietnamese, do case-sensitive search (Vietnamese has diacritics)
            # For English, case-insensitive
            if is_vietnamese:
                contains_term = term in sentence
            else:
                contains_term = term.lower() in sentence.lower()
            
            if contains_term:
                # Filter: not too short, not just the term itself
                if len(sentence) >= min_length and len(sentence) > len(term) + 10:
                    # Check if it's not just a menu/title
                    # For Vietnamese, check if it's not all caps (rare but possible)
                    if not (sentence.isupper() and len(sentence) < 50):
                        # Avoid very short sentences that are likely titles
                        if len(sentence.split()) >= 5:  # At least 5 words
                            results.append(sentence)
    
    return results

def search_duckduckgo(query, max_results=5):
    """Search using DuckDuckGo"""
    results = []
    
    if DDGS_AVAILABLE:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': clean_text(r.get('title', '')),
                        'snippet': clean_text(r.get('body', '')),
                        'url': r.get('href', '')
                    })
            return results
        except Exception as e:
            print(f"Error with DDGS library: {e}")
    
    # Fallback to HTML scraping
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results
        for result in soup.find_all('a', class_='result__a', limit=max_results):
            link = result.get('href', '')
            title = clean_text(result.get_text())
            
            # Try to get snippet from next sibling
            snippet_elem = result.find_next('a', class_='result__snippet')
            snippet = clean_text(snippet_elem.get_text()) if snippet_elem else ""
            
            if link and (title or snippet):
                results.append({
                    'title': title,
                    'snippet': snippet,
                    'url': link
                })
    except Exception as e:
        print(f"Error searching DuckDuckGo: {e}")
    
    return results

def fetch_page_content(url):
    """Fetch and extract text content from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Get text from main content areas
        text = ""
        for tag in soup.find_all(['p', 'div', 'article', 'section']):
            tag_text = tag.get_text()
            if len(tag_text) > 50:  # Only meaningful paragraphs
                text += tag_text + " "
        
        return clean_text(text)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def search_english_sentence(term):
    """Search for 1 English sentence containing the term"""
    queries = [
        f'"{term}" disease',
        f'"{term}" symptoms',
        f'"{term}" treatment',
        f'"{term}" medical',
        f'"{term}"'
    ]
    
    for query in queries:
        results = search_duckduckgo(query, max_results=5)
        
        for result in results:
            url = result['url']
            # Check if from medical site
            domain = urlparse(url).netloc.lower()
            is_medical = any(site in domain for site in EN_MEDICAL_SITES)
            
            # Try snippet first
            snippet = result.get('snippet', '')
            sentences = extract_sentences(snippet, term, is_vietnamese=False)
            if sentences:
                return sentences[0], url
            
            # If no good sentence in snippet, fetch page
            if is_medical:
                content = fetch_page_content(url)
                sentences = extract_sentences(content, term, is_vietnamese=False)
                if sentences:
                    return sentences[0], url
        
        time.sleep(1)  # Be polite to servers
    
    return None, None

def search_vietnamese_sentences(term):
    """Search for 2 Vietnamese sentences containing the term"""
    queries = [
        f'"{term}" là gì',
        f'"{term}" bệnh',
        f'"{term}" triệu chứng',
        f'"{term}" điều trị',
        f'"{term}"'
    ]
    
    sentences_found = []
    urls_found = []
    
    for query in queries:
        if len(sentences_found) >= 2:
            break
            
        results = search_duckduckgo(query, max_results=5)
        
        for result in results:
            if len(sentences_found) >= 2:
                break
                
            url = result['url']
            # Check if from medical site
            domain = urlparse(url).netloc.lower()
            is_medical = any(site in domain for site in VI_MEDICAL_SITES)
            
            # Try snippet first
            snippet = result.get('snippet', '')
            sentences = extract_sentences(snippet, term, is_vietnamese=True)
            for sent in sentences:
                if sent not in sentences_found:
                    sentences_found.append(sent)
                    urls_found.append(url)
                    if len(sentences_found) >= 2:
                        break
            
            # If not enough, fetch page
            if len(sentences_found) < 2 and is_medical:
                content = fetch_page_content(url)
                sentences = extract_sentences(content, term, is_vietnamese=True)
                for sent in sentences:
                    if sent not in sentences_found:
                        sentences_found.append(sent)
                        urls_found.append(url)
                        if len(sentences_found) >= 2:
                            break
        
        time.sleep(1)  # Be polite to servers
    
    # Return exactly 2 sentences or None
    if len(sentences_found) >= 2:
        return sentences_found[:2], urls_found[:2]
    return None, None

def process_csv(input_file, output_file, failed_file, max_terms=None):
    """Process CSV file and search for sentences"""
    results = []
    failed_terms = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if max_terms:
            rows = rows[:max_terms]
        
        total = len(rows)
        
        for idx, row in enumerate(rows, start=1):
            en_term = row.get('English', '').strip()
            vi_term = row.get('Vietnamese', '').strip()
            
            if not en_term or not vi_term:
                print(f"Row {idx}: Missing term, skipping")
                continue
            
            print(f"\n[{idx}/{total}] Processing: {en_term} / {vi_term}")
            
            # Search English sentence
            print(f"  Searching English sentence...")
            en_sentence, en_url = search_english_sentence(en_term)
            
            # Search Vietnamese sentences
            print(f"  Searching Vietnamese sentences...")
            vi_sentences, vi_urls = search_vietnamese_sentences(vi_term)
            
            # Check if we found everything
            if en_sentence and vi_sentences and len(vi_sentences) == 2:
                results.append({
                    'index': idx,
                    'en_term': en_term,
                    'vi_term': vi_term,
                    'en_sentence': en_sentence,
                    'vi_sentence_1': vi_sentences[0],
                    'vi_sentence_2': vi_sentences[1],
                    'en_url': en_url or '',
                    'vi_url_1': vi_urls[0] if vi_urls else '',
                    'vi_url_2': vi_urls[1] if len(vi_urls) > 1 else ''
                })
                print(f"  ✓ Success!")
            else:
                failed_terms.append({
                    'index': idx,
                    'en_term': en_term,
                    'vi_term': vi_term,
                    'reason': f"EN: {bool(en_sentence)}, VI: {len(vi_sentences) if vi_sentences else 0}/2"
                })
                print(f"  ✗ Failed - EN: {bool(en_sentence)}, VI: {len(vi_sentences) if vi_sentences else 0}/2")
            
            # Save progress periodically
            if idx % 10 == 0:
                save_results(results, output_file)
                save_failed(failed_terms, failed_file)
                print(f"\n  Progress saved at {idx} terms")
            
            time.sleep(2)  # Be polite to servers
    
    # Final save
    save_results(results, output_file)
    save_failed(failed_terms, failed_file)
    
    print(f"\n\nCompleted!")
    print(f"Success: {len(results)} terms")
    print(f"Failed: {len(failed_terms)} terms")

def save_results(results, output_file):
    """Save results to CSV"""
    if not results:
        return
    
    fieldnames = [
        'index', 'en_term', 'vi_term', 'en_sentence',
        'vi_sentence_1', 'vi_sentence_2', 'en_url', 'vi_url_1', 'vi_url_2'
    ]
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def save_failed(failed_terms, failed_file):
    """Save failed terms to CSV"""
    if not failed_terms:
        return
    
    fieldnames = ['index', 'en_term', 'vi_term', 'reason']
    
    with open(failed_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_terms)

if __name__ == '__main__':
    import sys
    
    input_file = 'test.csv'
    output_file = 'medical_sentences_output.csv'
    failed_file = 'failed_terms.csv'
    
    # Configuration
    # Set max_terms to None to process all terms, or a number to limit
    max_terms = None  # Change to 5 for testing, None for full dataset
    
    if len(sys.argv) > 1:
        try:
            max_terms = int(sys.argv[1])
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}. Using default.")
    
    print("=" * 60)
    print("Medical Term Sentence Search")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Failed terms file: {failed_file}")
    if max_terms:
        print(f"Processing first {max_terms} terms (testing mode)")
    else:
        print("Processing ALL terms (full mode)")
    print("\nNote: This will take a long time for full dataset")
    print("Progress is saved every 10 terms\n")
    print("=" * 60)
    
    process_csv(input_file, output_file, failed_file, max_terms=max_terms)

