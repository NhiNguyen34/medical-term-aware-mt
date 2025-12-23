import random
import re
import time
import pandas as pd
import logging
import traceback
import urllib.parse
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

CHROME_BINARY_LOCATION = "/usr/bin/chrome-linux64/chrome"
CHROMEDRIVER_BINARY_LOCATION = "/usr/bin/chromedriver-linux64/chromedriver"

MIN_WORDS = 8
MAX_WORDS = 150

CLINICAL_KEYWORDS = {
    "en": [
        "treatment", "diagnosis", "symptoms", "patient", "case",
        "management", "therapy", "clinical", "study", "results",
        "risk", "complications", "procedure", "presented with",
        "condition", "disorder", "disease", "medication", "pathology"
    ],
    "vi": [
        "điều trị", "chẩn đoán", "triệu chứng", "bệnh nhân", "ca bệnh",
        "phác đồ", "liệu pháp", "lâm sàng", "nghiên cứu", "kết quả",
        "biến chứng", "thủ thuật", "bệnh lý", "rối loạn", "thuốc"
    ]
}

def setup_logger(log_path):
    logger = logging.getLogger("medical_crawler")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def add_driver_options(options):
    chrome_options = Options()
    for opt in options:
        chrome_options.add_argument(opt)
    return chrome_options

def initialize_driver():
    options = add_driver_options([
        "--headless",
        "--no-sandbox",
        "--start-fullscreen",
        "--allow-insecure-localhost",
        "--disable-dev-shm-usage",
        "user-agent=Chrome/116.0.5845.96"
    ])
    options.binary_location = CHROME_BINARY_LOCATION
    driver = webdriver.Chrome(
        executable_path=CHROMEDRIVER_BINARY_LOCATION,
        options=options
    )
    return driver


def is_file_url(url):
    u = url.lower()
    return any(x in u for x in [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".png", ".jpg", ".jpeg"])

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'(?i)(jump to content|from wikipedia|navigation|search|encyclopedia|free encyclopedia|edit source|history|retrieved)', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def score_sentence(sentence, keyword, lang):
    s = sentence.lower()
    k = keyword.lower()
    if len(s) < 20:
        return -100
    if any(x in s for x in ["cookie", "browser", "javascript"]):
        return -100
    if any(s.startswith(x) for x in ["jump to", "from wikipedia", "copyright", "all rights", "log in", "sign up", "menu", "search", "retrieved", "skip to"]):
        return -100
    score = 0
    if k in s:
        score += 10
    if s.startswith(k):
        score += 5
    for kw in CLINICAL_KEYWORDS[lang]:
        if kw in s:
            score += 3
    wc = len(sentence.split())
    score += 5 if MIN_WORDS <= wc <= MAX_WORDS else -5
    score += 5 if s.endswith(('.', '!', '?')) else -10
    return score

def extract_best_sentence(text, keyword, lang):
    sentences = split_sentences(clean_text(text))
    best, best_score = None, -1
    for s in sentences:
        if keyword.lower() not in s.lower():
            continue
        if len(s.split()) < MIN_WORDS:
            continue
        sc = score_sentence(s, keyword, lang)
        if sc > best_score and sc > 0:
            best, best_score = s, sc
    return best

def search_duckduckgo_selenium(driver, query, logger):
    try:
        time.sleep(random.uniform(2.0, 4.0))
        driver.get(f"https://duckduckgo.com/?q={query}&t=h_&ia=web")
        WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.ID, "react-layout")))
        driver.execute_script(f"window.scrollTo(0, {random.randint(100, 500)});")
        time.sleep(1.0)
        urls = []
        for a in driver.find_elements(By.CSS_SELECTOR, "a[data-testid='result-title-a']")[:2]:
            href = a.get_attribute("href")
            if href and href.startswith("http") and not is_file_url(href):
                urls.append(href)
        return urls
    except Exception as e:
        logger.debug(e)
        return []

def crawl_url_selenium(driver, url, term, lang, fallback_terms, logger):
    try:
        if is_file_url(url):
            return None
        time.sleep(random.uniform(2.0, 5.0))
        driver.get(url)
        try:
            ct = driver.execute_script("return document.contentType;")
            if "application/pdf" in ct or driver.current_url.lower().endswith(".pdf"):
                return None
        except:
            pass
        WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header", "form", "noscript", "aside", "figure", "table", "button"]):
            t.decompose()
        text = soup.get_text("\n", strip=True)
        if len(text) < 50:
            return None
        sent = extract_best_sentence(text, term, lang)
        if sent:
            return sent
        for f in fallback_terms:
            sent = extract_best_sentence(text, f, lang)
            if sent:
                return sent
        return None
    except:
        return None

def crawl_pubmed_selenium(driver, term, logger):
    try:
        url = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(term)}"
        driver.get(url)
        time.sleep(random.uniform(2.0, 4.0))
        results = driver.find_elements(By.CSS_SELECTOR, "a.docsum-title")
        if results:
            driver.execute_script("arguments[0].scrollIntoView();", results[0])
            time.sleep(0.5)
            results[0].click()
            time.sleep(random.uniform(2.0, 4.0))
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.full-view-snippet")))
        text = driver.find_element(By.CSS_SELECTOR, "div.full-view-snippet").text
        sent = extract_best_sentence(text, term, "en")
        if not sent:
            sent = extract_best_sentence(text, term.split()[-1], "en")
        if sent:
            return sent, driver.current_url
    except Exception as e:
        logger.debug(e)
    return None, None

def enrich_medical_dataset(input_csv, output_csv, start_index=0, end_index=None, save_every=10, log_file="selenium_crawl.log"):
    logger = setup_logger(log_file)
    df = pd.read_csv(input_csv)
    for col in ["English example", "Vietnamese example", "English source URL", "Vietnamese source URL"]:
        if col not in df.columns:
            df[col] = "PENDING"
    if end_index is None:
        end_index = len(df)
    df_slice = df.iloc[start_index:end_index].copy()
    driver = initialize_driver()
    try:
        for i, (idx, row) in enumerate(df_slice.iterrows()):
            def process_lang(col_ex, col_url, term, lang):
                val = row[col_ex]
                if pd.notna(val) and val not in ["PENDING", "UNOBSERVED"] and len(str(val)) > 10:
                    return val, row[col_url]
                if pd.isna(term) or not str(term).strip():
                    return "UNOBSERVED", ""
                queries = (
                    [f'"{term}" medical', f'"{term}" disease', f'"{term}"'] if lang == "en"
                    else [f'"{term}" medical', f'"{term}" y', f'"{term}" bệnh', f'"{term}"']
                )
                urls = []
                for q in queries:
                    urls = search_duckduckgo_selenium(driver, q, logger)
                    if urls:
                        break
                found_sent, found_url = None, ""
                if urls:
                    for u in urls[:2]:
                        found_sent = crawl_url_selenium(driver, u, term, lang, [term.split()[-1]], logger)
                        if found_sent:
                            found_url = u
                            break
                if lang == "en" and not found_sent:
                    pm_sent, pm_url = crawl_pubmed_selenium(driver, term, logger)
                    if pm_sent:
                        return pm_sent, pm_url
                return (found_sent, found_url) if found_sent else ("UNOBSERVED", "")
            es, eu = process_lang("English example", "English source URL", row["English"], "en")
            df.at[idx, "English example"] = es
            df.at[idx, "English source URL"] = eu
            vs, vu = process_lang("Vietnamese example", "Vietnamese source URL", row["Vietnamese"], "vi")
            df.at[idx, "Vietnamese example"] = vs
            df.at[idx, "Vietnamese source URL"] = vu
            if (i + 1) % save_every == 0:
                df.to_csv(output_csv, index=False)
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
    finally:
        try:
            driver.quit()
        except:
            pass
    df.to_csv(output_csv, index=False)

enrich_medical_dataset(
    input_csv="/kaggle/input/medical-5000/MeddictGem01_5000samples(Sheet1).csv",
    output_csv="medical_pairs_selenium_500_v2.csv",
    start_index=0,
    end_index=2000,
    save_every=10,
    log_file="selenium_debug_500.log"
)
