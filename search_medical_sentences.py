import csv
import re
import time
import random
import requests
import pandas as pd
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse


HEADERS = {
    "User-Agent": "Mozilla/5.0 (NLP-Medical-Research-Bot/1.0; Contact: research@example.edu)"
}

MIN_WORDS = 8
MAX_WORDS = 100

EN_MEDICAL_DOMAINS = [
    'wikipedia.org', 'pubmed.ncbi.nlm.nih.gov', 'mayoclinic.org', 'nhs.uk',
    'webmd.com', 'medlineplus.gov', 'healthline.com', 'medscape.com',
    'who.int', 'ncbi.nlm.nih.gov', 'bmj.com', 'thelancet.com'
]

VI_MEDICAL_DOMAINS = [
    'vi.wikipedia.org', 'vinmec.com', 'medlatec.vn', 'youmed.vn',
    'suckhoedoisong.vn', 'bvct.org.vn', 'benhviennhidong.org.vn',
    'vietmedical.com', 'pharmacity.vn'
]

BLACKLIST_PATTERNS = {
    "en": [" is a ", " is an ", " is the ", " refers to ", "defined as ",
           " is defined", "known as ", "also known as ", "means ",
           " is common ", " is a disease", " is a condition"],
    "vi": [" là một ", " là bệnh ", " là chứng ", " là tình trạng ",
           "được định nghĩa ", "có nghĩa là ", "còn gọi là "]
}

CLINICAL_KEYWORDS = {
    "en": ["treatment", "diagnosis", "symptoms", "patient", "case",
           "management", "therapy", "clinical", "study", "results",
           "risk", "complications", "procedure", "presented with"],
    "vi": ["điều trị", "chẩn đoán", "triệu chứng", "bệnh nhân", "ca bệnh",
           "phác đồ", "liệu pháp", "lâm sàng", "nghiên cứu", "kết quả",
           "biến chứng", "thủ thuật"]
}


def setup_logger(log_path: str):
    logger = logging.getLogger("medical_crawler")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # avoid duplicate handlers

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_medical_domain(url):
    domain = urlparse(url).netloc.lower()
    return any(d in domain for d in EN_MEDICAL_DOMAINS + VI_MEDICAL_DOMAINS)


def search_duckduckgo(query, lang, num_results=5):
    try:
        region = "wt-wt" if lang == "en" else "vn-vn"
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}&kl={region}"
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        urls = []

        for a in soup.find_all("a", class_="result__a", limit=num_results):
            link = a.get("href")
            if link and link.startswith("http"):
                if "uddg=" in link:
                    link = link.split("uddg=")[1]
                urls.append(link)

        return urls
    except Exception:
        return []


def score_sentence(sentence, keyword, lang):
    score = 0
    s = sentence.lower()

    if keyword.lower() in s:
        score += 10

    for k in CLINICAL_KEYWORDS[lang]:
        if k in s:
            score += 3

    wc = len(sentence.split())
    if MIN_WORDS <= wc <= MAX_WORDS:
        score += 5

    for p in BLACKLIST_PATTERNS[lang]:
        if p in s:
            score -= 20

    return score


def extract_best_sentence(text, keyword, lang):
    text = clean_text(text)
    sentences = re.split(r'[.!?]\s*', text)

    best, best_score = None, -999
    for s in sentences:
        if keyword.lower() not in s.lower():
            continue
        if len(s.split()) < MIN_WORDS:
            continue
        score = score_sentence(s, keyword, lang)
        if score > best_score:
            best, best_score = s.strip(), score

    return best


def crawl_url(url, term, lang, fallback_terms):
    try:
        time.sleep(random.uniform(1.5, 3))
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()

        text = " ".join(p.get_text(" ") for p in soup.find_all("p"))
        sent = extract_best_sentence(text, term, lang)
        if sent:
            return sent

        for f in fallback_terms:
            sent = extract_best_sentence(text, f, lang)
            if sent:
                return sent

        return None
    except Exception:
        return None


def get_fallback_terms(term, lang):
    if lang == "en":
        return [term, term.split()[-1], f"{term} treatment"]
    return [term, term.split()[-1], f"điều trị {term}"]


def crawl_medical_example(term, lang, logger):
    if not term or pd.isna(term):
        return "UNOBSERVED"

    logger.info(f"Searching ({lang}) '{term}'")
    query = f'"{term}"' if lang == "en" else f'"{term}"'
    urls = search_duckduckgo(query, lang)

    urls = sorted(urls, key=lambda x: not is_medical_domain(x))
    fallback_terms = get_fallback_terms(term, lang)

    for url in urls:
        sent = crawl_url(url, term, lang, fallback_terms)
        if sent:
            logger.info(f"FOUND ({lang}) → {sent[:120]}")
            return sent

    logger.warning(f"FAILED ({lang}) '{term}'")
    return "UNOBSERVED"



def enrich_medical_dataset(
    input_csv,
    output_csv,
    save_every=20,
    sample_size=None,
    log_file="medical_crawl.log"
):
    logger = setup_logger(log_file)
    logger.info("STARTING CRAWL")

    df = pd.read_csv(input_csv)
    if sample_size:
        df = df.head(sample_size)

    for col in ["English example", "Vietnamese example"]:
        if col not in df.columns:
            df[col] = "PENDING"

    total = len(df)

    for idx, row in df.iterrows():
        logger.info(f"Row {idx+1}/{total}")

        if row["English example"] == "PENDING":
            df.at[idx, "English example"] = crawl_medical_example(
                str(row["English"]), "en", logger
            )

        if row["Vietnamese example"] == "PENDING":
            df.at[idx, "Vietnamese example"] = crawl_medical_example(
                str(row["Vietnamese"]), "vi", logger
            )

        if (idx + 1) % save_every == 0:
            autosave = output_csv.replace(".csv", ".autosave.csv")
            df.to_csv(autosave, index=False)
            logger.info(f"AUTO-SAVED → {autosave}")

    df.to_csv(output_csv, index=False)
    logger.info(f"FINAL SAVE → {output_csv}")
    logger.info("DONE")


enrich_medical_dataset(
    input_csv="/kaggle/input/medical-5000/MeddictGem01_5000samples(Sheet1).csv",
    output_csv="medical_pairs_with_examples_FINAL.csv",
    save_every=25,
    log_file="medical_crawl.log"
)
