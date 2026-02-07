import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from langchain_ollama import ChatOllama
from collections import Counter

# =========================
# CONFIG
# =========================
INPUT_CSV = "assignments/zb_assgn/data/file_content_cleaned.csv"
OUTPUT_CSV = "assignments/zb_assgn/data/file_content_populated.csv"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
USE_LLM_REFINEMENT = True

# =========================
# LLM INIT
# =========================
if USE_LLM_REFINEMENT:
    chat = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
        validate_model_on_init=False,
    )

# =========================
# RESPONSE EXTRACTION
# =========================
def extract_text_from_resp(resp):
    if isinstance(resp, str):
        return resp.strip()

    if isinstance(resp, (list, tuple)):
        return "\n".join(filter(None, (extract_text_from_resp(r) for r in resp))).strip()

    if hasattr(resp, "content"):
        return str(resp.content).strip()

    if hasattr(resp, "text"):
        return str(resp.text).strip()

    if hasattr(resp, "generations"):
        try:
            return str(resp.generations[0][0].text).strip()
        except Exception:
            pass

    return str(resp).strip()

# =========================
# UTILITIES
# =========================

# New code
def load_existing_urls(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path, dtype=str)
    if "URL" not in df.columns:
        return set()
    return set(df["URL"].dropna().astype(str))


def filter_only_new_rows(df_cleaned: pd.DataFrame, populated_csv: str) -> pd.DataFrame:
    """
    Return only rows whose URL is NOT already present in file_content_populated.csv
    """
    existing_urls = load_existing_urls(populated_csv)

    if not existing_urls:
        print("[info] No populated CSV found — full cleaned CSV will be processed")
        return df_cleaned.copy()

    df_new = df_cleaned[~df_cleaned["URL"].astype(str).isin(existing_urls)].copy()

    print(f"[info] Incremental mode: {len(df_new)} new rows to process")
    return df_new
# /New code

def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0

def chunk_text(text: str, max_chars=3000):
    if not text:
        return []
    paragraphs = text.split("\n\n")
    chunks, cur = [], ""
    for p in paragraphs:
        if len(cur) + len(p) <= max_chars:
            cur += ("\n\n" + p if cur else p)
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

def top_n_word_candidates(texts, n=30):
    """
    Safe CountVectorizer wrapper.
    Returns [] if vocabulary is empty (stopwords-only / junk text).
    """
    # Defensive: remove empty / whitespace-only texts
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9]+\b",
    )

    try:
        X = vectorizer.fit_transform(texts)
    except ValueError as e:
        # Expected edge case: stopwords-only or junk chunks
        if "empty vocabulary" in str(e).lower():
            return []
        raise  # re-raise anything else (real bug)

    freqs = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    return sorted(zip(terms, freqs), key=lambda x: -x[1])[:n]

# =========================
# LLM TASKS (SINGLE PURPOSE)
# =========================
def llm_refine_keywords(candidate_tokens, top_k=8, context=""):
    if not USE_LLM_REFINEMENT or not candidate_tokens:
        return [t for t, _ in candidate_tokens][:top_k]

    prompt = (
        "Select the most relevant keywords for this article.\n"
        f"Context: {context}\n"
        f"Candidates: {', '.join(t for t, _ in candidate_tokens)}\n\n"
        f"Return exactly {top_k} keywords as a comma-separated list."
    )

    resp = chat.invoke(prompt)
    text = extract_text_from_resp(resp)
    return [t.strip() for t in text.split(",") if t.strip()][:top_k]

def llm_classify_content_type(full_text):
    if not USE_LLM_REFINEMENT:
        st = full_text.lower()
        if "how to" in st or "step" in st:
            return "How-to Guide"
        if "faq" in st:
            return "FAQ"
        if "error" in st or "troubleshoot" in st:
            return "Troubleshooting"
        return "Reference"

    prompt = (
        "Classify the following article into exactly ONE category:\n"
        "How-to Guide, FAQ, Troubleshooting, Reference, Announcement, Other\n\n"
        "Article:\n"
        f"{full_text}\n\n"
        "Return ONLY the category name."
    )

    resp = chat.invoke(prompt)
    label = extract_text_from_resp(resp).strip('"').strip("'")

    valid = {
        "How-to Guide",
        "FAQ",
        "Troubleshooting",
        "Reference",
        "Announcement",
        "Other",
    }
    return label if label in valid else "Other"

# =========================
# MAIN PIPELINE
# =========================
def process_csv(infile=INPUT_CSV, outfile=OUTPUT_CSV):
    # df = pd.read_csv(infile, dtype=str).fillna("")
    # New Code
    df_all = pd.read_csv(infile, dtype=str).fillna("")

    # 🔹 SMART INCREMENTAL FILTER
    df = filter_only_new_rows(df_all, outfile)

    if df.empty:
        print("[info] No new rows to process — exiting cleanly.")
        return
    # /New Code

    for col in ["Topics Covered", "Content Type", "Word Count"]:
        if col not in df.columns:
            df[col] = ""

    for idx, row in df.iterrows():
        article_id = row.get("Article ID", f"row-{idx}")
        title = row.get("Name", "").strip()
        content = row.get("Cleaned Article Content") or row.get("Article Content") or ""

        wc = word_count(content)
        df.at[idx, "Word Count"] = wc

        # keyword candidates
        chunks = chunk_text(content) or [content]
        counter = Counter()
        for ch in chunks:
            for term, count in top_n_word_candidates([ch]):
                counter[term] += count

        cand_list = counter.most_common(40)
        context = f"{title} {row.get('Category','')}"
        topics = llm_refine_keywords(cand_list, top_k=8, context=context)
        df.at[idx, "Topics Covered"] = ", ".join(topics)

        # FULL ARTICLE classification
        ctype = llm_classify_content_type(content)
        df.at[idx, "Content Type"] = ctype

        # =========================
        # LIVE OUTPUT
        # =========================
        print("\n" + "=" * 80)
        print(f"[{idx+1}/{len(df)}] {article_id}")
        print(f"Title        : {title}")
        print(f"Word Count   : {wc}")
        print(f"Content Type: {ctype}")
        print(f"Topics      : {', '.join(topics)}")
        print("=" * 80)

    # df.to_csv(outfile, index=False)
    # print(f"\n✅ DONE — output written to: {outfile}")
    
    # New Code
    write_header = not os.path.exists(outfile)

    df.to_csv(
        outfile,
        mode="a" if not write_header else "w",
        header=write_header,
        index=False
    )

    print(f"\n✅ DONE — appended {len(df)} rows to: {outfile}")
    # /New Code

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    process_csv()