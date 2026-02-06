# (one-time) python -c "import nltk; nltk.download('punkt')"

import re
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- Config ----------------
INPUT_CSV = "assignments/zb_assgn/backup/file_content.csv"
OUTPUT_CSV = "assignments/zb_assgn/backup/file_content_cleaned.csv"
COMMON_PHRASES_OUT = "assignments/zb_assgn/backup/common_phrases.txt"

TEXT_COL = "Article Content"             # source text column
NAME_COL = "Name"                        # will be overwritten / fixed
URL_COL = "URL"
ARTICLE_ID_COL = "Article ID"

# phrase detection parameters
MIN_DOC_FRACTION = 0.9   # phrase must appear in >= 90% of articles
NGRAM_RANGE = (5, 10)    # ngram range for initial detection (min_n, max_n)
MIN_PHRASE_FREQ = 2      # absolute count floor for safety (set low if many docs)

# longest-gram search (search n from max_n down to 2)
LONGEST_SEARCH_MAX = NGRAM_RANGE[1]

# acronyms heuristics for naming (common ones)
ACRONYMS = {"api","html","css","js","http","https","sql","db","ui","ux","json","csv","svg","md","jwt","tls"}

# ---------------- Helpers ----------------

def extract_id_and_slug_from_url(url: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Extract numeric id and slug from URLs of form:
      - /article/<id>-slug...
      - /category/<id>-slug...
    Handles trailing pages like /category/303-features/2
    Returns (id:int or None, slug:str or None)
    """
    if not isinstance(url, str):
        return None, None
    url = url.strip()
    # canonicalize: strip trailing slash except if only domain
    url = re.sub(r"/+$", "", url)
    # regex: capture segment like /article/309-how-update-task-statuses or /category/172-api-documentation
    m = re.search(r"/(?:article|category)/(\d+)-([^/]+)", url, flags=re.IGNORECASE)
    if m:
        num = int(m.group(1))
        slug = m.group(2)
        # the slug might include trailing page numbers or extra segments; keep only the main slug part
        slug = slug.strip()
        return num, slug
    # fallback: some sites may have /<something>/<id> pattern, attempt generic
    m2 = re.search(r"/(\d+)-([^/]+)", url)
    if m2:
        return int(m2.group(1)), m2.group(2)
    return None, None

def slug_to_title(slug: str) -> str:
    """
    Convert slug 'how-update-task-statuses' -> 'How Update Task Statuses'
    Use small acronym mapping to keep API/HTML uppercase.
    """
    if not slug:
        return ""
    # remove page suffix numbers like '/2' that somehow got into slug
    slug = re.sub(r"/\d+$", "", slug)
    # replace separators with spaces
    words = re.split(r"[-_]+", slug)
    clean_words = []
    for w in words:
        if not w:
            continue
        lw = w.lower()
        if lw in ACRONYMS:
            clean_words.append(lw.upper())
        elif lw.isupper():
            clean_words.append(lw)  # preserve
        else:
            # capitalize first letter (but keep small words meaningful)
            clean_words.append(lw.capitalize())
    title = " ".join(clean_words).strip()
    # final cleanup: collapse multiple spaces
    title = re.sub(r"\s{2,}", " ", title)
    return title

def find_common_ngrams(docs: List[str], ngram_range: Tuple[int,int], min_doc_fraction: float) -> Tuple[List[str], List[int]]:
    """
    Use CountVectorizer to find ngrams across docs that appear in >= min_doc_fraction.
    Returns (phrases, counts)
    """
    min_df = max(1, int(min_doc_fraction * len(docs))) if min_doc_fraction < 1 else int(min_doc_fraction)
    vectorizer = CountVectorizer(ngram_range=ngram_range, lowercase=True, min_df=min_df)
    X = vectorizer.fit_transform(docs)
    phrases = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1.tolist()
    return list(phrases), counts

def find_longest_common_phrase(docs: List[str], max_n: int, min_doc_fraction: float) -> Optional[str]:
    """
    Search from max_n down to 2 for any n-grams present in >= min_doc_fraction of docs.
    Return the single longest phrase found (first encountered).
    """
    num_docs = len(docs)
    min_df = max(1, int(min_doc_fraction * num_docs)) if min_doc_fraction < 1 else int(min_doc_fraction)
    for n in range(max_n, 1, -1):
        vec = CountVectorizer(ngram_range=(n,n), lowercase=True, min_df=min_df)
        try:
            X = vec.fit_transform(docs)
        except ValueError:
            # no ngrams of this size
            continue
        phrases = vec.get_feature_names_out()
        if len(phrases) > 0:
            # choose the longest phrase (by token length — equal here) ; return first
            # since we're iterating descending n, the first found is longest by n
            return phrases[0]
    return None

def remove_phrases_from_text(text: str, phrases: List[str]) -> str:
    """
    Remove each phrase as a whole sequence from text (case-insensitive).
    We sort phrases by length desc to avoid partial matches interfering.
    """
    if not text:
        return ""
    cleaned = text
    # sort by length descending to remove longer phrases first
    phrases_sorted = sorted(set(phrases), key=lambda s: len(s), reverse=True)
    for phrase in phrases_sorted:
        if not phrase or len(phrase.strip()) == 0:
            continue
        # create word-boundary aware pattern but phrase may contain punctuation; use regex escape
        pattern = r"\b" + re.escape(phrase) + r"\b"
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    # normalize whitespace and trim
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()

# ---------------- Main ----------------

def main():
    p_in = Path(INPUT_CSV)
    if not p_in.exists():
        raise FileNotFoundError(f"Input CSV not found: {p_in}")

    df = pd.read_csv(p_in)
    # ensure columns exist
    if TEXT_COL not in df.columns:
        raise ValueError(f"Input CSV missing expected column '{TEXT_COL}'")

    # load docs
    docs = df[TEXT_COL].fillna("").astype(str).tolist()
    n_docs = len(docs)
    print(f"[info] Loaded {n_docs} documents")

    # 1) Find common phrases with the configured ngram range
    print(f"[info] Detecting common phrases (ngram_range={NGRAM_RANGE}) with min_doc_fraction={MIN_DOC_FRACTION}")
    phrases, counts = find_common_ngrams(docs, ngram_range=NGRAM_RANGE, min_doc_fraction=MIN_DOC_FRACTION)
    common_phrases = [p for p, c in zip(phrases, counts) if c >= max(MIN_PHRASE_FREQ, int(MIN_DOC_FRACTION * n_docs))]
    print(f"[info] Found {len(common_phrases)} candidate common phrases from initial range")

    # 2) Find the longest common phrase by scanning descending n
    print(f"[info] Searching for longest common phrase up to {LONGEST_SEARCH_MAX}-grams")
    longest_phrase = find_longest_common_phrase(docs, max_n=LONGEST_SEARCH_MAX, min_doc_fraction=MIN_DOC_FRACTION)
    if longest_phrase:
        print(f"[info] Longest common phrase detected: '{longest_phrase}'")
        if longest_phrase not in common_phrases:
            common_phrases.append(longest_phrase)
    else:
        print("[info] No long common phrase found")

    # save the common phrase list for auditing
    Path(COMMON_PHRASES_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(COMMON_PHRASES_OUT, "w", encoding="utf-8") as f:
        for phrase in sorted(common_phrases, key=lambda s: len(s), reverse=True):
            f.write(phrase + "\n")
    print(f"[info] Wrote {len(common_phrases)} common phrases to {COMMON_PHRASES_OUT}")

    # 3) Remove those phrases from each document
    print("[info] Removing boilerplate phrases from documents...")
    df["Cleaned Article Content"] = df[TEXT_COL].fillna("").astype(str).apply(
        lambda t: remove_phrases_from_text(t, common_phrases)
    )

    # 4) Fix Name and Article ID from URL
    print("[info] Extracting Article ID and Name from URL...")
    new_ids = []
    new_names = []
    for idx, row in df.iterrows():
        url = row.get(URL_COL, "")
        existing_name = row.get(NAME_COL, "")
        num, slug = extract_id_and_slug_from_url(url)
        if num:
            article_id = f"KB-{num}"
        else:
            # fallback: try to keep existing Article ID if present else generate sequential
            if ARTICLE_ID_COL in df.columns and pd.notna(row.get(ARTICLE_ID_COL)):
                article_id = row.get(ARTICLE_ID_COL)
            else:
                article_id = f"KB-{idx+1:03d}"
        # name: prefer slug if present; else try to derive from existing name; else empty
        if slug:
            name = slug_to_title(slug)
        else:
            # fallback: try to turn the URL tail into a readable name
            tail = ""
            try:
                tail = Path(urlparse(url).path).name
            except Exception:
                tail = ""
            if tail:
                name = slug_to_title(tail)
            elif isinstance(existing_name, str) and existing_name.strip():
                name = existing_name.strip()
            else:
                name = ""
        new_ids.append(article_id)
        new_names.append(name)

    df[ARTICLE_ID_COL] = new_ids
    df[NAME_COL] = new_names

    # 5) Extra: strip short worthless content (optional)
    # e.g. remove articles where cleaned content is less than 10 tokens? We'll keep as-is.
    # df['Cleaned Article Content'] = df['Cleaned Article Content'].apply(lambda t: t if len(t.split())>5 else "")

    # Save cleaned CSV
    out_p = Path(OUTPUT_CSV)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"[done] Wrote cleaned CSV to: {out_p}")

if __name__ == "__main__":
    main()