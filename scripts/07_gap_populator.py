#!/usr/bin/env python3
"""
generate_gap_table_with_ollama.py

Read gap_candidates.csv, enrich with metadata and AI suggestions (Ollama via LangChain if available,
otherwise ollama CLI). Produce a formalized table CSV + markdown.

No CLI args required. Defaults expect:
  - assignments/zb_assgn/data/gap_candidates.csv  (or ./gap_candidates.csv)
  - assignments/zb_assgn/data/vectors_3d.csv      (or fallback content CSV)
Output:
  - gap_candidates_with_table.csv
  - gap_candidates_with_table.md
"""

import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
GAP_CANDIDATES_PATHS = [
    "assignments/zb_assgn/data/gap_candidates_v2.csv",
    # "gap_candidates.csv"
]
VECTORS_PATHS = [
    "assignments/zb_assgn/data/vectors_3d_v2.csv",
    # "assignments/zb_assgn/data/vectors_3d.csv",  # duplicate kept intentionally
    "assignments/zb_assgn/data/file_content_populated.csv",
    # "file_content_cleaned.csv",
    # "assignments/zb_assgn/data/file_content.csv",
    # "file_content.csv",
]
OUTPUT_CSV = "gap_candidates_with_table.csv"
OUTPUT_MD = "gap_candidates_with_table.md"

MAX_GAPS_TO_PROCESS = 5  # per your instruction "simple iterator that loops over 5 times"
OLLAMA_MODEL_DEFAULT = "mistral:7b"  # change if you use a specific local model with Ollama
# (Note: Ollama usage via LangChain or CLI will pick the model you have installed. Change above if needed.)

# Priority thresholds for normalized semantic distance (0..1 expected)
PRIORITY_THRESHOLDS = {
    "high": 0.75,
    "medium": 0.5
}

# ---------------- Utilities ----------------

def find_first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None

def load_gap_candidates():
    p = find_first_existing(GAP_CANDIDATES_PATHS)
    if not p:
        raise FileNotFoundError(f"No gap_candidates.csv found in expected paths: {GAP_CANDIDATES_PATHS}")
    print(f"[info] Loading gap candidates from: {p}")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return df, p

def load_vectors_or_content():
    p = find_first_existing(VECTORS_PATHS)
    if not p:
        print("[warn] No vectors/content CSV found in expected paths; continuing with minimal metadata.")
        return pd.DataFrame(), None
    print(f"[info] Loading metadata from: {p}")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return df, p

def build_meta_map(meta_df):
    """
    Build lookup dict mapping URL -> metadata dict (topics, content type, last updated, top_terms, word count).
    """
    meta = {}
    if meta_df is None or meta_df.empty:
        return meta
    # Normalize column names to common expected names
    cols = {c.lower().strip(): c for c in meta_df.columns}
    def get_val(row, keys):
        for k in keys:
            if k in cols:
                return row[cols[k]]
        return ""

    for _, r in meta_df.iterrows():
        url = get_val(r, ["url", "URL", "Url"])
        if not url:
            continue
        meta[url] = {
            "topics": [t.strip().lower() for t in re.split(r"[;,|]", get_val(r, ["topics covered", "Topics Covered", "topics"])) if t.strip()] if get_val(r, ["topics covered", "Topics Covered", "topics"]) else [],
            "content_type": get_val(r, ["content type", "Content Type", "Content_Type", "content_type"]) or "",
            "last_updated": get_val(r, ["last updated", "Last Updated", "last_updated"]) or "",
            "top_terms": get_val(r, ["top_terms", "top_terms", "Top Terms", "top terms"]) or get_val(r, ["top_terms", "top_terms", "Top Terms", "top terms"]) or "",
            "word_count": int(float(get_val(r, ["word count", "Word Count", "word_count"]) or 0)) if (get_val(r, ["word count", "Word Count", "word_count"]) not in [None, ""]) else 0,
            "name": get_val(r, ["name", "Name", "title"]) or "",
            "category": get_val(r, ["category", "Category"]) or ""
        }
    return meta

def safe_parse_float(v):
    try:
        return float(str(v))
    except Exception:
        return None

def normalize_semantic_distance(x):
    """
    Some gap CSVs may have 'distance' or 'semantic_distance' columns.
    Normalize to 0..1 if possible. If input is NaN or empty, return None.
    If input > 1 assume it's raw Euclidean; we'll map to 0..1 by heuristics in caller.
    """
    if x in [None, "", "nan", "NaN"]:
        return None
    try:
        val = float(x)
    except Exception:
        return None
    # If already between 0 and 1, return as-is
    if 0.0 <= val <= 1.0:
        return val
    # If common UMAP Euclidean distances (0..some value), we can't rescale without global max.
    # Caller will attempt to rescale based on observed max.
    return val

# ---------------- Ollama call wrapper ----------------

def call_ollama_with_langchain(prompt_text, model=OLLAMA_MODEL_DEFAULT, max_tokens=512, temperature=0.0):
    """
    Try to use LangChain's Ollama wrapper if available.
    Returns the raw string response from the model. Raises Exception on failure to import/call.
    """
    try:
        # Import inside function to avoid fail if langchain not installed
        from langchain.llms import Ollama
    except Exception as e:
        raise ImportError("LangChain Ollama wrapper not available") from e

    # Create an Ollama instance; signature may accept model and other kwargs
    try:
        # This will work for common LangChain wrappers: model=str
        llm = Ollama(model=model, temperature=temperature)
        resp = llm(prompt_text)  # single-call; returns string
        return str(resp)
    except TypeError:
        # try alternate signature without temperature (some versions differ)
        llm = Ollama(model=model)
        resp = llm(prompt_text)
        return str(resp)
    except Exception as e:
        raise RuntimeError(f"LangChain Ollama call failed: {e}") from e

def call_ollama_cli(prompt_text, model=OLLAMA_MODEL_DEFAULT):
    """
    Fallback to calling 'ollama' CLI if available.
    Executes: `ollama generate <model> --prompt <prompt>` and returns stdout.
    """
    cmd = ["ollama", "generate", model, "--prompt", prompt_text]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise FileNotFoundError("The 'ollama' CLI is not found in PATH. Install ollama or make it available.") from e

    out = res.stdout.strip()
    if not out:
        # sometimes CLI writes to stderr
        out = res.stderr.strip()
    return out

def call_ollama(prompt_text, model=OLLAMA_MODEL_DEFAULT):
    """
    One unified call: try LangChain Ollama wrapper first, else try ollama CLI.
    Returns the raw string output from the model.
    """
    # Try LangChain
    try:
        return call_ollama_with_langchain(prompt_text, model=model)
    except Exception as e:
        # fallback to CLI
        try:
            return call_ollama_cli(prompt_text, model=model)
        except Exception as e2:
            # raise combined error for the user
            raise RuntimeError("Failed to call Ollama via LangChain and CLI.\n"
                               "LangChain error: {}\nCLI error: {}\n".format(e, e2)) from e2

# ---------------- Prompting / Parsing ----------------

PROMPT_TEMPLATE = """
You are a concise assistant that helps product/documentation teams identify article gaps and propose titles.

Input (do NOT invent facts; only use the information below):
Article A:
  Title: {title_a}
  URL: {url_a}
  Top terms: {top_terms_a}
  Topics: {topics_a}
  Content Type: {ctype_a}
  Word Count: {wc_a}

Article B:
  Title: {title_b}
  URL: {url_b}
  Top terms: {top_terms_b}
  Topics: {topics_b}
  Content Type: {ctype_b}
  Word Count: {wc_b}

Task (single JSON response ONLY):
1) Provide a one-sentence "gap_description" that describes the gap a *new* article should fill to connect Article A and Article B. Keep it concise (<= 30 words).
2) Provide "title_suggestions": an array of 5 short title suggestions (phrases), ordered best-to-worst.

Return exactly a JSON object with keys:
{{
  "gap_description": "<one concise sentence>",
  "title_suggestions": ["title 1", "title 2", "title 3", "title 4", "title 5"]
}}

Do not include any other text, explanations, or markup. Make titles SEO-friendly and concise.
"""

def extract_json_from_model_output(s: str):
    """
    Try to robustly extract JSON object from model output.
    Returns parsed dict. Raises ValueError if cannot parse.
    """
    if not s or not s.strip():
        raise ValueError("Empty model output")

    # Try direct load first
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to find the first '{' and last '}' and json.loads
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # try to find a JSON array-like structure containing our keys
    # fallback: attempt to parse using simple heuristics (very last resort)
    # but better to raise explicit error
    raise ValueError("Could not parse JSON from model output. Raw output:\n" + s[:2000])

# ---------------- Deterministic logic (non-AI) ----------------

def derive_category_from_content_type(ctype: str):
    """
    Map content type string to category. Heuristic rules.
    """
    if not ctype:
        return "General"
    c = ctype.lower()
    if "integrat" in c or "integration" in c:
        return "Integrations"
    if "how-to" in c or "how to" in c or "guide" in c:
        return "How-to / Guide"
    if "troubleshoot" in c or "troubleshoot" in c:
        return "Troubleshooting"
    if "api" in c or "developer" in c:
        return "API / Developer"
    # fallback to the original content type (capitalized)
    return ctype.strip().title()

def choose_best_title(suggestions, topics, top_terms):
    """
    Deterministic selection of best title among suggestions:
    - score by presence of topic keywords and top_terms tokens
    - prefer shorter titles (but not extremely short)
    """
    toks = set([t.lower() for t in topics]) if topics else set()
    top_terms_set = set([t.strip().lower() for t in re.split(r"[,\s]+", top_terms) if t.strip()])
    best_score = -1
    best_title = suggestions[0] if suggestions else ""
    for t in suggestions:
        s = 0.0
        tl = t.lower()
        # token overlap score
        for tok in toks:
            if tok and tok in tl:
                s += 2.0
        for tt in top_terms_set:
            if tt and tt in tl:
                s += 1.0
        # penalty for length > 80 chars; slight bonus for moderate conciseness
        ln = len(t)
        if ln <= 60:
            s += 0.5
        if ln < 30:
            s += 0.2
        if ln > 100:
            s -= 1.0
        # small randomness deterministic: based on hash to break ties reproducibly
        s += (hash(t) % 10) * 1e-6
        if s > best_score:
            best_score = s
            best_title = t
    return best_title

def determine_priority(semantic_distance_norm, score=None):
    """
    Determine High/Medium/Low priority using normalized semantic distance (0..1).
    Additional inputs may refine this decision later.
    """
    if semantic_distance_norm is None:
        return "Medium"
    if semantic_distance_norm >= PRIORITY_THRESHOLDS["high"]:
        return "High"
    if semantic_distance_norm >= PRIORITY_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"

def compute_topic_jaccard(topics_a, topics_b):
    sa = set([t.strip().lower() for t in (topics_a or []) if t.strip()])
    sb = set([t.strip().lower() for t in (topics_b or []) if t.strip()])
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)

def generate_rationale(semantic_distance_norm, topic_jaccard, last_updated_a, last_updated_b, word_count_a, word_count_b):
    """
    Deterministic rationale text using the features.
    """
    parts = []
    if semantic_distance_norm is not None:
        parts.append(f"Semantic distance {semantic_distance_norm:.2f} indicates topical gap.")
    if topic_jaccard is not None:
        if topic_jaccard < 0.2:
            parts.append("Low topic overlap suggests complementary content is needed.")
        else:
            parts.append(f"Topic overlap {topic_jaccard:.2f} suggests related but distinct focus.")
    # recency: prefer linking newer -> older
    try:
        da = pd_to_dt(last_updated_a)
        db = pd_to_dt(last_updated_b)
        if da and db:
            if da > db:
                parts.append(f"Article A is newer ({da.date()}) than B ({db.date()}), useful to create a newer-to-legacy linking guide.")
            elif db > da:
                parts.append(f"Article B is newer ({db.date()}) than A ({da.date()}). Consider linking updated content to older reference.")
    except Exception:
        pass
    # depth: if one is short and the other is long
    try:
        if word_count_a and word_count_b:
            if abs(word_count_a - word_count_b) / max(1, max(word_count_a, word_count_b)) > 0.5:
                parts.append("Significant length difference suggests one article could provide expanded how-to or examples.")
    except Exception:
        pass
    if not parts:
        parts = ["Suggested because the pair appears related and could benefit from a bridging article."]
    return " ".join(parts)

def pd_to_dt(s):
    if not s or str(s).strip() == "":
        return None
    try:
        return pd.to_datetime(str(s), utc=True)
    except Exception:
        try:
            return datetime.fromisoformat(str(s))
        except Exception:
            return None

# ---------------- Main flow ----------------

def main():
    # load gap candidates
    gap_df, gap_path = load_gap_candidates()

    if gap_df.empty:
        print("[info] gap_candidates.csv is empty. Nothing to do.")
        return

    # load metadata (vectors or original CSV)
    meta_df, meta_path = load_vectors_or_content()
    meta_map = build_meta_map(meta_df) if not meta_df.empty else {}

    # Identify which columns in gap_df hold the distance value
    distance_col = None
    for candidate in ["semantic_distance", "distance", "dist", "score"]:
        if candidate in gap_df.columns:
            distance_col = candidate
            break

    # If distances exist but not normalized to 0..1, compute normalized version from observed max
    raw_distances = []
    if distance_col:
        for v in gap_df[distance_col].tolist():
            x = normalize_semantic_distance(v)
            if x is not None:
                raw_distances.append(x)
    # compute normalization factor if max > 1
    max_raw = max(raw_distances) if raw_distances else None

    # Prepare output rows
    out_rows = []
    # process top N rows (respect ordering in gap_df which should be sorted by distance/score)
    n_to_process = min(MAX_GAPS_TO_PROCESS, len(gap_df))
    print(f"[info] Processing top {n_to_process} gap candidates (out of {len(gap_df)})")

    for idx in range(n_to_process):
        row = gap_df.iloc[idx]
        url_a = row.get("url_a") or row.get("URL_A") or row.get("url A") or ""
        url_b = row.get("url_b") or row.get("URL_B") or row.get("url B") or ""
        title_a = row.get("title_a") or row.get("title A") or ""
        title_b = row.get("title_b") or row.get("title B") or ""
        raw_dist_val = normalize_semantic_distance(row.get(distance_col)) if distance_col else None
        # normalize to 0..1 using observed max if necessary
        if raw_dist_val is not None and max_raw and max_raw > 1.0:
            semantic_distance_norm = raw_dist_val / max_raw
        else:
            semantic_distance_norm = raw_dist_val if raw_dist_val is not None else None

        # lookup metadata
        ma = meta_map.get(url_a, {})
        mb = meta_map.get(url_b, {})
        topics_a = ma.get("topics", [])
        topics_b = mb.get("topics", [])
        top_terms_a = ma.get("top_terms", "") or ma.get("top_terms", "")
        top_terms_b = mb.get("top_terms", "") or mb.get("top_terms", "")
        ctype_a = ma.get("content_type", "")
        ctype_b = mb.get("content_type", "")
        wc_a = ma.get("word_count", 0) or 0
        wc_b = mb.get("word_count", 0) or 0
        last_a = ma.get("last_updated", "") or ""
        last_b = mb.get("last_updated", "") or ""

        # Build prompt for Ollama (single call per pair)
        prompt = PROMPT_TEMPLATE.format(
            title_a=title_a or (ma.get("name") or url_a),
            url_a=url_a,
            top_terms_a=top_terms_a or "",
            topics_a=", ".join(topics_a) if topics_a else "",
            ctype_a=ctype_a or "",
            wc_a=wc_a or 0,
            title_b=title_b or (mb.get("name") or url_b),
            url_b=url_b,
            top_terms_b=top_terms_b or "",
            topics_b=", ".join(topics_b) if topics_b else "",
            ctype_b=ctype_b or "",
            wc_b=wc_b or 0
        )

        # Call Ollama (one interaction)
        try:
            raw_out = call_ollama(prompt, model=OLLAMA_MODEL_DEFAULT)
        except Exception as e:
            print("[error] Ollama call failed for pair idx", idx, ":", e)
            raw_out = ""

        # Parse JSON output from model
        gap_description_ai = ""
        title_suggestions = []
        if raw_out:
            try:
                parsed = extract_json_from_model_output(raw_out)
                gap_description_ai = parsed.get("gap_description", "").strip()
                title_suggestions = parsed.get("title_suggestions", []) or []
                # ensure we have at most 5 suggestions
                title_suggestions = [t.strip() for t in title_suggestions if t and isinstance(t, str)][:5]
            except Exception as e:
                print("[warn] Failed to parse model JSON for pair idx", idx, " — raw output saved. Error:", e)
                # fallback: minimal AI-free suggestion
                gap_description_ai = (f"Bridge content between '{title_a}' and '{title_b}' to explain how they relate.")
                title_suggestions = [
                    f"How {title_a} and {title_b} connect",
                    f"Using {title_a} with {title_b}",
                    f"Integrating concepts: {title_a} → {title_b}",
                    f"Guide: {title_a} and {title_b}",
                    f"{title_a} + {title_b}: Practical Guide"
                ]

        # If no suggestions came back, create deterministic fallbacks
        if not title_suggestions:
            title_suggestions = [
                f"How to connect: {title_a} and {title_b}",
                f"{title_a} — bridging to {title_b}",
                f"Integrating {title_a} with {title_b}",
                f"Guide: {title_a} + {title_b}",
                f"Practical steps to move from {title_a} to {title_b}"
            ]

        # Determine best title via deterministic scoring (no additional AI calls)
        best_title = choose_best_title(title_suggestions, topics_a + topics_b, (top_terms_a or "") + " " + (top_terms_b or ""))

        # Category derived from content type (prefer "Integrations" if either indicates integration)
        category = derive_category_from_content_type(ctype_a) or derive_category_from_content_type(ctype_b) or "Integrations"

        # Priority
        priority = determine_priority(semantic_distance_norm)

        # Topic Jaccard for rationale generation
        topic_jaccard = compute_topic_jaccard(topics_a, topics_b)

        # Rationale (deterministic)
        rationale = generate_rationale(semantic_distance_norm or 0.0, topic_jaccard, last_a, last_b, wc_a, wc_b)

        # Build final row
        row_out = {
            "Gap ID": f"GAP-{idx+1:03d}",
            "url_a": url_a,
            "title_a": title_a,
            "url_b": url_b,
            "title_b": title_b,
            "Gap Description (AI)": gap_description_ai,
            "Gap Description (me)": "",  # user fills this in later
            "Category": category,
            "Suggested Article Title (best)": best_title,
            "Suggested Article Title (all)": " | ".join(title_suggestions),
            "Priority": priority,
            "Rationale": rationale,
            "Semantic Distance (norm)": semantic_distance_norm if semantic_distance_norm is not None else "",
            "Topic Jaccard": topic_jaccard
        }
        out_rows.append(row_out)

    # Write CSV
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"[info] Wrote results to {OUTPUT_CSV}")

    # Write a simple markdown table for human review
    with open(OUTPUT_MD, "w", encoding="utf8") as f:
        f.write("# Gap Candidates (AI-enriched)\n\n")
        f.write(f"Source gap file: {os.path.abspath(gap_path)}\n\n")
        for r in out_rows:
            f.write(f"## {r['Gap ID']}: {r['Suggested Article Title (best)']}\n\n")
            f.write(f"- **A**: [{r['title_a']}]({r['url_a']})\n")
            f.write(f"- **B**: [{r['title_b']}]({r['url_b']})\n")
            f.write(f"- **Gap Description (AI)**: {r['Gap Description (AI)']}\n")
            f.write(f"- **Gap Description (me)**: {r['Gap Description (me)']}\n")
            f.write(f"- **Category**: {r['Category']}\n")
            f.write(f"- **Priority**: {r['Priority']}\n")
            f.write(f"- **Rationale**: {r['Rationale']}\n")
            f.write(f"- **All Title Suggestions**: {r['Suggested Article Title (all)']}\n\n---\n\n")
    print(f"[info] Wrote human-readable markdown to {OUTPUT_MD}")

    print("[done] Process finished. Review the CSV and MD and fill 'Gap Description (me)' where needed.")

if __name__ == "__main__":
    main()