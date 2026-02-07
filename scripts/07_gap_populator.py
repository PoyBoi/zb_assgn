import csv
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------- CONFIG ----------------
GAP_CANDIDATES_PATHS = [
    "assignments/zb_assgn/data/gap_candidates_v2.csv",
]

METADATA_PATHS = [
    "assignments/zb_assgn/data/file_content_populated.csv",
    "assignments/zb_assgn/data/vectors_3d_v2.csv",
]

OUTPUT_CSV = "assignments/zb_assgn/data/gap analysis results/gap_candidates_with_table.csv"
OUTPUT_MD = "assignments/zb_assgn/data/gap analysis results/gap_candidates_with_table.md"
OUT_OUTLINES_JSON = "assignments/zb_assgn/data/gap analysis results/gap_outlines.json"

MAX_GAPS_TO_PROCESS = 5  # number of unique gap pairs to produce
TOP_OUTLINE_COUNT = 2    # top N to enrich with outlines and bodies
OLLAMA_MODEL_DEFAULT = "mistral:7b"

# Maximum characters to include as an excerpt in outputs (not full article)
EXCERPT_CHARS = 300

# ---------------- Helpers ----------------

def load_existing_gap_pairs(output_csv: str) -> set:
    """
    Returns a set of frozensets: {frozenset({url_a, url_b})}
    """
    if not output_csv or not Path(output_csv).exists():
        return set()

    df = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
    pairs = set()
    for _, r in df.iterrows():
        a = (r.get("url_a") or "").strip()
        b = (r.get("url_b") or "").strip()
        if a and b and a != b:
            pairs.add(frozenset((a, b)))
    return pairs

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

def load_metadata():
    p = find_first_existing(METADATA_PATHS)
    if not p:
        print("[warn] No metadata CSV found in expected paths; continuing with minimal metadata.")
        return pd.DataFrame(), None
    print(f"[info] Loading metadata from: {p}")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return df, p

def build_meta_map(meta_df):
    """Build a lookup map keyed by URL with fields including cleaned article content and content type."""
    meta = {}
    if meta_df is None or meta_df.empty:
        return meta
    cols = {c.lower().strip(): c for c in meta_df.columns}

    def get_val(row, *keys):
        for k in keys:
            if k.lower() in cols:
                return row[cols[k.lower()]]
        return ""

    for _, r in meta_df.iterrows():
        url = get_val(r, "url", "URL", "link")
        if not url:
            continue
        cleaned = get_val(r, "cleaned article content", "cleaned_content", "content", "body")
        wc = 0
        try:
            wc = len(re.findall(r"\w+", cleaned))
        except Exception:
            wc = 0
        meta[url] = {
            "name": get_val(r, "name", "title", "Name"),
            "content_type": get_val(r, "content type", "content_type", "ContentType"),
            "last_updated": get_val(r, "last updated", "last_updated", "updated"),
            # keep the cleaned content internally for prompting only; do NOT write full content to outputs
            "_cleaned_content": cleaned,
            "top_terms": get_val(r, "top_terms", "top terms", "top_terms") or "",
            "word_count": wc
        }
    return meta

# ---------------- Ollama wrapper (unchanged) ----------------

def call_ollama_cli(prompt_text, model=OLLAMA_MODEL_DEFAULT):
    cmd = ["ollama", "generate", model, "--prompt", prompt_text]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise FileNotFoundError("The 'ollama' CLI is not found in PATH. Install ollama or make it available.") from e
    out = res.stdout.strip()
    if not out:
        out = res.stderr.strip()
    return out

def call_ollama(prompt_text, model=OLLAMA_MODEL_DEFAULT):
    # Try LangChain Ollama then CLI (keep compatibility)
    try:
        from langchain_community.llms import Ollama
        try:
            llm = Ollama(model=model, temperature=0.0)
            resp = llm(prompt_text)
            return str(resp)
        except TypeError:
            llm = Ollama(model=model)
            resp = llm(prompt_text)
            return str(resp)
    except Exception:
        return call_ollama_cli(prompt_text, model=model)

# ---------------- JSON extraction helper ----------------

def extract_json_from_model_output(s: str):
    if not s or not s.strip():
        raise ValueError("Empty model output")
    try:
        return json.loads(s)
    except Exception:
        pass
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    raise ValueError("Could not parse JSON from model output. Raw output preview:\n" + (s[:2000] if len(s) > 2000 else s))

# ---------------- Prompt templates ----------------
PROMPT_TEMPLATE_PAIR = """
You are a concise assistant that helps product/documentation teams identify article gaps and propose titles.


Input (do NOT invent facts; only use the explicit content below):
Article A Title: {title_a}
Article A URL: {url_a}
Article A Cleaned Content:
{content_a}


Article B Title: {title_b}
Article B URL: {url_b}
Article B Cleaned Content:
{content_b}


Task (single JSON response ONLY):
1) Provide a one-sentence "gap_description" that describes the gap a NEW article should fill to connect A and B. Keep it concise (<= 30 words).
2) Provide "title_suggestions": an array of 5 short title suggestions (phrases), ordered best-to-worst.


Return exactly a JSON object with keys:
{{
"gap_description": "<one concise sentence>",
"title_suggestions": ["title 1", "title 2", "title 3", "title 4", "title 5"]
}}


Do not include any other text, explanations, or markup.
"""

PROMPT_TEMPLATE_OUTLINE = """
You are a documentation writer. Given Article A and B and the AI-provided gap description, produce an article OUTLINE that will best fill that gap.


Article A: {title_a} ({url_a})
Article B: {title_b} ({url_b})
Gap summary: {gap_description}


Return exactly JSON with key "outline" whose value is an array of section objects in order. Each section object must have:
{{
"section_title": "<short section heading>",
"section_summary": "<one-sentence summary of what this section will cover>",
"section_bullets": ["short bullet 1", "short bullet 2", ...]
}}


Aim for between 5 and 9 sections. Keep summaries concise (<=25 words). Return ONLY the JSON object.
"""


PROMPT_TEMPLATE_EXPAND = """
You are a documentation writer. Given the article OUTLINE (JSON) below, expand it into a detailed article BODY in Markdown.


Outline JSON:
{outline_json}


Return exactly a JSON object:
{{
"body_markdown": "<full article body in Markdown, with headings matching the outline, section content, code blocks if needed, examples, and a short conclusion>"
}}


Make the body practical and instructional. Do not include any text outside the JSON object.
"""

# ---------------- Utility selection helpers ----------------

def derive_category_from_content_type(ctype: str):
    if not ctype or str(ctype).strip() == "":
        return "Integrations"
    c = str(ctype).lower()
    if "integrat" in c:
        return "Integrations"
    if "how-to" in c or "how to" in c or "guide" in c:
        return "How-to / Guide"
    if "troubleshoot" in c or "troubleshooting" in c:
        return "Troubleshooting"
    if "api" in c or "developer" in c:
        return "API / Developer"
    return str(ctype).strip().title()

def choose_best_title(suggestions, content_a, content_b):
    # Simple heuristic scoring: prefer short titles that share tokens with combined contents
    combined = (content_a or "") + "\n" + (content_b or "")
    combined_lower = combined.lower()
    best = suggestions[0] if suggestions else ""
    best_score = -1
    for t in suggestions:
        score = 0
        tl = t.lower()
        # length bonus for brevity
        if len(t) <= 60:
            score += 1.0
        # presence of meaningful words from title in content
        toks = re.findall(r"\w{4,}", tl)
        for tok in toks:
            if tok in combined_lower:
                score += 2.0
        # small deterministic tie-breaker
        score += (hash(t) % 10) * 1e-6
        if score > best_score:
            best_score = score
            best = t
    return best

# ---------------- Main flow ----------------
def main():
    gap_df, gap_path = load_gap_candidates()
    if gap_df.empty:
        print("[info] gap_candidates.csv is empty. Exiting.")
        return
    
    existing_pairs = load_existing_gap_pairs(OUTPUT_CSV)
    if existing_pairs:
        print(f"[info] Incremental mode: {len(existing_pairs)} gap pairs already processed")

    meta_df, meta_path = load_metadata()
    meta_map = build_meta_map(meta_df)

    # We'll produce enriched rows for top MAX_GAPS_TO_PROCESS, ensuring no article repeats
    n_total = len(gap_df)
    print(f"[info] Available gap rows: {n_total}")

    enriched_rows = []
    outlines_collection = []

    used_urls = set()
    processed = 0


    for idx, row in gap_df.iterrows():
        if processed >= MAX_GAPS_TO_PROCESS:
            break
        url_a = (row.get("url_a") or row.get("URL_A") or row.get("urlA") or "").strip()
        url_b = (row.get("url_b") or row.get("URL_B") or row.get("urlB") or "").strip()

        pair_key = frozenset((url_a, url_b))
        if pair_key in existing_pairs:
            continue

        if not url_a or not url_b or url_a == url_b:
            continue
        # avoid repeating any article (either as A or B) across processed pairs
        if url_a in used_urls or url_b in used_urls:
            continue

        title_a = (row.get("title_a") or row.get("title A") or row.get("titleA") or meta_map.get(url_a, {}).get("name") or url_a)
        title_b = (row.get("title_b") or row.get("title B") or row.get("titleB") or meta_map.get(url_b, {}).get("name") or url_b)

        content_a = meta_map.get(url_a, {}).get("_cleaned_content", "")
        content_b = meta_map.get(url_b, {}).get("_cleaned_content", "")

        ctype_a = meta_map.get(url_a, {}).get("content_type", "")
        ctype_b = meta_map.get(url_b, {}).get("content_type", "")

        wc_a = meta_map.get(url_a, {}).get("word_count", 0)
        wc_b = meta_map.get(url_b, {}).get("word_count", 0)

        # Compose prompt for gap detection + title suggestions
        prompt_pair = PROMPT_TEMPLATE_PAIR.format(
            title_a=title_a,
            url_a=url_a,
            content_a=content_a[:3000],  # guard prompt length
            title_b=title_b,
            url_b=url_b,
            content_b=content_b[:3000]
        )

        gap_desc_ai = ""
        title_suggestions = []
        try:
            out_pair_raw = call_ollama(prompt_pair, model=OLLAMA_MODEL_DEFAULT)
            parsed_pair = extract_json_from_model_output(out_pair_raw)
            gap_desc_ai = parsed_pair.get("gap_description", "").strip()
            title_suggestions = parsed_pair.get("title_suggestions", []) or []
            title_suggestions = [t.strip() for t in title_suggestions if isinstance(t, str)][:5]
        except Exception as e:
            print(f"[warn] Pair AI call failed for idx {idx}: {e}")

        if not title_suggestions:
            # deterministic fallbacks
            title_suggestions = [
                f"How to connect: {title_a} and {title_b}",
                f"{title_a} — bridging to {title_b}",
                f"Integrating {title_a} with {title_b}",
                f"Guide: {title_a} + {title_b}",
                f"Practical steps to move from {title_a} to {title_b}",
            ]

        best_title = choose_best_title(title_suggestions, content_a, content_b)

        # Simple reason for selected title (short heuristic)
        reason_for_title = f"Selected because it is concise and uses terms present in both articles." if best_title else ""

        category = derive_category_from_content_type(ctype_a) or derive_category_from_content_type(ctype_b) or "Integrations"

        # Rationale: ask the model to explain why the gap should be filled, based on the contents
        prompt_rationale = (
            "Act as a senior documentation editor. Given the short gap summary and the (truncated) article contents, "
            "produce a single, concise sentence that explains, in third person, why this gap matters for users. "
            "Focus on user impact (task completion, clarity, risk, or support load). "
            "Do NOT use first-person or second-person pronouns (do not use 'I', 'we', or 'you'). "
            "Do NOT mention the model, assistant, prompts, or internal process. "
            "Do NOT include anything other than the single sentence.\n\n"
            f"Gap summary: {gap_desc_ai or '(bridge between A and B)'}\n\n"
            "Article A :\n" + (content_a or "(no content)") + "\n\n"
            "Article B :\n" + (content_b or "(no content)") + "\n\n"
            "Return exactly one sentence, not more than that."
        )

        rationale = ""
        try:
            out_rat = call_ollama(prompt_rationale, model=OLLAMA_MODEL_DEFAULT)
            # Model may return plain text; take first line or parse JSON if provided
            rationale = out_rat.strip().splitlines()[0]
            # trim long responses
            # if len(rationale) > 300:
            #     rationale = rationale[:297] + "..."
        except Exception as e:
            print(f"[warn] Rationale AI call failed for idx {idx}: {e}")
            # heuristic rationale
            rationale = "Bridging the two articles will provide a focused how-to and examples connecting concepts in A and B."

        # Build safe excerpts (do NOT include full article content in outputs)
        excerpt_a = (content_a[:EXCERPT_CHARS] + "...") if content_a and len(content_a) > EXCERPT_CHARS else content_a
        excerpt_b = (content_b[:EXCERPT_CHARS] + "...") if content_b and len(content_b) > EXCERPT_CHARS else content_b

        enriched = {
            "Gap ID": f"GAP-{processed+1:03d}",
            "url_a": url_a,
            "title_a": title_a,
            "Content Excerpt A": excerpt_a,
            "Word Count A": wc_a,
            "url_b": url_b,
            "title_b": title_b,
            "Content Excerpt B": excerpt_b,
            "Word Count B": wc_b,
            "Gap Description (by AI)": gap_desc_ai,
            "Gap Description (by me)": "",
            "Category": category,
            "Suggested Article Title (s)": " | ".join(title_suggestions),
            "Suggested Article Title": best_title,
            "Reason for Article Title": reason_for_title,
            "Priority": "Medium",
            "Rationale": rationale,
        }

        enriched_rows.append(enriched)

        # For top outlines, generate structured outline + expanded body
        if processed < TOP_OUTLINE_COUNT:
            try:
                prompt_outline = PROMPT_TEMPLATE_OUTLINE.format(
                    title_a=title_a,
                    url_a=url_a,
                    title_b=title_b,
                    url_b=url_b,
                    gap_description=gap_desc_ai or "(bridge between the two articles)"
                )
                out_outline_raw = call_ollama(prompt_outline, model=OLLAMA_MODEL_DEFAULT)
                outline_json = extract_json_from_model_output(out_outline_raw).get("outline", [])
            except Exception as e:
                print(f"[warn] Outline generation failed for gap idx {idx}: {e}")
                outline_json = []

            # Expand outline into article body
            body_markdown = ""
            if outline_json:
                try:
                    outline_json_str = json.dumps(outline_json, ensure_ascii=False)
                    prompt_expand = PROMPT_TEMPLATE_EXPAND.format(outline_json=outline_json_str)
                    out_expand_raw = call_ollama(prompt_expand, model=OLLAMA_MODEL_DEFAULT)
                    expand_parsed = extract_json_from_model_output(out_expand_raw)
                    body_markdown = expand_parsed.get("body_markdown", "").strip()
                except Exception as e:
                    print(f"[warn] Outline expansion failed for gap idx {idx}: {e}")

            outlines_collection.append({
                "Gap ID": enriched["Gap ID"],
                "Suggested Title": best_title,
                "Outline": outline_json,
                "Body Markdown": body_markdown
            })

        # mark urls used and increment processed count
        used_urls.add(url_a)
        used_urls.add(url_b)
        processed += 1

    # Write enriched CSV (flatten long text safely)
    df_out = pd.DataFrame(enriched_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # df_out.to_csv(OUTPUT_CSV, index=False)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    write_header = not Path(OUTPUT_CSV).exists()
    df_out.to_csv(
        OUTPUT_CSV,
        mode="a" if not write_header else "w",
        header=write_header,
        index=False
    )

    print(f"[info] Appended {len(df_out)} enriched gaps to {OUTPUT_CSV}")

    
    print(f"[info] Wrote enriched gap table to {OUTPUT_CSV}")

    # Write markdown summary (do NOT include full article content)
    
    # mode = "a" if Path(OUTPUT_MD).exists() else "w"
    # with open(OUTPUT_MD, mode, encoding="utf8") as f:
    
    with open(OUTPUT_MD, "w", encoding="utf8") as f:
        # if mode == "w":
        f.write("# Gap Candidates (AI-enriched)\n\n")
        f.write(f"Source gap file: {os.path.abspath(gap_path)}\n\n")

        f.write("# Gap Candidates (AI-enriched)\n\n")
        f.write(f"Source gap file: {os.path.abspath(gap_path)}\n\n")
        for r in enriched_rows:
            f.write(f"## {r['Gap ID']}: {r['Suggested Article Title']}\n\n")
            f.write(f"- **A**: [{r['title_a']}]({r['url_a']})\n")
            f.write(f"  - Excerpt: {r['Content Excerpt A']}\n")
            f.write(f"  - Word count: {r['Word Count A']}\n")
            f.write(f"- **B**: [{r['title_b']}]({r['url_b']})\n")
            f.write(f"  - Excerpt: {r['Content Excerpt B']}\n")
            f.write(f"  - Word count: {r['Word Count B']}\n")
            f.write(f"- **Gap Description (AI)**: {r['Gap Description (by AI)']}\n")
            f.write(f"- **Gap Description (me)**: {r['Gap Description (by me)']}\n")
            f.write(f"- **Category**: {r['Category']}\n")
            f.write(f"- **Suggested Titles**: {r['Suggested Article Title (s)']}\n")
            f.write(f"- **Chosen Title**: {r['Suggested Article Title']}\n")
            f.write(f"- **Reason for Title**: {r['Reason for Article Title']}\n")
            f.write(f"- **Priority**: {r['Priority']}\n")
            f.write(f"- **Rationale**: {r['Rationale']}\n\n---\n\n")
    print(f"[info] Wrote human-readable markdown to {OUTPUT_MD}")

    # Write outlines JSON
    
    # with open(OUT_OUTLINES_JSON, "w", encoding="utf8") as f:
    #     json.dump(outlines_collection, f, indent=2, ensure_ascii=False)

    existing_outlines = []
    if Path(OUT_OUTLINES_JSON).exists():
        try:
            existing_outlines = json.loads(Path(OUT_OUTLINES_JSON).read_text(encoding="utf8"))
        except Exception:
            existing_outlines = []

    merged_outlines = existing_outlines + outlines_collection

    with open(OUT_OUTLINES_JSON, "w", encoding="utf8") as f:
        json.dump(merged_outlines, f, indent=2, ensure_ascii=False)

    print(f"[info] Appended {len(outlines_collection)} outlines to {OUT_OUTLINES_JSON}")

        
    print(f"[info] Wrote outlines JSON to {OUT_OUTLINES_JSON}")

    print("[done] All outputs written. Excerpts (not full content) were included in outputs as requested.")

if __name__ == "__main__":
    main()