import json
import os
from pathlib import Path

import pandas as pd

# ---------------- CONFIG ----------------

INPUT_CSV = "assignments/zb_assgn/data/file_content_populated.csv"
OUTPUT_CSV = "assignments/zb_assgn/data/gap analysis results/file_content_submission.csv"

OLLAMA_MODEL = "mistral:7b"

# hard cap to avoid prompt blowups
MAX_CONTENT_CHARS = 3500

# ---------------- LLM ----------------

def call_ollama(prompt: str) -> str:
    """
    Calls Ollama via LangChain.
    """
    from langchain_community.llms import Ollama

    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0.0
    )
    return str(llm(prompt))


def extract_text_block(raw: str) -> str:
    """
    Defensive cleanup: return plain text even if model adds noise.
    """
    if not raw:
        return ""
    return raw.strip()


# ---------------- PROMPT ----------------

GAP_PROMPT_TEMPLATE = """
You are a senior technical documentation reviewer.

Your task:
Analyze the article content below and identify gaps such as:
- Missing explanations
- Assumed prior knowledge
- Unanswered "how" or "why" questions
- Steps that are mentioned but not explained clearly
- Areas where examples, screenshots, or clarification would help

Rules:
- DO NOT rewrite the article
- DO NOT summarize
- ONLY list gaps
- Be specific and actionable
- If the article is complete, explicitly say "No major gaps identified."

Return format:
A concise bullet-style list, each point on a new line.

Article Title: {title}
Category: {category}
Content Type: {content_type}

Article Content:
{content}
"""


# ---------------- MAIN ----------------

def main():
    if not Path(INPUT_CSV).exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

    output_rows = []

    for idx, row in df.iterrows():
        article_id = row.get("Article ID", "")
        title = row.get("Name", "")
        category = row.get("Category", "")
        url = row.get("URL", "")
        last_updated = row.get("Last Updated", "")
        topics = row.get("Topics Covered", "")
        content_type = row.get("Content Type", "")
        word_count = row.get("Word Count", "")
        has_screenshots = row.get("Has Screenshots", "")

        content = row.get("Cleaned Article Content", "")
        content = content[:MAX_CONTENT_CHARS]

        if not content.strip():
            gaps_identified = "No content available to analyze."
        else:
            prompt = GAP_PROMPT_TEMPLATE.format(
                title=title,
                category=category,
                content_type=content_type,
                content=content
            )

            try:
                raw_output = call_ollama(prompt)
                gaps_identified = extract_text_block(raw_output)
            except Exception as e:
                gaps_identified = f"LLM error while analyzing article: {e}"

        output_rows.append({
            "Article ID": article_id,
            "Article Title": title,
            "Category": category,
            "URL": url,
            "Last Updated": last_updated,
            "Topics Covered": topics,
            "Content Type": content_type,
            "Word Count": word_count,
            "Has Screenshots": has_screenshots,
            "Gaps Identified": gaps_identified
        })

        print(f"[processed] {article_id} | {title}")

    out_df = pd.DataFrame(output_rows)
    os.makedirs(Path(OUTPUT_CSV).parent, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[done] Article gap analysis written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()