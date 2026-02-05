# import nltk
# nltk.download("punkt")

import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize

INPUT_CSV = "assignments/zb_assgn/backup/file_content.csv"
OUTPUT_CSV = "assignments/zb_assgn/backup/file_content_cleaned.csv"
COMMON_PHRASES_OUT = "assignments/zb_assgn/backup/common_phrases.txt"

TEXT_COL = "Article Content"
MIN_DOC_FRACTION = 0.9   # phrase must appear in >90% of articles
NGRAM_RANGE = (2, 10)      # bigrams → 10-grams
MIN_PHRASE_FREQ = 5       # safety threshold

# ---------------- Load ----------------

df = pd.read_csv(INPUT_CSV)
docs = df[TEXT_COL].fillna("").astype(str).tolist()
num_docs = len(docs)

print(f"Loaded {num_docs} documents")

# ---------------- Vectorize n-grams ----------------

vectorizer = CountVectorizer(
    ngram_range=NGRAM_RANGE,
    lowercase=True,
    min_df=MIN_DOC_FRACTION
)

X = vectorizer.fit_transform(docs)
phrases = vectorizer.get_feature_names_out()
phrase_counts = X.sum(axis=0).A1

common_phrases = [
    phrase for phrase, count in zip(phrases, phrase_counts)
    if count >= MIN_PHRASE_FREQ
]

print(f"Identified {len(common_phrases)} common boilerplate phrases")

# ---------------- Persist phrase list ----------------

with open(COMMON_PHRASES_OUT, "w", encoding="utf-8") as f:
    for p in sorted(common_phrases, key=len, reverse=True):
        f.write(p + "\n")

print(f"Saved common phrases → {COMMON_PHRASES_OUT}")

# ---------------- Phrase removal ----------------

def remove_phrases(text: str, phrases: list[str]) -> str:
    cleaned = text
    for phrase in phrases:
        # remove phrase as whole word sequence
        pattern = r"\b" + re.escape(phrase) + r"\b"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # normalize whitespace
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()

# ---------------- Clean documents ----------------

df["Cleaned Article Content"] = df[TEXT_COL].apply(
    lambda t: remove_phrases(str(t), common_phrases)
)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Cleaned CSV written to {OUTPUT_CSV}")
