#!/usr/bin/env python3
"""
vectorise_and_find_gaps_full.py

Full pipeline:
 - read CSV (prefers file_content_cleaned.csv; falls back to file_content.csv)
 - augment site_map from csv 'Linked Pages'
 - compute embeddings (SBERT if available; TF-IDF fallback)
 - compute semantic distances (cosine)
 - reduce to 3D with UMAP for visualization
 - cluster optionally (KMeans)
 - compute top terms per doc (TF-IDF)
 - score unlinked pairs using multiple signals and save top-K
 - produce interactive HTML visualizations (plotly)
"""

import argparse
import json
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans

# UMAP and Plotly are used for visualization
try:
    import umap
except Exception as e:
    umap = None
    print("[warn] umap not available:", e)

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception as e:
    px = None
    go = None
    print("[warn] plotly not available:", e)

# Optional: networkx for PageRank
try:
    import networkx as nx
except Exception:
    nx = None

# ---------------- Configuration / Defaults ----------------
DEFAULT_INPUT_CSVS = [
    "assignments/zb_assgn/data/file_content_populated.csv",
    # "assignments/zb_assgn/data/file_content.csv",
    # "file_content_cleaned.csv",
    # "file_content.csv"
]
SITE_MAP_FILE = "assignments/zb_assgn/data/site_map.json"
OUT_VECTORS_CSV = "assignments/zb_assgn/data/vectors_3d_v2.csv"
OUT_GAPS_CSV = "assignments/zb_assgn/data/gap_candidates_v2.csv"
OUT_HTML = "assignments/zb_assgn/data/site_vectors_3d_v2.html"
OUT_HTML_TOGGLES = "assignments/zb_assgn/data/map_with_toggles_v2.html"

TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 2
UMAP_N_COMPONENTS = 3
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42
KMEANS_CLUSTERS = 10  # set 0 or None to skip clustering
TOP_TERMS_PER_DOC = 8

TOP_K_GAPS = 200  # top K furthest unlinked pairs to save
MAX_PAIRWISE_N_FOR_EXACT = 3000  # above this, warn about O(N^2) cost; sample if requested

# Default scoring weights (can be overridden by CLI)
DEFAULT_WEIGHTS = {
    "w_sem": 0.50,
    "w_topic": 0.15,
    "w_recency": 0.15,
    "w_media": 0.10,
    "w_word": 0.05,
    "w_central": 0.05
}


# ---------------- Helpers ----------------
def load_input_csv(default_paths=DEFAULT_INPUT_CSVS):
    for p in default_paths:
        if Path(p).exists():
            print(f"[info] Loading {p}")
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            return df, p
    raise FileNotFoundError(f"None of the expected input CSVs found: {default_paths}")


def select_text_column(df: pd.DataFrame):
    # prefer cleaned column
    for col in ["Cleaned Article Content", "Article Content", "Article_Content", "content", "Body", "body"]:
        if col in df.columns:
            print(f"[info] Using text column: {col}")
            return col
    raise ValueError("No article content column found. Expected 'Cleaned Article Content' or 'Article Content'.")


def load_site_map(path=SITE_MAP_FILE):
    p = Path(path)
    if not p.exists():
        print(f"[warn] site_map not found at {path}. Proceeding with empty map (no links).")
        return {}
    try:
        return json.loads(p.read_text(encoding="utf8"))
    except Exception as e:
        print("[warn] failed to load site_map.json:", e)
        return {}


def has_link(site_map, a, b):
    # check both directions quickly
    links_a = site_map.get(a, []) or []
    if b in links_a:
        return True
    links_b = site_map.get(b, []) or []
    if a in links_b:
        return True
    return False


def parse_topics(series):
    """
    Parse 'Topics Covered' (a series of strings) into list-of-sets (lowercased tokens).
    Accepts separators: comma, semicolon, pipe.
    """
    out = []
    for s in series.fillna("").astype(str):
        s = s.strip()
        if not s:
            out.append(set())
            continue
        parts = [p.strip().lower() for p in re.split(r"[;,|]", s) if p.strip()]
        out.append(set(parts))
    return out


def augment_site_map_with_csv_links(df, site_map):
    """
    Use CSV column 'Linked Pages' to add outgoing links into site_map.
    Accepts comma-separated URLs in the cell.
    """
    for _, row in df.iterrows():
        src = row.get("URL", "")
        if not src:
            continue
        raw = row.get("Linked Pages", "")
        if not raw:
            continue
        # split by comma, semicolon, or pipe
        parts = [t.strip() for t in re.split(r"[,;|]", str(raw)) if t.strip()]
        if not parts:
            continue
        site_map.setdefault(src, [])
        for t in parts:
            if t not in site_map[src]:
                site_map[src].append(t)
    return site_map


def media_flag_from_value(v):
    if v is None:
        return False
    v = str(v).strip().lower()
    if v in ["", "no", "false", "nan", "none", "0"]:
        return False
    return True


def compute_page_centrality(site_map):
    """
    Compute PageRank scores if networkx available; otherwise return empty dict.
    """
    if nx is None:
        print("[warn] networkx not available; skipping centrality computation.")
        return {}
    G = nx.DiGraph()
    for src, targets in site_map.items():
        for t in targets:
            G.add_edge(src, t)
    if len(G) == 0:
        return {}
    try:
        pr = nx.pagerank(G)
        return pr
    except Exception as e:
        print("[warn] PageRank computation failed:", e)
        return {}


def get_embeddings(docs, use_sbert=True, tfidf_max_features=TFIDF_MAX_FEATURES, tfidf_min_df=TFIDF_MIN_DF):
    """
    Returns (embeddings_array, embedding_source, tfidf_vectorizer_if_used_or_None)

    - embeddings_array: numpy array shape (N, D)
    - embedding_source: string "sbert" or "tfidf"
    - tfidf_vectorizer_if_used: if embeddings come from tfidf, returns that vectorizer (useful for top terms)
    """
    if use_sbert:
        try:
            from sentence_transformers import SentenceTransformer
            print("[step] computing sentence-transformers embeddings (all-MiniLM-L6-v2)...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
            emb = np.asarray(emb, dtype=np.float32)
            return emb, "sbert", None
        except Exception as e:
            print("[warn] sentence-transformers unavailable or failed; falling back to TF-IDF embeddings. Error:", e)

    # TF-IDF fallback
    print("[step] computing TF-IDF embeddings (dense) as fallback...")
    vec = TfidfVectorizer(max_features=tfidf_max_features, min_df=tfidf_min_df, stop_words="english")
    X = vec.fit_transform(docs)
    X_dense = X.toarray().astype(np.float32)
    return X_dense, "tfidf", vec


def compute_semantic_distance_matrix(embeddings):
    """
    Compute cosine distances matrix (0..2) and normalized 0..1 matrix (divide by max observed).
    Returns (dist_raw, dist_norm)
    """
    print("[step] computing cosine distance matrix...")
    if embeddings is None or len(embeddings) == 0:
        return None, None
    # compute cosine_distances (0..2); convert to float64
    dist = cosine_distances(embeddings).astype(np.float64)
    # clamp diagonal to 0
    np.fill_diagonal(dist, 0.0)
    maxv = float(np.nanmax(dist))
    if maxv <= 0 or not np.isfinite(maxv):
        dist_norm = dist.copy()
    else:
        dist_norm = dist / maxv
    return dist, dist_norm


def compute_top_terms_tfidf(docs, max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF, top_n=TOP_TERMS_PER_DOC):
    """
    Compute TF-IDF vectorizer fit on docs and return a list of comma-separated top terms per document.
    """
    print("[step] computing TF-IDF for top terms...")
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df, stop_words="english")
    X = vec.fit_transform(docs)
    feature_names = vec.get_feature_names_out()
    top_terms = []
    for i in range(X.shape[0]):
        row = X[i]
        try:
            arr = row.toarray().ravel()
        except Exception:
            arr = np.asarray(row).ravel()
        if arr.sum() == 0:
            top_terms.append("")
            continue
        idxs = np.argsort(-arr)[:top_n]
        terms = [feature_names[idx] for idx in idxs if arr[idx] > 0]
        top_terms.append(", ".join(terms))
    return top_terms, vec


# ---------------- Scoring ----------------
def score_pair(i, j, df_meta, dist_norm, topics_list, pr_map, weights, now_dt):
    """
    Score pair (i,j) using precomputed normalized distances (0..1), topics (list of sets),
    page rank map (url -> score), and df_meta for recency/media/wordcount.
    """
    # semantic distance normalized 0..1
    sem = float(dist_norm[i, j]) if dist_norm is not None else 0.0

    # topic overlap Jaccard
    A = topics_list[i]; B = topics_list[j]
    if not A and not B:
        topic_j = 0.0
    else:
        union = len(A | B)
        if union == 0:
            topic_j = 0.0
        else:
            topic_j = len(A & B) / union
    topic_component = 1.0 - topic_j  # higher when topics differ (complementary)

    # recency: prefer newer -> older
    def parse_date_safe(s):
        if s is None:
            return None
        s = str(s).strip()
        if not s:
            return None
        try:
            # let pandas parse various formats
            dt = pd.to_datetime(s, utc=True)
            return dt
        except Exception:
            try:
                dt = datetime.fromisoformat(s)
                return pd.to_datetime(dt, utc=True)
            except Exception:
                return None

    da = parse_date_safe(df_meta.iloc[i].get("Last Updated", None))
    db = parse_date_safe(df_meta.iloc[j].get("Last Updated", None))
    recency_boost = 0.0
    if da is not None and db is not None:
        # positive when a is newer than b
        days_diff = (da - db).days
        recency_boost = max(days_diff, 0) / (abs(days_diff) + 30.0)
    # else 0

    # media complementarity: XOR across screenshot/youtube/image
    def media_flags(row):
        return {
            "screens": media_flag_from_value(row.get("Has Screenshots", "")),
            "youtube": media_flag_from_value(row.get("YouTube Links", "")),
            "images": media_flag_from_value(row.get("Image Links", ""))
        }

    ma = media_flags(df_meta.iloc[i])
    mb = media_flags(df_meta.iloc[j])
    media_xor = sum(int(ma[k]) ^ int(mb[k]) for k in ["screens", "youtube", "images"])
    media_comp = media_xor / 3.0

    # word count difference (log scaled)
    def to_float_safe(x):
        try:
            return float(str(x).strip())
        except Exception:
            return 0.0

    wa = to_float_safe(df_meta.iloc[i].get("Word Count", 0))
    wb = to_float_safe(df_meta.iloc[j].get("Word Count", 0))
    if wa > 0 and wb > 0:
        wc = abs(math.log(wa + 1) - math.log(wb + 1))
        wc_norm = wc / (wc + 1.0)
    else:
        wc_norm = 0.0

    # centrality (prefer central -> peripheral)
    pa = pr_map.get(df_meta.iloc[i].get("URL", ""), 0.0)
    pb = pr_map.get(df_meta.iloc[j].get("URL", ""), 0.0)
    central_diff = max(pa - pb, 0.0)

    s = (
        weights["w_sem"] * sem +
        weights["w_topic"] * topic_component +
        weights["w_recency"] * recency_boost +
        weights["w_media"] * media_comp +
        weights["w_word"] * wc_norm +
        weights["w_central"] * central_diff
    )
    return float(s)


def find_top_unlinked_pairs_scored(df_meta, dist_norm, site_map, top_k=TOP_K_GAPS, weights=None, max_pairs_warning=MAX_PAIRWISE_N_FOR_EXACT):
    """
    Scores all unlinked pairs (i<j) and returns top_k as DataFrame.
    dist_norm is normalized semantic distance matrix (0..1).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    n = dist_norm.shape[0]
    if n <= 1:
        return pd.DataFrame([], columns=["url_a", "title_a", "url_b", "title_b", "score", "semantic_distance"])

    print("[step] preparing topics and centrality for scoring...")
    topics_list = parse_topics(df_meta.get("Topics Covered", pd.Series([""] * n)))
    pr_map = compute_page_centrality(site_map)
    now_dt = datetime.utcnow()

    # Choose compute strategy: full O(N^2) if n reasonable, else warn
    pair_count = n * (n - 1) // 2
    if pair_count > max_pairs_warning * max_pairs_warning:
        print(f"[warn] Pair count {pair_count} looks very large. Consider using --approximate sampling or raising resources.")

    results = []
    idx_i, idx_j = np.triu_indices(n, k=1)
    for pos in range(len(idx_i)):
        i = int(idx_i[pos]); j = int(idx_j[pos])
        a_url = df_meta.iloc[i].get("URL", "")
        b_url = df_meta.iloc[j].get("URL", "")
        # Skip if urls already linked
        if has_link(site_map, a_url, b_url):
            continue
        s = score_pair(i, j, df_meta, dist_norm, topics_list, pr_map, weights, now_dt)
        results.append((s, float(dist_norm[i, j]), i, j))

    if not results:
        print("[info] No candidate pairs after excluding existing links.")
        return pd.DataFrame([], columns=["url_a", "title_a", "url_b", "title_b", "score", "semantic_distance"])

    # sort descending by score
    results.sort(key=lambda x: x[0], reverse=True)
    out_rows = []
    for s, semd, i, j in results[:top_k]:
        out_rows.append({
            "url_a": df_meta.iloc[i].get("URL", ""),
            "title_a": df_meta.iloc[i].get("Name", ""),
            "url_b": df_meta.iloc[j].get("URL", ""),
            "title_b": df_meta.iloc[j].get("Name", ""),
            "score": float(s),
            "semantic_distance": float(semd)
        })
    return pd.DataFrame(out_rows)


# ---------------- Visualization (Plotly) ----------------

def plotly_3d(df_meta, coords, out_html=OUT_HTML):
    if px is None:
        print("[warn] plotly not available; skipping HTML plot save.")
        return
    print("[step] generating 3D interactive HTML plot...")
    df_plot = df_meta.copy().reset_index(drop=True)
    df_plot["x"] = coords[:, 0]
    df_plot["y"] = coords[:, 1]
    df_plot["z"] = coords[:, 2]
    df_plot["hover_text"] = df_plot.apply(
        lambda r: f"{r.get('Name','')}<br>{r.get('URL','')}<br>Category: {r.get('Category','')}",
        axis=1
    )
    color_col = "cluster" if "cluster" in df_plot.columns else None
    fig = px.scatter_3d(
        df_plot,
        x="x",
        y="y",
        z="z",
        color=color_col,
        hover_name="Name",
        hover_data={"URL": True, "Category": True},
        title="Semantic 3D map (UMAP on embeddings)",
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color="DarkSlateGrey")), selector=dict(mode="markers"))
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                      margin=dict(t=40, l=0, r=0, b=0), height=900)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[info] 3D plot saved to {out_html}")


def plotly_3d_with_colorful_edges_and_toggle(df_meta, coords, site_map, gaps_df=None, out_html=OUT_HTML_TOGGLES, seed=42):
    if go is None:
        print("[warn] plotly not available; skipping toggleable HTML plot.")
        return
    random.seed(seed)
    np.random.seed(seed)
    df_plot = df_meta.copy().reset_index(drop=True)
    df_plot["x"] = coords[:, 0]
    df_plot["y"] = coords[:, 1]
    df_plot["z"] = coords[:, 2]
    df_plot["hover_text"] = df_plot.apply(
        lambda r: f"<b>{r.get('Name','')}</b><br>{r.get('URL','')}<br>Category: {r.get('Category','')}",
        axis=1
    )

    url_to_coord = {row["URL"]: (row["x"], row["y"], row["z"]) for _, row in df_plot.iterrows() if row.get("URL", "")}

    # existing edges
    existing_edges = []
    for parent, children in site_map.items():
        if parent not in url_to_coord:
            continue
        for ch in children:
            if ch not in url_to_coord:
                continue
            a, b = sorted((parent, ch))
            existing_edges.append((a, b))
    existing_edges = list(dict.fromkeys(existing_edges))
    n_edges = len(existing_edges)
    print(f"[info] existing edges: {n_edges}")

    traces = []
    existing_edge_trace_indices = []
    gap_trace_indices = []

    PER_EDGE_LIMIT = 800
    if n_edges <= PER_EDGE_LIMIT:
        for (a, b) in existing_edges:
            x0, y0, z0 = url_to_coord[a]
            x1, y1, z1 = url_to_coord[b]
            color = "rgb({}, {}, {})".format(*[random.randint(20, 220) for _ in range(3)])
            tr = go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines",
                              line=dict(color=color, width=3), hoverinfo="none", visible=True, showlegend=False)
            existing_edge_trace_indices.append(len(traces))
            traces.append(tr)
    else:
        # group into buckets
        N_BUCKETS = 30
        buckets = [[] for _ in range(N_BUCKETS)]
        for (a, b) in existing_edges:
            idx = (hash(a + "||" + b) % N_BUCKETS)
            buckets[idx].append((a, b))
        for bucket in buckets:
            if not bucket:
                continue
            xs, ys, zs = [], [], []
            color = "rgb({}, {}, {})".format(*[random.randint(20, 220) for _ in range(3)])
            for (a, b) in bucket:
                x0, y0, z0 = url_to_coord[a]
                x1, y1, z1 = url_to_coord[b]
                xs += [x0, x1, None]
                ys += [y0, y1, None]
                zs += [z0, z1, None]
            tr = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                              line=dict(color=color, width=2.5), hoverinfo="none", visible=True, showlegend=False)
            existing_edge_trace_indices.append(len(traces))
            traces.append(tr)

    # gap traces (hidden by default)
    if gaps_df is not None and len(gaps_df) > 0:
        gaps = []
        for _, r in gaps_df.iterrows():
            a = r.get("url_a", ""); b = r.get("url_b", "")
            if a in url_to_coord and b in url_to_coord:
                gaps.append((a, b, float(r.get("score", 0.0))))
        print(f"[info] gap candidate pairs present: {len(gaps)}")
        MAX_GAP_PER_EDGE = 1000
        if len(gaps) <= MAX_GAP_PER_EDGE:
            for (a, b, sc) in gaps:
                x0, y0, z0 = url_to_coord[a]
                x1, y1, z1 = url_to_coord[b]
                # red-ish scaled by score
                rcol = int(150 + min(100, sc * 200))
                color = f"rgb({rcol}, {random.randint(20,80)}, {random.randint(20,80)})"
                tr = go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines",
                                  line=dict(color=color, width=3, dash="dash"), hoverinfo="none", visible=False, showlegend=False)
                gap_trace_indices.append(len(traces))
                traces.append(tr)
        else:
            G_BUCKETS = 20
            buckets = [[] for _ in range(G_BUCKETS)]
            for (a, b, sc) in gaps:
                idx = (hash(a + "||" + b) % G_BUCKETS)
                buckets[idx].append((a, b, sc))
            for bucket in buckets:
                if not bucket:
                    continue
                xs, ys, zs = [], [], []
                color = "rgb({}, {}, {})".format(220, random.randint(30, 100), random.randint(30, 100))
                for (a, b, sc) in bucket:
                    x0, y0, z0 = url_to_coord[a]
                    x1, y1, z1 = url_to_coord[b]
                    xs += [x0, x1, None]
                    ys += [y0, y1, None]
                    zs += [z0, z1, None]
                tr = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                  line=dict(color=color, width=2.5, dash="dash"), hoverinfo="none", visible=False, showlegend=False)
                gap_trace_indices.append(len(traces))
                traces.append(tr)

    # node trace
    if "cluster" in df_plot.columns:
        node_colors = df_plot["cluster"].astype(int)
        show_colorbar = True
    else:
        node_colors = "orange"
        show_colorbar = False

    node_trace = go.Scatter3d(
        x=df_plot["x"], y=df_plot["y"], z=df_plot["z"],
        mode="markers",
        marker=dict(size=7, color=node_colors, colorscale="Viridis", opacity=0.95, showscale=show_colorbar,
                    colorbar=dict(title="Cluster", thickness=15), line=dict(width=1.0, color="black")),
        text=df_plot["hover_text"],
        hoverinfo="text",
        name="Articles",
        visible=True
    )

    node_trace_index = len(traces)
    traces.append(node_trace)

    layout = go.Layout(
        title="Semantic 3D Map — edges colored per connection",
        showlegend=False,
        height=900,
        scene=dict(camera=dict(eye=dict(x=1.2, y=1.2, z=0.9)),
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(t=50, l=0, r=0, b=0),
    )
    fig = go.Figure(data=traces, layout=layout)

    updatemenus = []
    if existing_edge_trace_indices:
        updatemenus.append(dict(type="buttons", direction="left", x=0.02, y=1.02, showactive=False,
                                buttons=[
                                    dict(label="Hide existing connections", method="restyle", args=[{"visible": False}, existing_edge_trace_indices]),
                                    dict(label="Show existing connections", method="restyle", args=[{"visible": True}, existing_edge_trace_indices])
                                ]))
    if gap_trace_indices:
        updatemenus.append(dict(type="buttons", direction="left", x=0.33, y=1.02, showactive=False,
                                buttons=[
                                    dict(label="Show gap candidates", method="restyle", args=[{"visible": True}, gap_trace_indices]),
                                    dict(label="Hide gap candidates", method="restyle", args=[{"visible": False}, gap_trace_indices])
                                ]))
    updatemenus.append(dict(type="buttons", direction="left", x=0.66, y=1.02, showactive=False, buttons=[
        dict(label="Reset camera", method="relayout", args=[{"scene.camera.eye": dict(x=1.2, y=1.2, z=0.9)}])
    ]))

    fig.update_layout(updatemenus=updatemenus)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[info] Saved interactive plot with toggles to: {out_html}")


# ---------------- Main ----------------

def main(args):
    # Load CSV
    df, csv_path = load_input_csv()
    text_col = select_text_column(df)

    # Ensure essential columns exist
    if "URL" not in df.columns:
        raise ValueError("Input CSV must contain a 'URL' column.")
    # Ensure Name/Category exist even if empty
    for c in ["Name", "Category"]:
        if c not in df.columns:
            df[c] = ""

    # Build df_meta (keep many columns for scoring)
    if "Article ID" in df.columns:
        df_meta = df[["Article ID", "URL", "Name", "Category", "Last Updated", "Has Screenshots", "YouTube Links", "Image Links", "Linked Pages", "Topics Covered", "Content Type", "Word Count"]].copy()
    else:
        df_meta = df[["URL", "Name", "Category", "Last Updated", "Has Screenshots", "YouTube Links", "Image Links", "Linked Pages", "Topics Covered", "Content Type", "Word Count"]].copy()

    docs = df[text_col].fillna("").astype(str).tolist()

    # Embeddings (SBERT if available and not disabled)
    use_sbert = args.use_sbert and not args.no_sbert is False
    embeddings, emb_source, emb_tfidf_vec = get_embeddings(docs, use_sbert, tfidf_max_features=args.max_feats, tfidf_min_df=args.min_df)
    print(f"[info] Embedding source used: {emb_source}; shape: {embeddings.shape}")

    # UMAP (for visualization) - only if available and requested
    coords = None
    if umap is not None:
        try:
            print("[step] reducing to 3D with UMAP for visualization...")
            reducer = umap.UMAP(n_components=args.n_components, metric=args.umap_metric, random_state=args.seed)
            coords = reducer.fit_transform(embeddings)
            print(f"[info] UMAP output shape: {coords.shape}")
        except Exception as e:
            print("[warn] UMAP reduction failed:", e)
            # fallback: use first 3 dims of embeddings or PCA
            if embeddings.shape[1] >= 3:
                coords = embeddings[:, :3]
            else:
                coords = np.pad(embeddings, ((0, 0), (0, max(0, 3 - embeddings.shape[1]))), mode="constant")
    else:
        # fallback
        if embeddings.shape[1] >= 3:
            coords = embeddings[:, :3]
        else:
            coords = np.pad(embeddings, ((0, 0), (0, max(0, 3 - embeddings.shape[1]))), mode="constant")

    # Optional clustering
    labels = None
    if args.kmeans and args.kmeans > 0:
        print(f"[step] running KMeans with k={args.kmeans} ...")
        k = KMeans(n_clusters=args.kmeans, random_state=args.seed, n_init=10)
        labels = k.fit_predict(coords)
        df_meta["cluster"] = labels

    # Attach coords to df_meta
    df_meta["x"] = coords[:, 0]
    df_meta["y"] = coords[:, 1]
    df_meta["z"] = coords[:, 2]

    # Top terms (use TF-IDF fit separately)
    top_terms, top_tfidf_vec = compute_top_terms_tfidf(docs, max_features=args.max_feats, min_df=args.min_df, top_n=args.top_terms)
    df_meta["top_terms"] = top_terms

    # Save vectors CSV (with top_terms & coords)
    df_meta.to_csv(args.out_vectors or OUT_VECTORS_CSV, index=False)
    print(f"[info] Saved vectors + metadata to {args.out_vectors or OUT_VECTORS_CSV}")

    # Load and augment site_map
    site_map = load_site_map(args.site_map or SITE_MAP_FILE)
    site_map = augment_site_map_with_csv_links(df, site_map)

    # Save basic visualization HTML
    plotly_3d(df_meta, coords, out_html=args.out_html or OUT_HTML)

    # Compute semantic distance matrices (raw + normalized)
    dist_raw, dist_norm = compute_semantic_distance_matrix(embeddings)
    if dist_raw is None:
        raise RuntimeError("Failed to compute distance matrix.")

    # Decide exact vs approximate gap detection
    n = dist_raw.shape[0]
    if n > args.max_exact and args.approximate:
        print(f"[warn] N={n} > max_exact={args.max_exact}. Running approximate detection on a sample of size {args.sample_n}.")
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(np.arange(n), size=min(n, args.sample_n), replace=False)
        dist_sample_raw = dist_raw[np.ix_(chosen, chosen)]
        dist_sample_norm = dist_norm[np.ix_(chosen, chosen)]
        df_sample = df_meta.iloc[chosen].reset_index(drop=True)

        gaps_df = find_top_unlinked_pairs_scored(df_sample, dist_sample_norm, site_map, top_k=args.top_k, weights={
            "w_sem": args.w_sem, "w_topic": args.w_topic, "w_recency": args.w_recency,
            "w_media": args.w_media, "w_word": args.w_word, "w_central": args.w_central
        }, max_pairs_warning=args.max_exact)
        gaps_df.to_csv(args.out_gaps or OUT_GAPS_CSV, index=False)
        print(f"[info] Saved gap candidates (sample) to {args.out_gaps or OUT_GAPS_CSV}")
    else:
        # exact
        gaps_df = find_top_unlinked_pairs_scored(df_meta.reset_index(drop=True), dist_norm, site_map, top_k=args.top_k, weights={
            "w_sem": args.w_sem, "w_topic": args.w_topic, "w_recency": args.w_recency,
            "w_media": args.w_media, "w_word": args.w_word, "w_central": args.w_central
        }, max_pairs_warning=args.max_exact)
        gaps_df.to_csv(args.out_gaps or OUT_GAPS_CSV, index=False)
        print(f"[info] Saved gap candidates to {args.out_gaps or OUT_GAPS_CSV}")

    # Produce toggles map including existing edges and gap candidate toggles (if any)
    try:
        plotly_3d_with_colorful_edges_and_toggle(df_meta, coords, site_map, gaps_df=gaps_df, out_html=args.out_html_toggles or OUT_HTML_TOGGLES)
    except Exception as e:
        print("[warn] failed to produce toggle map:", e)

    print("\n✅ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF/SBERT -> UMAP (3D) vectorisation and scored gap candidate finder")
    parser.add_argument("--max-feats", type=int, default=TFIDF_MAX_FEATURES, help="max TF-IDF features")
    parser.add_argument("--min-df", type=int, default=TFIDF_MIN_DF, help="min document frequency for TF-IDF")
    parser.add_argument("--n-components", type=int, default=UMAP_N_COMPONENTS, help="UMAP n_components (should be 3)")
    parser.add_argument("--umap-metric", type=str, default=UMAP_METRIC, help="UMAP metric (e.g. cosine)")
    parser.add_argument("--seed", type=int, default=UMAP_RANDOM_STATE, help="random seed")
    parser.add_argument("--kmeans", type=int, default=KMEANS_CLUSTERS, help="k for KMeans clustering (0 to skip)")
    parser.add_argument("--top-terms", type=int, default=TOP_TERMS_PER_DOC, help="top terms to save per doc")
    parser.add_argument("--top-k", type=int, default=TOP_K_GAPS, help="top K gap candidates to output")
    parser.add_argument("--max-exact", type=int, default=MAX_PAIRWISE_N_FOR_EXACT, help="max N for exact pairwise computation")
    parser.add_argument("--approximate", action="store_true", help="if N > max-exact, sample instead of exact (faster)")
    parser.add_argument("--sample-n", type=int, default=2000, help="sample size when approximate=True")
    parser.add_argument("--out-html", type=str, default=OUT_HTML, help="output interactive HTML file")
    parser.add_argument("--out-html-toggles", type=str, default=OUT_HTML_TOGGLES, help="output HTML file with edge/gap toggles")
    parser.add_argument("--out-vectors", type=str, default=OUT_VECTORS_CSV, help="output vectors CSV")
    parser.add_argument("--out-gaps", type=str, default=OUT_GAPS_CSV, help="output gaps CSV")
    parser.add_argument("--site-map", type=str, default=SITE_MAP_FILE, help="path to site_map.json")
    parser.add_argument("--use-sbert", action="store_true", help="Use sentence-transformers for embeddings if available")
    parser.add_argument("--no-sbert", action="store_true", help="Force NOT to use sentence-transformers (use TF-IDF embeddings)")
    # weights
    parser.add_argument("--w-sem", type=float, default=DEFAULT_WEIGHTS["w_sem"], help="weight: semantic distance")
    parser.add_argument("--w-topic", type=float, default=DEFAULT_WEIGHTS["w_topic"], help="weight: topic complementarity")
    parser.add_argument("--w-recency", type=float, default=DEFAULT_WEIGHTS["w_recency"], help="weight: recency boost")
    parser.add_argument("--w-media", type=float, default=DEFAULT_WEIGHTS["w_media"], help="weight: media complementarity")
    parser.add_argument("--w-word", type=float, default=DEFAULT_WEIGHTS["w_word"], help="weight: wordcount difference")
    parser.add_argument("--w-central", type=float, default=DEFAULT_WEIGHTS["w_central"], help="weight: central -> peripheral preference")

    args = parser.parse_args()
    main(args)
