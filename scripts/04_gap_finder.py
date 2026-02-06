#!/usr/bin/env python3
"""
vectorise_and_find_gaps.py

Inputs expected in CWD:
 - file_content_cleaned.csv  (preferred)
   or file_content.csv (fallback) with columns:
     - Article ID, URL, Name, Category, Article Content (or Cleaned Article Content)
 - site_map.json  (parent->children dict produced by crawler)

Outputs:
 - vectors_3d.csv       : ArticleID, URL, Name, Category, x,y,z, cluster (optional), top_terms
 - site_vectors_3d.html : interactive 3D plot (Plotly)
 - gap_candidates.csv   : top K furthest unlinked pairs (url_a, title_a, url_b, title_b, distance)
"""

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
import argparse
import sys

# ---------------- Configuration / Defaults ----------------
DEFAULT_INPUT_CSVS = ["assignments/zb_assgn/data/file_content_cleaned.csv", "assignments/zb_assgn/data/file_content.csv"]
SITE_MAP_FILE = "assignments/zb_assgn/data/site_map.json"
OUT_VECTORS_CSV = "assignments/zb_assgn/data/vectors_3d.csv"
OUT_GAPS_CSV = "assignments/zb_assgn/data/gap_candidates.csv"
OUT_HTML = "assignments/zb_assgn/data/site_vectors_3d.html"

TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 2
UMAP_N_COMPONENTS = 3
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42
KMEANS_CLUSTERS = 10  # set 0 or None to skip clustering
TOP_TERMS_PER_DOC = 8

TOP_K_GAPS = 200  # top K furthest unlinked pairs to save
MAX_PAIRWISE_N_FOR_EXACT = 3000  # above this, warn about O(N^2) cost

# ---------------- Helpers ----------------

def load_input_csv():
    for p in DEFAULT_INPUT_CSVS:
        if Path(p).exists():
            print(f"[info] Loading {p}")
            df = pd.read_csv(p)
            return df, p
    raise FileNotFoundError(f"None of the expected input CSVs found: {DEFAULT_INPUT_CSVS}")

def select_text_column(df: pd.DataFrame):
    # prefer cleaned column
    for col in ["Cleaned Article Content", "Article Content", "Article_Content", "content"]:
        if col in df.columns:
            print(f"[info] Using text column: {col}")
            return col
    raise ValueError("No article content column found. Expected 'Cleaned Article Content' or 'Article Content'.")

def load_site_map(path=SITE_MAP_FILE):
    p = Path(path)
    if not p.exists():
        print(f"[warn] site_map not found at {path}. Proceeding with empty map (no links).")
        return {}
    return json.loads(p.read_text(encoding="utf8"))

def has_link(site_map, a, b):
    # check both directions quickly
    links_a = site_map.get(a, [])
    if b in links_a:
        return True
    links_b = site_map.get(b, [])
    if a in links_b:
        return True
    return False

def top_terms_for_doc(tfidf_vector, feature_names, top_n=TOP_TERMS_PER_DOC):
    # tfidf_vector: 1D array or sparse vector
    if hasattr(tfidf_vector, "toarray"):
        arr = tfidf_vector.toarray().ravel()
    else:
        arr = np.asarray(tfidf_vector).ravel()
    if arr.sum() == 0:
        return []
    idxs = np.argsort(-arr)[:top_n]
    return [feature_names[i] for i in idxs if arr[i] > 0]

# ---------------- Pipeline Functions ----------------

def compute_tfidf(docs, max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF):
    print("[step] computing TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words="english"
    )
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    print(f"[info] TF-IDF matrix shape: {X.shape}")
    return X, feature_names, vectorizer

def reduce_umap(X, n_components=UMAP_N_COMPONENTS, metric=UMAP_METRIC, random_state=UMAP_RANDOM_STATE):
    print("[step] reducing to 3D with UMAP...")
    reducer = umap.UMAP(n_components=n_components, metric=metric, random_state=random_state)
    coords = reducer.fit_transform(X)
    print(f"[info] UMAP output shape: {coords.shape}")
    return coords, reducer

def compute_pairwise_distances(coords):
    # compute condensed pdist then squareform; distances in Euclidean on coords
    print("[step] computing pairwise distances (Euclidean on 3D coords)...")
    # ensure float64
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be shape (N,3)")
    condensed = pdist(coords, metric="euclidean")
    full = squareform(condensed)
    return full

def find_top_unlinked_pairs(df_meta, dist_matrix, site_map, top_k=TOP_K_GAPS):
    print("[step] finding top unlinked pairs by distance...")
    n = dist_matrix.shape[0]
    # We'll produce list of (i, j, dist), i < j
    idxs = np.triu_indices(n, k=1)
    dists = dist_matrix[idxs]
    # compute order descending
    order = np.argsort(-dists)
    results = []
    count = 0
    for pos in order:
        i = idxs[0][pos]
        j = idxs[1][pos]
        a_url = df_meta.iloc[i]["URL"]
        b_url = df_meta.iloc[j]["URL"]
        if has_link(site_map, a_url, b_url):
            continue
        results.append({
            "url_a": a_url,
            "title_a": df_meta.iloc[i].get("Name",""),
            "url_b": b_url,
            "title_b": df_meta.iloc[j].get("Name",""),
            "distance": float(dists[pos])
        })
        count += 1
        if count >= top_k:
            break
    print(f"[info] Found {len(results)} gap candidates")
    return pd.DataFrame(results)

def run_kmeans(coords, n_clusters=KMEANS_CLUSTERS, seed=42):
    if not n_clusters:
        return None
    print(f"[step] running KMeans with k={n_clusters} ...")
    k = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = k.fit_predict(coords)
    return labels

# ---------------- Visualization ----------------

def plotly_3d(df_meta, coords, out_html=OUT_HTML):
    """
    Create a Plotly 3D scatter by constructing a small data-frame with coords and metadata.
    This avoids the ValueError that occurs when passing arrays + hover_data without a data_frame.
    """
    import pandas as _pd

    print("[step] generating 3D interactive HTML plot...")

    # Make a copy to avoid mutating the original df_meta
    df_plot = df_meta.copy().reset_index(drop=True)

    # Attach coords as columns (ensure matching order)
    df_plot["x"] = coords[:, 0]
    df_plot["y"] = coords[:, 1]
    df_plot["z"] = coords[:, 2]

    # Prepare hover text (safer than hover_data if you want a custom string)
    # This gives you a succinct hover box; you can also use hover_data below.
    df_plot["hover_text"] = df_plot.apply(
        lambda r: f"{r.get('Name','')}<br>{r.get('URL','')}<br>Category: {r.get('Category','')}",
        axis=1
    )

    # Determine color column if present
    color_col = None
    if "cluster" in df_plot.columns:
        color_col = "cluster"

    # Use plotly.express with a proper data_frame reference
    fig = px.scatter_3d(
        df_plot,
        x="x",
        y="y",
        z="z",
        color=color_col,
        hover_name="Name",
        hover_data={"URL": True, "Category": True},  # small dict -> explicit display
        title="Semantic 3D map (UMAP on TF-IDF)",
    )

    # Tune marker appearance
    fig.update_traces(
        marker=dict(size=4, line=dict(width=0.5, color="DarkSlateGrey")),
        selector=dict(mode="markers")
    )

    # Improve layout: hide axis lines/ticks for cleaner look
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False)
        ),
        margin=dict(t=40, l=0, r=0, b=0),
        height=900,
    )

    # Save
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[info] 3D plot saved to {out_html}")

def plotly_3d_with_colorful_edges_and_toggle(df_meta, coords, site_map, gaps_df=None, out_html="site_vectors_3d_with_edges.html", seed=42):
    """
    Plot 3D semantic map with:
      - each existing connection in a (random) color (or grouped colors if too many)
      - gap candidate links hidden by default, toggleable via buttons
      - UI buttons: show/hide existing edges, show/hide gaps
    Args:
      df_meta : DataFrame containing at least ['URL','Name','Category'] and optional 'cluster'
      coords  : numpy array (N,3)
      site_map: dict parent -> [children...]
      gaps_df : DataFrame with columns ['url_a','url_b','distance', ...] (optional)
    """
    import plotly.graph_objects as go
    import numpy as np
    import random
    from math import ceil
    import plotly.express as px

    random.seed(seed)
    np.random.seed(seed)

    # build plotting dataframe
    df_plot = df_meta.copy().reset_index(drop=True)
    df_plot["x"] = coords[:, 0]
    df_plot["y"] = coords[:, 1]
    df_plot["z"] = coords[:, 2]
    df_plot["hover_text"] = df_plot.apply(
        lambda r: f"<b>{r.get('Name','')}</b><br>{r.get('URL','')}<br>Category: {r.get('Category','')}",
        axis=1
    )

    # quick map URL -> coords
    url_to_coord = {row["URL"]: (row["x"], row["y"], row["z"]) for _, row in df_plot.iterrows()}

    # collect existing directed/undirected edges from site_map
    existing_edges = []
    for parent, children in site_map.items():
        if parent not in url_to_coord:
            continue
        for ch in children:
            if ch not in url_to_coord:
                continue
            # treat edges as undirected for visualization
            a, b = sorted((parent, ch))
            existing_edges.append((a, b))
    # dedupe
    existing_edges = list(dict.fromkeys(existing_edges))

    n_edges = len(existing_edges)
    print(f"[info] existing edges: {n_edges}")

    # decide per-edge traces vs grouped traces
    PER_EDGE_LIMIT = 800   # threshold for making per-edge traces
    if n_edges <= PER_EDGE_LIMIT:
        per_edge = True
        print("[info] creating one trace per existing edge (per-edge mode).")
    else:
        per_edge = False
        N_BUCKETS = 30
        print(f"[info] too many edges ({n_edges}), grouping into {N_BUCKETS} color buckets for performance.")

    traces = []
    existing_edge_trace_indices = []  # indices of traces corresponding to existing edges (one or many)
    gap_trace_indices = []

    # ---------- Existing edges traces ----------
    if per_edge:
        # make one trace per edge with random color
        for (a, b) in existing_edges:
            x0, y0, z0 = url_to_coord[a]
            x1, y1, z1 = url_to_coord[b]
            color = "rgb({}, {}, {})".format(*[random.randint(20,220) for _ in range(3)])
            tr = go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="none",
                visible=True,
                showlegend=False
            )
            existing_edge_trace_indices.append(len(traces))
            traces.append(tr)
    else:
        # group edges into buckets by hashing, create one trace per bucket with a single color
        N_BUCKETS = 30
        buckets = [[] for _ in range(N_BUCKETS)]
        for (a, b) in existing_edges:
            idx = hash(a + "||" + b) % N_BUCKETS
            buckets[idx].append((a, b))
        # create a trace per non-empty bucket
        for bucket in buckets:
            if not bucket:
                continue
            xs, ys, zs = [], [], []
            color = "rgb({}, {}, {})".format(*[random.randint(20,220) for _ in range(3)])
            for (a,b) in bucket:
                x0, y0, z0 = url_to_coord[a]
                x1, y1, z1 = url_to_coord[b]
                xs += [x0, x1, None]
                ys += [y0, y1, None]
                zs += [z0, z1, None]
            tr = go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=2.5),
                hoverinfo="none",
                visible=True,
                showlegend=False
            )
            existing_edge_trace_indices.append(len(traces))
            traces.append(tr)

    # ---------- Gap candidate traces (hidden by default) ----------
    if gaps_df is not None and len(gaps_df) > 0:
        # ensure gaps_df rows correspond to urls present in url_to_coord
        gaps = []
        for _, r in gaps_df.iterrows():
            a = r["url_a"]; b = r["url_b"]
            if a in url_to_coord and b in url_to_coord:
                gaps.append((a, b))
        print(f"[info] gap candidate pairs present: {len(gaps)}")
        # If not too many, make per-gap trace; otherwise group into a few traces
        MAX_GAP_PER_EDGE = 1000
        if len(gaps) <= MAX_GAP_PER_EDGE:
            for (a,b) in gaps:
                x0, y0, z0 = url_to_coord[a]
                x1, y1, z1 = url_to_coord[b]
                # use a distinct palette (red-ish)
                color = "rgb({}, {}, {})".format(220, random.randint(30,120), random.randint(30,120))
                tr = go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode="lines",
                    line=dict(color=color, width=3, dash="dash"),
                    hoverinfo="none",
                    visible=False,   # hidden by default
                    showlegend=False
                )
                gap_trace_indices.append(len(traces))
                traces.append(tr)
        else:
            # bucket gaps into groups
            G_BUCKETS = 20
            buckets = [[] for _ in range(G_BUCKETS)]
            for (a, b) in gaps:
                idx = hash(a + "||" + b) % G_BUCKETS
                buckets[idx].append((a,b))
            for bucket in buckets:
                if not bucket:
                    continue
                xs, ys, zs = [], [], []
                color = "rgb({}, {}, {})".format(220, random.randint(30,100), random.randint(30,100))
                for (a,b) in bucket:
                    x0, y0, z0 = url_to_coord[a]
                    x1, y1, z1 = url_to_coord[b]
                    xs += [x0, x1, None]
                    ys += [y0, y1, None]
                    zs += [z0, z1, None]
                tr = go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(color=color, width=2.5, dash="dash"),
                    hoverinfo="none",
                    visible=False,
                    showlegend=False
                )
                gap_trace_indices.append(len(traces))
                traces.append(tr)

    # ---------------------
    # NODE TRACE (FIXED COLOR HANDLING)
    # ---------------------
    if "cluster" in df_plot.columns:
        # numeric cluster ids → colorscale
        node_colors = df_plot["cluster"].astype(int)
        show_colorbar = True
    else:
        node_colors = "orange"
        show_colorbar = False

    node_trace = go.Scatter3d(
        x=df_plot["x"],
        y=df_plot["y"],
        z=df_plot["z"],
        mode="markers",
        marker=dict(
            size=7,
            color=node_colors,
            colorscale="Viridis",   # perceptually uniform
            opacity=0.95,
            showscale=show_colorbar,
            colorbar=dict(
                title="Cluster",
                thickness=15
            ),
            line=dict(width=1.0, color="black")
        ),
        text=df_plot["hover_text"],
        hoverinfo="text",
        name="Articles",
        visible=True
    )

    # append node trace last so nodes are drawn on top
    node_trace_index = len(traces)
    traces.append(node_trace)

    # ---------- Build figure ----------
    layout = go.Layout(
        title="Semantic 3D Map — edges colored per connection",
        showlegend=False,
        height=900,
        scene=dict(
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.9)),
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
        ),
        margin=dict(t=50, l=0, r=0, b=0),
    )

    fig = go.Figure(data=traces, layout=layout)

    # ---------- Updatemenus (buttons to show/hide existing edges & gaps) ----------
    updatemenus = []

    if existing_edge_trace_indices:
        updatemenus.append(dict(
            type="buttons",
            direction="left",
            x=0.02, y=1.02,
            showactive=False,
            buttons=[
                dict(label="Hide existing connections",
                     method="restyle",
                     args=[{"visible": False}, existing_edge_trace_indices]),
                dict(label="Show existing connections",
                     method="restyle",
                     args=[{"visible": True}, existing_edge_trace_indices])
            ]
        ))

    if gap_trace_indices:
        updatemenus.append(dict(
            type="buttons",
            direction="left",
            x=0.33, y=1.02,
            showactive=False,
            buttons=[
                dict(label="Show gap candidates",
                     method="restyle",
                     args=[{"visible": True}, gap_trace_indices]),
                dict(label="Hide gap candidates",
                     method="restyle",
                     args=[{"visible": False}, gap_trace_indices])
            ]
        ))

    # add a camera reset button (optional)
    updatemenus.append(dict(
        type="buttons",
        direction="left",
        x=0.66, y=1.02,
        showactive=False,
        buttons=[
            dict(label="Reset camera",
                 method="relayout",
                 args=[{"scene.camera.eye": dict(x=1.2, y=1.2, z=0.9)}])
        ]
    ))

    fig.update_layout(updatemenus=updatemenus)

    # Save HTML
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[info] Saved interactive plot with toggles to: {out_html}")

# ---------------- Main ----------------

def main(args):
    df, csv_path = load_input_csv()
    text_col = select_text_column(df)
    # ensure URL and Article ID columns exist
    if "URL" not in df.columns:
        raise ValueError("Input CSV must contain a 'URL' column.")
    df_meta = df[["Article ID","URL","Name","Category"]].copy() if "Article ID" in df.columns else df[["URL","Name","Category"]].copy()
    docs = df[text_col].fillna("").astype(str).tolist()

    # TF-IDF
    X, feature_names, vectorizer = compute_tfidf(docs, max_features=args.max_feats, min_df=args.min_df)

    # UMAP
    coords, reducer = reduce_umap(X, n_components=args.n_components, metric=args.umap_metric, random_state=args.seed)

    # Optionally cluster
    labels = None
    if args.kmeans and args.kmeans > 0:
        labels = run_kmeans(coords, n_clusters=args.kmeans, seed=args.seed)
        df_meta["cluster"] = labels

    # attach coords & top terms
    df_meta["x"] = coords[:,0]
    df_meta["y"] = coords[:,1]
    df_meta["z"] = coords[:,2]

    # compute top terms per doc using TF-IDF sparse matrix rows
    print("[step] extracting top terms per document...")
    try:
        # X is sparse; get row i via X[i]
        top_terms = []
        for i in range(X.shape[0]):
            row = X[i]
            terms = top_terms_for_doc(row, feature_names, top_n=args.top_terms)
            top_terms.append(", ".join(terms))
    except Exception as e:
        print("[warn] top term extraction failed:", e)
        top_terms = [""] * X.shape[0]

    df_meta["top_terms"] = top_terms

    # Save vectors CSV
    df_meta.to_csv(OUT_VECTORS_CSV, index=False)
    print(f"[info] Saved vectors + metadata to {OUT_VECTORS_CSV}")

    # Load site_map (needed for drawing edges) and save 3D interactive plot with edges
    # site_map = load_site_map(SITE_MAP_FILE)
    # plotly_3d(df_meta, coords, site_map, out_html=args.out_html)
    plotly_3d(df_meta, coords, out_html=args.out_html)

    # Load colored map
    site_map_01 = json.load(open("site_map.json"))
    gaps_df = pd.read_csv("assignments/zb_assgn/data/gap_candidates.csv")  # produced by vectorisation script
    plotly_3d_with_colorful_edges_and_toggle(df_meta, coords, site_map_01, gaps_df=gaps_df, out_html="assignments/zb_assgn/data/map_with_toggles.html")


    # compute pairwise distances
    n = coords.shape[0]
    if n > args.max_exact and args.approximate:
        print(f"[warn] N={n} > {args.max_exact}. Exact O(N^2) pairwise computation is expensive.")
        print("        You chose approximate mode; sampling pairs for candidate selection.")
        # sample a subset of nodes, e.g. 2000
        sample_n = min(n, args.sample_n)
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(np.arange(n), size=sample_n, replace=False)
        coords_sample = coords[chosen]
        dist_mat = squareform(pdist(coords_sample, metric="euclidean"))
        # map indices back for output - simpler to warn user to run with larger resources
        # For now we will compute gaps only on the sample and annotate with original indices
        print("[warn] gap detection on sample only; to compute exact gaps rerun with approximate=False and enough memory.")
        # load site_map
        site_map = load_site_map()
        df_sample = df_meta.iloc[chosen].reset_index(drop=True)
        gaps_df = find_top_unlinked_pairs(df_sample, dist_mat, site_map, top_k=args.top_k)
        gaps_df.to_csv(OUT_GAPS_CSV, index=False)
        print(f"[info] Saved gap candidates (sample) to {OUT_GAPS_CSV}")
    else:
        # exact pairwise
        dist_mat = compute_pairwise_distances(coords)
        site_map = load_site_map()
        gaps_df = find_top_unlinked_pairs(df_meta.reset_index(drop=True), dist_mat, site_map, top_k=args.top_k)
        gaps_df.to_csv(OUT_GAPS_CSV, index=False)
        print(f"[info] Saved gap candidates to {OUT_GAPS_CSV}")

    print("\n✅ Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF -> UMAP (3D) vectorisation and gap candidate finder")
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
    args = parser.parse_args()
    main(args)
