"""
Evaluation metrics for portrait similarity.

Metrics: Precision@K, mAP, SSIM. Using artist as proxy relevance label.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from PIL import Image
from tqdm import tqdm
# leftover from debugging, harmless
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

import src.config as config


def precision_at_k(retrieved_artists, query_artist, k):
    # precision@k for retrieval
    topk = retrieved_artists[:k]
    relevant = sum(1 for a in topk if a == query_artist)
    return relevant / max(k, 1)


def average_precision(retrieved_artists, query_artist):
    # average precision for a single query
    hits = 0
    running_precision = 0.0
    for rank, artist in enumerate(retrieved_artists, start=1):
        if artist == query_artist:
            hits += 1
            running_precision += hits / rank
    # avoid division by zero if no positive exists
    n_relevant = sum(1 for a in retrieved_artists if a == query_artist)
    if n_relevant == 0:
        return 0.0
    return running_precision / n_relevant


def compute_ssim_pair(img_path_a, img_path_b, size=(128, 128)):
    # structural similarity between two images
    img_a = np.array(Image.open(img_path_a).convert("RGB"))
    img_b = np.array(Image.open(img_path_b).convert("RGB"))

    # resize to 128x128 to keep this fast across many pairs
    img_a = resize(img_a, (*size, 3), anti_aliasing=True)
    img_b = resize(img_b, (*size, 3), anti_aliasing=True)

    score, _ = ssim(img_a, img_b, channel_axis=2, full=True,
                    win_size=7, data_range=1.0)
    return float(score)


def run_evaluation(embeddings_path, metadata_path, image_dir, top_k=10, n_queries=50, seed=42):
    # run evaluation over n_queries randomly sampled images
    import faiss

    embeddings = np.load(embeddings_path).astype("float32")
    metadata = pd.read_csv(metadata_path)

    assert len(embeddings) == len(metadata), "Embedding count doesn't match metadata rows"

    # Clean artist labels for robustness
    metadata["artist_hint"] = metadata["artist_hint"].fillna("unknown").astype(str).str.strip()
    unknown_labels = {"unknown", "nan", "", "none"}
    metadata["artist_hint"] = metadata["artist_hint"].apply(
        lambda x: x if x.lower() not in unknown_labels else "unknown"
    )

    # only evaluate artists where we have multiple images
    artist_counts = metadata["artist_hint"].value_counts()
    valid_artists = set(artist_counts[artist_counts >= 3].index)

    valid_idx = metadata[metadata["artist_hint"].isin(valid_artists)].index.tolist()
    if len(valid_idx) == 0:
        raise ValueError("No artists with >= 3 images found. Can't evaluate meaningfully.")

    rng = np.random.default_rng(seed)
    query_indices = rng.choice(valid_idx, size=min(n_queries, len(valid_idx)), replace=False)

    # build a flat inner-product index (embeddings are already l2-normalized)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    p_at_1 = []
    p_at_5 = []
    p_at_k = []
    ap_scores = []
    ssim_scores = []

    image_dir = Path(image_dir)

    for q_idx in tqdm(query_indices, desc="Evaluating queries"):
        q_vec = embeddings[q_idx:q_idx+1]
        q_artist = metadata.iloc[q_idx]["artist_hint"]

        scores, indices = index.search(q_vec, top_k + 1)
        retrieved = [i for i in indices[0] if i != q_idx][:top_k]
        retrieved_artists = [metadata.iloc[i]["artist_hint"] for i in retrieved]

        p_at_1.append(precision_at_k(retrieved_artists, q_artist, 1))
        p_at_5.append(precision_at_k(retrieved_artists, q_artist, min(5, top_k)))
        p_at_k.append(precision_at_k(retrieved_artists, q_artist, top_k))
        ap_scores.append(average_precision(retrieved_artists, q_artist))

        # ssim for top 3 results only — too slow to do all k
        q_img_path = image_dir / metadata.iloc[q_idx]["filename"]
        if q_img_path.exists():
            for r_idx in retrieved[:3]:
                r_img_path = image_dir / metadata.iloc[r_idx]["filename"]
                if r_img_path.exists():
                    try:
                        s = compute_ssim_pair(q_img_path, r_img_path)
                        ssim_scores.append(s)
                    except Exception:
                        pass

    results = {
        "n_queries": len(query_indices),
        "mean_precision_at_1": float(np.mean(p_at_1)),
        "mean_precision_at_5": float(np.mean(p_at_5)),
        f"mean_precision_at_{top_k}": float(np.mean(p_at_k)),
        "mean_average_precision": float(np.mean(ap_scores)),
        "mean_ssim_top3": float(np.mean(ssim_scores)) if ssim_scores else None,
    }

    return results


if __name__ == "__main__":
    # quick sanity check
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate portrait similarity")
    parser.add_argument("--model", type=str, default="vit")
    parser.add_argument("--feature_mode", type=str, default="full")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--n_queries", type=int, default=50)
    args = parser.parse_args()

    embeddings_path = config.PROCESSED_DIR / f"embeddings_{args.model}_{args.feature_mode}.npy"
    metadata_path = config.PROCESSED_DIR / f"embedding_metadata_{args.model}_{args.feature_mode}.csv"

    if not embeddings_path.exists():
        print(f"Embeddings not found: {embeddings_path}")
        print("Run run_embed.py first.")
        sys.exit(1)

    results = run_evaluation(
        embeddings_path=str(embeddings_path),
        metadata_path=str(metadata_path),
        image_dir=str(config.IMAGE_DIR),
        top_k=args.top_k,
        n_queries=args.n_queries,
    )

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")