"""
Search for similar portraits to a given query image.

Shows the query + top-K results in a matplotlib grid and saves it to outputs/.
Also prints a text summary with similarity scores and artist metadata.

Usage:
    python run_search.py --query_filename painting_12345.jpg
    python run_search.py --query_filename painting_12345.jpg --top_k 8 --model vit
"""

import argparse
import sys
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import src.config as config


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve similar paintings")
    parser.add_argument("--query_filename", type=str, required=True,
                        help="e.g. painting_12345.jpg")
    parser.add_argument("--model", type=str, default="vit",
                        choices=["vit", "resnet50", "resnet18"])
    parser.add_argument("--feature_mode", type=str, default="full",
                        choices=["full", "face", "combined", "pose"],
                        help="Feature mode for retrieval")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="If PCA was used when building the index, specify the dim here")
    parser.add_argument("--index_type", type=str, default="flat",
                        choices=["flat", "ivfpq"])
    return parser.parse_args()


def load_index_and_data(args):
    pca_suffix = f"_pca{args.pca_dim}" if args.pca_dim is not None else ""
    index_path = config.MODELS_DIR / f"portrait_{args.model}_{args.feature_mode}{pca_suffix}_{args.index_type}.index"
    embeddings_path = config.PROCESSED_DIR / f"embeddings_{args.model}_{args.feature_mode}.npy"
    metadata_path = config.PROCESSED_DIR / f"embedding_metadata_{args.model}_{args.feature_mode}.csv"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\nRun run_build_index.py first."
        )

    index = faiss.read_index(str(index_path))
    embeddings = np.load(embeddings_path).astype("float32")
    metadata = pd.read_csv(metadata_path)

    # if pca was used, apply it to embeddings before search
    if args.pca_dim is not None:
        import joblib
        pca_path = config.MODELS_DIR / f"pca_{args.model}_{args.feature_mode}_{args.pca_dim}.joblib"
        if pca_path.exists():
            pca = joblib.load(pca_path)
            embeddings = pca.transform(embeddings).astype("float32")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-12, None)

    return index, embeddings, metadata


def show_and_save_results(query_row, result_rows, scores, image_dir, output_path):
    # query on left, top-k results to the right; green border = same artist
    n = len(result_rows) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5))
    if n == 1:
        axes = [axes]

    query_artist = str(query_row["artist_hint"])

    # load and show query
    q_img = Image.open(image_dir / query_row["filename"]).convert("RGB")
    axes[0].imshow(q_img)
    axes[0].set_title("Query\n" + _short_label(query_row), fontsize=8, pad=4)
    axes[0].axis("off")
    # draw a gold border around the query
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#C08000")
        spine.set_linewidth(3)
        spine.set_visible(True)

    for i, (_, row) in enumerate(result_rows.iterrows(), start=1):
        img = Image.open(image_dir / row["filename"]).convert("RGB")
        axes[i].imshow(img)

        same_artist = str(row["artist_hint"]) == query_artist
        border_color = "#2e8b57" if same_artist else "#666666"

        title = f"rank {i} | score={scores[i-1]:.3f}\n" + _short_label(row)
        axes[i].set_title(title, fontsize=8, pad=4)
        axes[i].axis("off")

        for spine in axes[i].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    # legend
    same_patch = mpatches.Patch(color="#2e8b57", label="same artist")
    diff_patch = mpatches.Patch(color="#666666", label="different artist")
    fig.legend(handles=[same_patch, diff_patch], loc="lower center",
               ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Portrait similarity results", fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved result image to: {output_path}")


def _short_label(row):
    title = str(row.get("title", ""))[:30]
    artist = str(row.get("artist_hint", ""))[:25]
    return f"{title}\n{artist}"


def main():
    args = parse_args()

    index, embeddings, metadata = load_index_and_data(args)

    matches = metadata[metadata["filename"] == args.query_filename]
    if len(matches) == 0:
        raise ValueError(f"'{args.query_filename}' not found in embedding metadata.")

    q_idx = matches.index[0]
    q_vec = embeddings[q_idx:q_idx+1]

    scores, indices = index.search(q_vec, args.top_k + 1)

    results = [
        (int(idx), float(sc))
        for idx, sc in zip(indices[0], scores[0])
        if idx != q_idx
    ][:args.top_k]

    result_indices = [r[0] for r in results]
    result_scores = [r[1] for r in results]

    query_row = metadata.iloc[q_idx]
    result_rows = metadata.iloc[result_indices]

    print(f"\nQuery: {query_row['filename']}")
    print(f"  title:  {query_row['title']}")
    print(f"  artist: {query_row['artist_hint']}")
    print(f"  model:  {args.model}, mode: {args.feature_mode}")
    print()
    print(f"Top {args.top_k} similar portraits:")
    for rank, (r_idx, score) in enumerate(results, start=1):
        r = metadata.iloc[r_idx]
        same = "(same artist)" if r["artist_hint"] == query_row["artist_hint"] else ""
        print(f"  {rank}. score={score:.4f} | {r['filename']} | {r['artist_hint']} {same}")

    # Use query filename stem for clean output naming
    query_stem = Path(args.query_filename).stem
    output_path = config.OUTPUTS_DIR / f"result_{args.model}_{args.feature_mode}_{query_stem}.png"
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    show_and_save_results(query_row, result_rows, result_scores, config.IMAGE_DIR, output_path)


if __name__ == "__main__":
    main()
