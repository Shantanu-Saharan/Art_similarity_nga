"""
Build the FAISS index for similarity search.

Supports flat (exact) and ivfpq (approximate) modes.
"""

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
from sklearn.decomposition import PCA
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import src.config as config


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index for portrait similarity")
    parser.add_argument("--model", type=str, default="vit",
                        choices=["vit", "resnet50", "resnet18"])
    parser.add_argument("--feature_mode", type=str, default="full",
                        choices=["full", "face", "combined", "pose"],
                        help="Feature mode for embeddings")
    parser.add_argument("--index_type", type=str, default="flat",
                        choices=["flat", "ivfpq"],
                        help="flat=exact search, ivfpq=approximate (faster, good for >10k images)")
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="If set, apply PCA to reduce to this many dimensions before indexing. "
                             "256 is a good default. Speeds up search and helps with large collections.")
    return parser.parse_args()


def apply_pca(embeddings, n_components, save_path):
    # fit PCA, renormalize, save model
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings).astype("float32")

    # re-normalize after PCA since it changes the norms
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    reduced = reduced / norms

    joblib.dump(pca, save_path)
    print(f"PCA: {embeddings.shape[1]}d -> {n_components}d, "
          f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    return reduced, pca


def build_flat_index(vectors):
    # exact inner product — fine for small/medium collections
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def build_ivfpq_index(vectors):
    # approximate IVF+PQ — faster for larger collections
    dim = vectors.shape[1]
    n_vectors = vectors.shape[0]

    # nlist = number of voronoi cells, rule of thumb: sqrt(N)
    nlist = max(4, int(np.sqrt(n_vectors)))
    nlist = min(nlist, n_vectors // 10)

    # m = subspace count for PQ, must divide dim evenly
    m = 8
    while dim % m != 0 and m > 1:
        m -= 1

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)

    # IVFPQ needs explicit training
    index.train(vectors)
    index.add(vectors)

    # this many nearest cells to search — trades off speed vs recall
    index.nprobe = max(1, nlist // 4)

    return index


def main():
    args = parse_args()

    embeddings_path = config.PROCESSED_DIR / f"embeddings_{args.model}_{args.feature_mode}.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"No embeddings at {embeddings_path}. Run run_embed.py first."
        )

    embeddings = np.load(embeddings_path).astype("float32")
    print(f"Loaded embeddings: {embeddings.shape}")

    vectors = embeddings
    pca_suffix = ""

    if args.pca_dim is not None:
        pca_save_path = config.MODELS_DIR / f"pca_{args.model}_{args.feature_mode}_{args.pca_dim}.joblib"
        vectors, _ = apply_pca(embeddings, args.pca_dim, pca_save_path)
        pca_suffix = f"_pca{args.pca_dim}"

    if args.index_type == "flat":
        index = build_flat_index(vectors)
    else:
        index = build_ivfpq_index(vectors)

    index_path = config.MODELS_DIR / f"portrait_{args.model}_{args.feature_mode}{pca_suffix}_{args.index_type}.index"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    print(f"Saved FAISS index: {index_path}")
    print(f"Vectors indexed: {index.ntotal}")
    print(f"Dimension: {vectors.shape[1]}")


if __name__ == "__main__":
    main()
