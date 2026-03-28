import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import src.config as config


class RetrievalEngine:
    # loads FAISS index + embeddings once, handles search queries

    def __init__(self, model_name="vit", feature_mode="full", pca_dim=None, index_type="flat"):
        self.model_name = model_name
        self.feature_mode = feature_mode
        self.pca_dim = pca_dim
        self.index_type = index_type

        pca_suffix = f"_pca{pca_dim}" if pca_dim is not None else ""
        index_path = config.MODELS_DIR / f"portrait_{model_name}_{feature_mode}{pca_suffix}_{index_type}.index"
        embeddings_path = config.PROCESSED_DIR / f"embeddings_{model_name}_{feature_mode}.npy"
        metadata_path = config.PROCESSED_DIR / f"embedding_metadata_{model_name}_{feature_mode}.csv"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                "Run run_build_index.py first."
            )

        self.index = faiss.read_index(str(index_path))
        self.embeddings = np.load(embeddings_path).astype("float32")
        self.metadata = pd.read_csv(metadata_path)

        # if pca was used, we need to apply it to embeddings too when querying
        # by raw embedding (e.g. from an external image)
        self._pca = None
        if pca_dim is not None:
            pca_path = config.MODELS_DIR / f"pca_{model_name}_{feature_mode}_{pca_dim}.joblib"
            if pca_path.exists():
                import joblib
                self._pca = joblib.load(pca_path)
                # pre-transform stored embeddings for direct lookup
                reduced = self._pca.transform(self.embeddings).astype("float32")
                norms = np.linalg.norm(reduced, axis=1, keepdims=True)
                self.emb_idx = reduced / np.clip(norms, 1e-12, None)
            else:
                self.emb_idx = self.embeddings
        else:
            self.emb_idx = self.embeddings

        print(f"Loaded index: {index_path.name}  ({self.index.ntotal} vectors, dim={self.embeddings.shape[1]}, mode={feature_mode})")

    def query_by_index(self, idx, top_k=10):
        # query by row index in metadata dataframe
        q_vec = self.emb_idx[idx:idx+1]
        scores, indices = self.index.search(q_vec, top_k + 1)

        results = []
        for rank, (i, s) in enumerate(zip(indices[0], scores[0]), start=1):
            if i == idx:
                # skip the query itself
                continue
            if len(results) >= top_k:
                break
            row = self.metadata.iloc[i].to_dict()
            row["_rank"] = rank
            row["_score"] = float(s)
            row["_idx"] = int(i)
            results.append(row)

        return results

    def query_by_filename(self, filename, top_k=10):
        # query by filename string, e.g. "painting_12345.jpg"
        matches = self.metadata[self.metadata["filename"] == filename]
        if len(matches) == 0:
            raise ValueError(f"'{filename}' not found in embedding metadata.")
        idx = matches.index[0]
        return self.query_by_index(int(idx), top_k=top_k)  # type: ignore[arg-type]

    def query_by_vector(self, vector, top_k=10):
        # query with arbitrary embedding vector (e.g. from new image)
        vec = vector.astype("float32").reshape(1, -1)

        # normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm

        # apply pca if we have one
        if self._pca is not None:
            vec = self._pca.transform(vec).astype("float32")
            n = np.linalg.norm(vec)
            if n > 1e-12:
                vec = vec / n

        scores, indices = self.index.search(vec, top_k)

        results = []
        for rank, (i, s) in enumerate(zip(indices[0], scores[0]), start=1):
            row = self.metadata.iloc[i].to_dict()
            row["_rank"] = rank
            row["_score"] = float(s)
            row["_idx"] = int(i)  # type: ignore[arg-type]
            results.append(row)

        return results

    def get_metadata_for_index(self, idx):
        return self.metadata.iloc[idx].to_dict()

    def all_filenames(self):
        return self.metadata["filename"].tolist()

    def index_for_filename(self, filename):
        matches = self.metadata[self.metadata["filename"] == filename]
        if len(matches) == 0:
            return None
        return int(matches.index[0])  # type: ignore[arg-type]


def batch_similarity_matrix(engine, indices, normalize=True):
    # pairwise cosine sim (used in notebook)
    vecs = engine.emb_idx[indices]
    if normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-12, None)

    # cosine similarity = dot product of normalized vectors
    sim_matrix = np.dot(vecs, vecs.T)
    return sim_matrix
