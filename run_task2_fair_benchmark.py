# Task-2 benchmark: fuse resnet50 + efficientnet + finetuned vit, then DBA + face rerank

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
TASK2_DIR = ROOT / "Task-2"
sys.path.append(str(ROOT))
sys.path.append(str(TASK2_DIR))

from src.models.embedding_model import build_model, get_device
from task2.nga_similarity import (
    build_encoder,
    build_transform_for_model,
    precision_recall_ap,
)


COMPETITOR_BASELINE = {
    "precision@10": 0.112981,
    "recall@10": 0.332479,
    "map@10": 0.244879,
    "mrr": 0.377223,
}


class TorchvisionDataset(Dataset):
    def __init__(self, metadata, image_dir, transform):
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        object_id = int(self.metadata.iloc[idx]["objectid"])
        image = Image.open(self.image_dir / f"{object_id}.jpg").convert("RGB")
        return self.transform(image), idx


class VitDataset(Dataset):
    def __init__(self, metadata, image_dir):
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        object_id = int(self.metadata.iloc[idx]["objectid"])
        image = Image.open(self.image_dir / f"{object_id}.jpg").convert("RGB")
        return self.transform(image), idx


class FaceCropVitDataset(Dataset):
    def __init__(self, metadata, image_dir, original_indices, face_boxes, padding):
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.original_indices = original_indices
        self.face_boxes = face_boxes
        self.padding = padding
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        object_id = int(self.metadata.iloc[idx]["objectid"])
        image = Image.open(self.image_dir / f"{object_id}.jpg").convert("RGB")
        x, y, w, h = self.face_boxes[idx]
        pad_x = int(w * self.padding)
        pad_y = int(h * self.padding)
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(image.width, x + w + pad_x)
        bottom = min(image.height, y + h + pad_y)
        crop = image.crop((left, top, right, bottom))
        return self.transform(crop), self.original_indices[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a fair NGA Task-2 benchmark")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=TASK2_DIR / "data/processed/nga_similarity_metadata.csv",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=TASK2_DIR / "data/images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "task2_fair_benchmark",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label-column", type=str, default="artist_name")
    parser.add_argument("--resnet-weight", type=float, default=0.42)
    parser.add_argument("--efficientnet-weight", type=float, default=0.28)
    parser.add_argument("--vit-weight", type=float, default=0.30)
    parser.add_argument("--dba-k", type=int, default=2)
    parser.add_argument("--dba-alpha", type=float, default=3.0)
    parser.add_argument("--face-weight", type=float, default=0.0)
    parser.add_argument("--face-rerank-topn", type=int, default=0)
    parser.add_argument("--face-padding", type=float, default=0.20)
    parser.add_argument("--face-min-rel-area", type=float, default=0.0)
    parser.add_argument(
        "--face-detector-mode",
        type=str,
        default="hybrid",
        choices=["strict", "hybrid", "loose"],
    )
    parser.add_argument(
        "--vit-checkpoint",
        type=Path,
        default=ROOT / "models/portrait_vit_improved_l40.pth",
    )
    return parser.parse_args()


def download_missing_images(metadata, image_dir, max_workers):
    image_dir.mkdir(parents=True, exist_ok=True)
    missing = metadata[
        ~metadata["objectid"].map(lambda object_id: (image_dir / f"{int(object_id)}.jpg").exists())
    ].copy()
    if missing.empty:
        return 0

    session = requests.Session()

    def _download(row):
        object_id = int(row["objectid"])
        path = image_dir / f"{object_id}.jpg"
        try:
            response = session.get(row["image_url"], timeout=20)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(path, quality=90)
            return object_id, True, ""
        except Exception as exc:
            return object_id, False, str(exc)

    ok = 0
    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download, row) for row in missing.to_dict("records")]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading missing NGA images"):
            object_id, success, error = future.result()
            if success:
                ok += 1
            else:
                failures.append({"objectid": object_id, "error": error})

    if failures:
        print(f"Warning: {len(failures)} downloads failed")
        for failure in failures[:10]:
            print(f"  {failure['objectid']}: {failure['error']}")

    return ok


def load_task2_metadata(metadata_csv, image_dir, limit):
    metadata = pd.read_csv(metadata_csv).head(limit).copy()
    metadata = metadata[
        metadata["objectid"].map(lambda object_id: (image_dir / f"{int(object_id)}.jpg").exists())
    ].reset_index(drop=True)
    return metadata


def extract_torchvision_embeddings(metadata, image_dir, model_name, batch_size, num_workers, device):
    transform = build_transform_for_model(model_name)
    dataset = TorchvisionDataset(metadata, image_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model, dim = build_encoder(model_name=model_name)
    model = model.to(device)
    model.eval()

    embeddings = np.zeros((len(metadata), dim), dtype=np.float32)
    with torch.no_grad():
        for images, indices in tqdm(loader, desc=f"Embedding {model_name}"):
            features = model(images.to(device)).flatten(1).cpu().numpy()
            embeddings[indices.numpy()] = features

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def extract_finetuned_vit_embeddings(metadata, image_dir, batch_size, num_workers, checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ViT checkpoint not found: {checkpoint_path}")

    dataset = VitDataset(metadata, image_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model("vit", embed_dim=512, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    embeddings = np.zeros((len(metadata), 512), dtype=np.float32)
    with torch.no_grad():
        for images, indices in tqdm(loader, desc="Embedding fine-tuned vit"):
            features = model(images.to(device)).cpu().numpy()
            embeddings[indices.numpy()] = features

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def _select_face_box(detections, image_width, image_height):
    image_area = max(image_width * image_height, 1)

    def score(det) -> float:
        x, y, w, h = [int(value) for value in det]
        box_area = (w * h) / image_area
        cx = (x + 0.5 * w) / image_width
        cy = (y + 0.5 * h) / image_height
        # 0.42 not 0.5 - portrait faces tend to sit slightly above vertical center
        center_penalty = (cx - 0.5) ** 2 + (cy - 0.42) ** 2
        return float(box_area * math.exp(-4.0 * center_penalty))

    x, y, w, h = max(detections, key=score)
    return int(x), int(y), int(w), int(h)


def detect_face_boxes(metadata, image_dir, detector_mode, min_rel_area):
    cascades = {
        "frontal_default": cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        ),
        "frontal_alt2": cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        ),
        "profile": cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        ),
    }
    if any(cascade.empty() for cascade in cascades.values()):
        raise RuntimeError("OpenCV Haar cascade could not be loaded")

    detector_steps = {
        "strict": [
            ("frontal_default", 1.10, 4, (40, 40)),
        ],
        "hybrid": [
            ("frontal_default", 1.10, 4, (40, 40)),
            ("frontal_alt2", 1.10, 4, (40, 40)),
            ("frontal_default", 1.05, 3, (30, 30)),
            ("profile", 1.10, 4, (40, 40)),
        ],
        "loose": [
            ("frontal_default", 1.05, 3, (30, 30)),
        ],
    }

    face_boxes: list[tuple[int, int, int, int] | None] = []
    for object_id in metadata["objectid"].astype(int).tolist():
        image = Image.open(image_dir / f"{object_id}.jpg").convert("RGB")
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        chosen_box = None
        for cascade_name, scale_factor, min_neighbors, min_size in detector_steps[detector_mode]:
            detections = cascades[cascade_name].detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
            )
            if len(detections) == 0:
                continue
            chosen_box = _select_face_box(detections, image.width, image.height)
            _, _, w, h = chosen_box
            rel_area = (w * h) / max(image.width * image.height, 1)
            if rel_area < min_rel_area:
                chosen_box = None
            break
        face_boxes.append(chosen_box)
    return face_boxes


def extract_face_vit_embeddings(metadata, image_dir, face_boxes, batch_size, num_workers, checkpoint_path, device, padding):
    valid_entries = [(idx, box) for idx, box in enumerate(face_boxes) if box is not None]
    embeddings = np.zeros((len(metadata), 512), dtype=np.float32)
    mask = np.zeros(len(metadata), dtype=bool)

    if not valid_entries:
        return embeddings, mask

    original_indices = [idx for idx, _ in valid_entries]
    valid_boxes = [box for _, box in valid_entries if box is not None]
    valid_metadata = metadata.iloc[original_indices].reset_index(drop=True)
    dataset = FaceCropVitDataset(valid_metadata, image_dir, original_indices, valid_boxes, padding)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model("vit", embed_dim=512, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        for images, indices in tqdm(loader, desc="Embedding face crops"):
            features = model(images.to(device)).cpu().numpy()
            embeddings[indices.numpy()] = features
            mask[indices.numpy()] = True

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    return embeddings, mask


def fuse_embeddings(resnet_emb, efficientnet_emb, vit_emb, resnet_w, efficientnet_w, vit_w):
    fused = np.concatenate(
        [
            resnet_emb * resnet_w,
            efficientnet_emb * efficientnet_w,
            vit_emb * vit_w,
        ],
        axis=1,
    ).astype(np.float32)
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    return fused / np.clip(norms, 1e-12, None)


def apply_database_side_augmentation(embeddings, top_k, alpha):
    if top_k <= 0:
        return embeddings

    similarity = embeddings @ embeddings.T
    augmented = np.empty_like(embeddings)
    for idx in range(len(embeddings)):
        neighbors = np.argsort(-similarity[idx])[: top_k + 1]
        weights = np.array(
            [max(similarity[idx, neighbor], 0.0) ** alpha for neighbor in neighbors],
            dtype=np.float32,
        )
        vector = (embeddings[neighbors] * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-12)
        augmented[idx] = vector / max(np.linalg.norm(vector), 1e-12)

    return augmented


def search_indices_with_face_rerank(embeddings, query_index, top_k, face_embeddings=None, face_mask=None, face_weight=0.0, face_rerank_topn=0):
    scores = embeddings @ embeddings[query_index]
    scores[query_index] = -math.inf

    if (
        face_embeddings is None
        or face_mask is None
        or face_weight <= 0.0
        or face_rerank_topn <= 0
        or not bool(face_mask[query_index].item())
    ):
        top_indices = torch.topk(scores, k=min(top_k, len(scores) - 1)).indices.tolist()
        return top_indices, {idx: float(scores[idx]) for idx in top_indices}

    rerank_k = min(max(top_k, face_rerank_topn), len(scores) - 1)
    initial_indices = torch.topk(scores, k=rerank_k).indices
    rerank_scores = scores[initial_indices].clone()

    valid_positions = torch.nonzero(face_mask[initial_indices], as_tuple=False).flatten()
    if len(valid_positions) > 0:
        candidate_indices = initial_indices[valid_positions]
        face_scores = face_embeddings[candidate_indices] @ face_embeddings[query_index]
        face_scores = torch.clamp(face_scores, min=0.0)
        rerank_scores[valid_positions] += face_weight * face_scores

    order = torch.argsort(rerank_scores, descending=True)
    reranked_indices = initial_indices[order][:top_k].tolist()
    final_scores = {
        int(initial_indices[pos]): float(rerank_scores[pos])
        for pos in range(len(initial_indices))
    }
    return reranked_indices, final_scores


def collect_query_results(metadata, embeddings, label_column, top_k, face_embeddings=None, face_mask=None, face_weight=0.0, face_rerank_topn=0):
    if label_column not in metadata.columns:
        raise SystemExit(f"Unknown label column: {label_column}")

    labels = metadata[label_column].fillna("").astype(str).tolist()
    query_results: list[dict[str, object]] = []

    for query_index, query_label in enumerate(labels):
        if not query_label:
            continue

        relevant = [idx for idx, value in enumerate(labels) if value == query_label and idx != query_index]
        if not relevant:
            continue

        retrieved_indices, final_scores = search_indices_with_face_rerank(
            embeddings=embeddings,
            query_index=query_index,
            top_k=top_k,
            face_embeddings=face_embeddings,
            face_mask=face_mask,
            face_weight=face_weight,
            face_rerank_topn=face_rerank_topn,
        )

        p1, _, _, _ = precision_recall_ap(relevant, retrieved_indices, 1)
        p5, r5, _, _ = precision_recall_ap(relevant, retrieved_indices, 5)
        p10, r10, ap10, rr = precision_recall_ap(relevant, retrieved_indices, 10)

        first_relevant_rank = None
        neighbors = []
        relevant_set = set(relevant)
        for rank, idx in enumerate(retrieved_indices, start=1):
            row = metadata.iloc[idx]
            is_relevant = idx in relevant_set
            if is_relevant and first_relevant_rank is None:
                first_relevant_rank = rank
            neighbors.append(
                {
                    "rank": rank,
                    "objectid": int(row["objectid"]),
                    "title": str(row.get("title", "")),
                    "artist_name": str(row.get(label_column, "")),
                    "score": float(final_scores.get(idx, 0.0)),
                    "relevant": is_relevant,
                }
            )

        query_row = metadata.iloc[query_index]
        query_results.append(
            {
                "query_index": query_index,
                "query_objectid": int(query_row["objectid"]),
                "query_title": str(query_row.get("title", "")),
                "query_artist_name": str(query_row.get(label_column, "")),
                "relevant_count": len(relevant),
                "first_relevant_rank": first_relevant_rank,
                "precision@1": p1,
                "precision@5": p5,
                "precision@10": p10,
                "recall@5": r5,
                "recall@10": r10,
                "ap@10": ap10,
                "rr": rr,
                "neighbors": neighbors,
            }
        )

    return query_results


def summarize_query_results(query_results):
    if not query_results:
        return {
            "precision@1": 0.0,
            "precision@5": 0.0,
            "precision@10": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "map@10": 0.0,
            "mrr": 0.0,
            "queries_evaluated": 0.0,
        }

    def avg(key: str) -> float:
        return sum(float(row[key]) for row in query_results) / len(query_results)

    return {
        "precision@1": avg("precision@1"),
        "precision@5": avg("precision@5"),
        "precision@10": avg("precision@10"),
        "recall@5": avg("recall@5"),
        "recall@10": avg("recall@10"),
        "map@10": avg("ap@10"),
        "mrr": avg("rr"),
        "queries_evaluated": float(len(query_results)),
    }


def build_qualitative_examples(query_results):
    if not query_results:
        return []

    selected_ids: set[int] = set()

    def choose(candidates: list[dict[str, object]]) -> dict[str, object] | None:
        for row in candidates:
            query_objectid = int(row["query_objectid"])
            if query_objectid not in selected_ids:
                selected_ids.add(query_objectid)
                return row
        return None

    by_strength = sorted(
        query_results,
        key=lambda row: (
            float(row["precision@5"]),
            float(row["rr"]),
            float(row["ap@10"]),
        ),
        reverse=True,
    )
    by_difficulty = sorted(
        query_results,
        key=lambda row: (
            float(row["ap@10"]),
            float(row["rr"]),
            float(row["precision@5"]),
        ),
    )
    by_middle = sorted(query_results, key=lambda row: float(row["ap@10"]))
    middle_index = len(by_middle) // 2
    middle_candidates = by_middle[middle_index:] + list(reversed(by_middle[:middle_index]))

    picks = [
        ("strong_retrieval", choose(by_strength)),
        ("typical_retrieval", choose(middle_candidates)),
        ("hard_case", choose(by_difficulty)),
    ]

    examples = []
    for label, row in picks:
        if row is None:
            continue

        top5 = list(row["neighbors"])[:5]
        first_rank = row["first_relevant_rank"]
        if label == "strong_retrieval":
            comment = "top of the list is all the same artist - clean result"
        elif label == "typical_retrieval":
            comment = "some good matches but also pulls in style neighbors that aren't the right artist"
        else:
            if first_rank is None:
                comment = "hard case - model goes by composition/style, misses the artist"
            else:
                comment = "first relevant match comes in late, ranking by coarse visual similarity"

        examples.append(
            {
                "case": label,
                "query_objectid": int(row["query_objectid"]),
                "query_title": str(row["query_title"]),
                "query_artist_name": str(row["query_artist_name"]),
                "first_relevant_rank": first_rank,
                "precision@5": float(row["precision@5"]),
                "ap@10": float(row["ap@10"]),
                "comment": comment,
                "top5_neighbors": top5,
            }
        )

    return examples


def write_qualitative_examples(output_dir, examples):
    lines = ["# Qualitative Examples", ""]
    for example in examples:
        lines.append(f"## {str(example['case']).replace('_', ' ').title()}")
        lines.append(
            f"Query: {example['query_objectid']} | {example['query_title']} | {example['query_artist_name']}"
        )
        lines.append(
            f"First relevant rank: {example['first_relevant_rank']} | "
            f"Precision@5: {example['precision@5']:.3f} | AP@10: {example['ap@10']:.3f}"
        )
        lines.append(f"Comment: {example['comment']}")
        lines.append("Top-5 retrieved:")
        for neighbor in example["top5_neighbors"]:
            marker = "relevant" if neighbor["relevant"] else "non-relevant"
            lines.append(
                f"- {neighbor['rank']}. {neighbor['objectid']} | {neighbor['title']} | "
                f"{neighbor['artist_name']} | score={neighbor['score']:.4f} | {marker}"
            )
        lines.append("")

    (output_dir / "qualitative_examples.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    (output_dir / "qualitative_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")


def write_report(report_path, args, metrics, qualitative_examples, n_images):
    improvement = {
        key: metrics[key] / value
        for key, value in COMPETITOR_BASELINE.items()
        if key in metrics and value > 0
    }

    report = {
        "task2_corpus_size": n_images,
        "weights": {
            "resnet50": args.resnet_weight,
            "efficientnet_b0": args.efficientnet_weight,
            "vit_finetuned": args.vit_weight,
        },
        "dba": {
            "top_k": args.dba_k,
            "alpha": args.dba_alpha,
        },
        "face_rerank": {
            "enabled": bool(args.face_weight > 0 and args.face_rerank_topn > 0),
            "detector_mode": args.face_detector_mode,
            "weight": args.face_weight,
            "top_n": args.face_rerank_topn,
            "padding": args.face_padding,
            "min_rel_area": args.face_min_rel_area,
        },
        "metrics": metrics,
        "qualitative_examples": qualitative_examples,
        "competitor_baseline": COMPETITOR_BASELINE,
        "improvement_vs_competitor": improvement,
    }

    text_lines = [
        f"Task-2 fair benchmark images: {n_images}",
        f"Weights: resnet50={args.resnet_weight:.2f}, efficientnet_b0={args.efficientnet_weight:.2f}, vit_finetuned={args.vit_weight:.2f}",
        f"DBA: top_k={args.dba_k}, alpha={args.dba_alpha:.2f}",
        (
            "Face rerank: "
            f"mode={args.face_detector_mode}, weight={args.face_weight:.2f}, "
            f"top_n={args.face_rerank_topn}, padding={args.face_padding:.2f}, "
            f"min_rel_area={args.face_min_rel_area:.4f}"
        ),
        "",
        "=== OUR RESULTS ===",
    ]
    for key in ["precision@1", "precision@5", "precision@10", "recall@5", "recall@10", "map@10", "mrr", "queries_evaluated"]:
        if key in metrics:
            text_lines.append(f"{key}: {metrics[key]:.6f}")

    text_lines.extend(
        [
            "",
            "=== COMPETITOR BASELINE ===",
            f"precision@10: {COMPETITOR_BASELINE['precision@10']:.6f}",
            f"recall@10: {COMPETITOR_BASELINE['recall@10']:.6f}",
            f"map@10: {COMPETITOR_BASELINE['map@10']:.6f}",
            f"mrr: {COMPETITOR_BASELINE['mrr']:.6f}",
            "",
            "=== IMPROVEMENT ===",
        ]
    )
    for key in ["precision@10", "recall@10", "map@10", "mrr"]:
        if key in improvement:
            text_lines.append(f"{key}: {improvement[key]:.2f}x")

    if qualitative_examples:
        text_lines.extend(["", "=== QUALITATIVE EXAMPLES ==="])
        for example in qualitative_examples:
            text_lines.append(
                f"{str(example['case']).replace('_', ' ').title()}: "
                f"{example['query_objectid']} | {example['query_title']} | {example['query_artist_name']}"
            )
            text_lines.append(f"  {example['comment']}")
            for neighbor in example["top5_neighbors"][:3]:
                marker = "relevant" if neighbor["relevant"] else "non-relevant"
                text_lines.append(
                    f"  - {neighbor['rank']}. {neighbor['objectid']} | {neighbor['title']} | "
                    f"{neighbor['artist_name']} | {marker}"
                )

    report_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    report_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_lines = [
        "Final fair benchmark summary",
        f"precision@1: {metrics['precision@1']:.6f}",
        f"precision@5: {metrics['precision@5']:.6f}",
        f"precision@10: {metrics['precision@10']:.6f}",
        f"recall@5: {metrics['recall@5']:.6f}",
        f"recall@10: {metrics['recall@10']:.6f}",
        f"map@10: {metrics['map@10']:.6f}",
        f"mrr: {metrics['mrr']:.6f}",
    ]
    (report_path.parent / "selection_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    device = get_device()
    print(f"Device: {device}")

    if args.download_missing:
        source_metadata = pd.read_csv(args.metadata_csv).head(args.limit)
        downloaded = download_missing_images(source_metadata, args.image_dir, args.max_workers)
        print(f"Downloaded {downloaded} missing images")

    metadata = load_task2_metadata(args.metadata_csv, args.image_dir, args.limit)
    if metadata.empty:
        raise SystemExit("No images available for the benchmark.")

    print(f"Benchmark rows with local images: {len(metadata)}")

    resnet_embeddings = extract_torchvision_embeddings(
        metadata,
        args.image_dir,
        "resnet50",
        args.batch_size,
        args.num_workers,
        device,
    )
    efficientnet_embeddings = extract_torchvision_embeddings(
        metadata,
        args.image_dir,
        "efficientnet_b0",
        args.batch_size,
        args.num_workers,
        device,
    )
    vit_embeddings = extract_finetuned_vit_embeddings(
        metadata,
        args.image_dir,
        args.batch_size,
        args.num_workers,
        args.vit_checkpoint,
        device,
    )

    fused_embeddings = fuse_embeddings(
        resnet_embeddings,
        efficientnet_embeddings,
        vit_embeddings,
        args.resnet_weight,
        args.efficientnet_weight,
        args.vit_weight,
    )
    final_embeddings = apply_database_side_augmentation(
        fused_embeddings,
        top_k=args.dba_k,
        alpha=args.dba_alpha,
    )

    embeddings_tensor = torch.tensor(final_embeddings)
    face_embeddings_tensor = None
    face_mask_tensor = None
    if args.face_weight > 0.0 and args.face_rerank_topn > 0:
        face_boxes = detect_face_boxes(
            metadata=metadata,
            image_dir=args.image_dir,
            detector_mode=args.face_detector_mode,
            min_rel_area=args.face_min_rel_area,
        )
        face_embeddings, face_mask = extract_face_vit_embeddings(
            metadata=metadata,
            image_dir=args.image_dir,
            face_boxes=face_boxes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            checkpoint_path=args.vit_checkpoint,
            device=device,
            padding=args.face_padding,
        )
        face_embeddings_tensor = torch.tensor(face_embeddings)
        face_mask_tensor = torch.tensor(face_mask)
        print(f"Face detections: {int(face_mask.sum())}/{len(face_mask)}")

    query_results = collect_query_results(
        metadata,
        embeddings_tensor,
        label_column=args.label_column,
        top_k=args.top_k,
        face_embeddings=face_embeddings_tensor,
        face_mask=face_mask_tensor,
        face_weight=args.face_weight,
        face_rerank_topn=args.face_rerank_topn,
    )
    metrics = summarize_query_results(query_results)
    qualitative_examples = build_qualitative_examples(query_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output_dir / "index_metadata.csv", index=False)
    torch.save(embeddings_tensor, args.output_dir / "embeddings.pt")
    write_qualitative_examples(args.output_dir, qualitative_examples)
    write_report(args.output_dir / "report.txt", args, metrics, qualitative_examples, len(metadata))

    print("\n=== Fair Task-2 Benchmark ===")
    for key in ["precision@1", "precision@5", "precision@10", "recall@5", "recall@10", "map@10", "mrr", "queries_evaluated"]:
        if key in metrics:
            print(f"{key}: {metrics[key]:.6f}")
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
