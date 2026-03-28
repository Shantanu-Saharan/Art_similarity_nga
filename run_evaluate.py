"""
Standalone evaluation script — run this after building the FAISS index.

Computes Precision@K, mAP, and SSIM across a sample of query images
and prints a summary table.

Usage:
    python run_evaluate.py
    python run_evaluate.py --model vit --n_queries 100 --top_k 10
    python run_evaluate.py --model resnet50

I put this as a top-level script because calling
"python src/evaluation/metrics.py" felt awkward and I kept forgetting the path.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import src.config as config
from src.evaluation.metrics import run_evaluation


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the portrait similarity retrieval system")
    p.add_argument("--model", type=str, default="vit",
                   choices=["vit", "resnet50", "resnet18"])
    p.add_argument("--feature_mode", type=str, default="full",
                   choices=["full", "face", "combined", "pose"],
                   help="Feature mode for evaluation")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_queries", type=int, default=50,
                   help="How many query images to sample for evaluation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    embeddings_path = config.PROCESSED_DIR / f"embeddings_{args.model}_{args.feature_mode}.npy"
    metadata_path = config.PROCESSED_DIR / f"embedding_metadata_{args.model}_{args.feature_mode}.csv"

    if not embeddings_path.exists():
        print(f"Embeddings not found at: {embeddings_path}")
        print("Run run_embed.py first.")
        sys.exit(1)

    if not metadata_path.exists():
        print(f"Metadata CSV not found at: {metadata_path}")
        sys.exit(1)

    print(f"Evaluating model: {args.model}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Queries: {args.n_queries}, top-K: {args.top_k}")
    print()

    results = run_evaluation(
        embeddings_path=str(embeddings_path),
        metadata_path=str(metadata_path),
        image_dir=str(config.IMAGE_DIR),
        top_k=args.top_k,
        n_queries=args.n_queries,
        seed=args.seed,
    )

    print("=== Evaluation Results ===")
    print(f"  Queries evaluated:     {results['n_queries']}")
    print(f"  Precision@1:           {results['mean_precision_at_1']:.4f}")
    print(f"  Precision@5:           {results['mean_precision_at_5']:.4f}")
    print(f"  Precision@{args.top_k}:          {results.get(f'mean_precision_at_{args.top_k}', 0.0):.4f}")
    print(f"  mAP:                   {results['mean_average_precision']:.4f}")
    if results.get("mean_ssim_top3") is not None:
        print(f"  Mean SSIM (top-3):     {results['mean_ssim_top3']:.4f}")
    print()

    # also save results to a text file so I can reference it in the README later
    out_path = config.OUTPUTS_DIR / f"eval_results_{args.model}_{args.feature_mode}.txt"
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Feature mode: {args.feature_mode}\n")
        f.write(f"n_queries: {results['n_queries']}\n")
        f.write(f"top_k: {args.top_k}\n")
        for k, v in results.items():
            if v is not None:
                f.write(f"{k}: {v:.4f}\n")

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()