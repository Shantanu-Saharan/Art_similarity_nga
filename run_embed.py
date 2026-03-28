"""
Generate embeddings for all downloaded NGA portrait images.

If a trained checkpoint exists (from run_train.py), it loads that.
Otherwise uses the pretrained backbone directly — still works fine
for similarity search, just not as tuned to this specific dataset.

Run after run_prepare.py (and optionally run_train.py):
    python run_embed.py
    python run_embed.py --model resnet50   # faster, less accurate
    python run_embed.py --model vit        # slower, better
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from src.models.face_pose_extractor import FacePoseEmbeddingExtractor

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import src.config as config
from src.data.image_dataset import PortraitDataset
from src.models.embedding_model import build_model, build_transform, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings for NGA portrait images")
    parser.add_argument("--metadata_csv", type=str,
                        default=str(config.FILTERED_METADATA_CSV))
    parser.add_argument("--image_dir", type=str,
                        default=str(config.IMAGE_DIR))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default="vit",
                        choices=["vit", "resnet50", "resnet18"])
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pth checkpoint from run_train.py. "
                            "If not given, will look for the default path.")
    parser.add_argument(
    "--feature_mode",
    type=str,
    default="full",
    choices=["full", "face", "combined", "pose"],
    help="How to build embeddings: full image, face crop, combined, or pose-aware"
)
    return parser.parse_args()


def load_model(args, device):
    model = build_model(model_name=args.model, embed_dim=args.embed_dim, device=device)

    # check for a fine-tuned checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        default_ckpt = config.MODELS_DIR / f"portrait_{args.model}_triplet.pth"
        if default_ckpt.exists():
            ckpt_path = str(default_ckpt)

    if ckpt_path is not None:
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print("No checkpoint found, using pretrained backbone weights only.")
        # tried cosine on raw resnet features — wasn't stable

    return model


def extract_single_embedding(args, model, face_pose_extractor, image_path, device, transform=None):
    # returns (embedding, info_dict) — routes to face/pose extractor or plain model
    if args.feature_mode == "full":
        # Use existing model for full-image embeddings
        image = Image.open(image_path).convert("RGB")
        if transform is None:
            transform = build_transform()
        tensor = transform(image).unsqueeze(0).to(device)  # type: ignore[union-attr]
        with torch.no_grad():
            embedding = model(tensor).squeeze(0).cpu().numpy()
        info = {"feature_mode": "full", "face_found": False, "pose_found": False, "mode_used": "full"}
        return embedding, info
    else:
        # Use face/pose extractor for non-full modes
        image = Image.open(image_path).convert("RGB")
        embedding, info = face_pose_extractor.extract(image, mode=args.feature_mode)
        info["feature_mode"] = args.feature_mode
        return embedding, info


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    if args.feature_mode == "full":
        # Use existing batched pipeline for full mode
        transform = build_transform()
        dataset = PortraitDataset(
            csv_path=args.metadata_csv,
            image_dir=args.image_dir,
            transform=transform,
        )
        print(f"Dataset size: {len(dataset)} portraits")

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = load_model(args, device)
        model.eval()

        all_embeddings = []
        all_rows = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting embeddings"):
                images = batch["image"].to(device)
                feats = model(images)  # already normalized so dot product = cosine
                all_embeddings.append(feats.cpu().numpy())

                for i in range(len(batch["filename"])):
                    all_rows.append({
                        "filename": batch["filename"][i],
                        "objectid": int(batch["objectid"][i]),
                        "title": batch["title"][i],
                        "artist_hint": batch["artist_hint"][i],
                        "displaydate": batch["displaydate"][i],
                        "model": args.model,
                        "feature_mode": args.feature_mode,
                        "face_found": False,
                        "pose_found": False,
                        "mode_used": "full",
                    })
    else:
        # Process image-by-image for face/combined/pose modes
        model = load_model(args, device)
        model.eval()
        transform = build_transform()
        # leaving this hardcoded for now, not worth parameterizing
        face_pose_extractor = FacePoseEmbeddingExtractor(model, transform, device)
        
        # Load dataset without transform for individual processing
        dataset = PortraitDataset(
            csv_path=args.metadata_csv,
            image_dir=args.image_dir,
            transform=None,
        )
        print(f"Dataset size: {len(dataset)} portraits (processing individually for {args.feature_mode} mode)")
        
        all_embeddings = []
        all_rows = []
        
        for i in tqdm(range(len(dataset)), desc="Extracting embeddings"):
            sample = dataset[i]
            image_path = Path(args.image_dir) / sample["filename"]
            
            try:
                embedding, info = extract_single_embedding(args, model, face_pose_extractor, image_path, device)
                all_embeddings.append(embedding)
                
                row = {
                    "filename": sample["filename"],
                    "objectid": int(sample["objectid"]),
                    "title": sample["title"],
                    "artist_hint": sample["artist_hint"],
                    "displaydate": sample["displaydate"],
                    "model": args.model,
                    "feature_mode": info.get("feature_mode", args.feature_mode),
                    "face_found": info.get("face_found", False),
                    "pose_found": info.get("pose_found", False),
                    "mode_used": info.get("mode_used", args.feature_mode),
                }
                all_rows.append(row)
            except Exception as e:
                print(f"Warning: Failed to process {sample['filename']}: {e}")
                # Skip this image or use a fallback embedding
                continue

    if args.feature_mode == "full":
        embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
    else:
        embeddings = np.stack(all_embeddings).astype("float32")

    # save with feature-aware naming
    embeddings_path = config.PROCESSED_DIR / f"embeddings_{args.model}_{args.feature_mode}.npy"
    metadata_path = config.PROCESSED_DIR / f"embedding_metadata_{args.model}_{args.feature_mode}.csv"

    np.save(embeddings_path, embeddings)
    pd.DataFrame(all_rows).to_csv(metadata_path, index=False)

    print(f"\nSaved embeddings: {embeddings_path}")
    print(f"Saved metadata:   {metadata_path}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
