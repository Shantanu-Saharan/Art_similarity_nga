"""
Fine-tune the embedding model using triplet loss on NGA portrait pairs.

Run this after run_prepare.py has downloaded images.
It will first auto-build triplets if triplets.csv doesn't exist yet,
then train for a few epochs and save the best checkpoint.

Typical run:
    python run_train.py --model vit --epochs 5 --lr 1e-4
    python run_train.py --model vit --epochs 5 --loss hard_negative

The --loss hard_negative mode skips the pre-mined triplets CSV entirely and
does online hard negative mining within each batch. It needs artist labels
directly from the portrait metadata, which we already have. This is more
efficient and generally converges better since it adapts to the model as it
improves rather than using static triplets built before training starts.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import src.config as config
from src.data.triplet_builder import TripletDataset, build_triplets_from_metadata, save_triplets
from src.data.image_dataset import PortraitDataset
from src.models.embedding_model import build_model, build_transform, get_device
from src.models.triplet_loss import CosineTripletLoss, HardNegativeTripletLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Triplet fine-tuning for portrait similarity")
    parser.add_argument("--model", type=str, default="vit",
                        choices=["vit", "resnet50", "resnet18"],
                        help="Which backbone to use")
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--loss", type=str, default="standard",
                        choices=["standard", "hard_negative"],
                        help="standard: pre-mined. hard_negative: in-batch mining")
    parser.add_argument("--rebuild_triplets", action="store_true",
                        help="Force rebuild triplets.csv even if it already exists")
    return parser.parse_args()


def _build_label_map(metadata_csv):
    # artist name -> int index for HardNegativeTripletLoss
    import pandas as pd
    df = pd.read_csv(metadata_csv)
    if "downloaded" in df.columns:
        df = df[df["downloaded"] == True].copy()
    df = df.reset_index(drop=True)

    unique_artists = sorted(df["artist_hint"].dropna().unique())  # type: ignore[attr-defined]
    artist_to_idx = {a: i for i, a in enumerate(unique_artists)}
    labels = [artist_to_idx.get(str(row["artist_hint"]), -1) for _, row in df.iterrows()]
    return labels, artist_to_idx


class LabeledPortraitDataset(PortraitDataset):
    # same as PortraitDataset but also returns artist label int (for hard negative mining)

    def __init__(self, csv_path, image_dir, transform=None, artist_to_idx=None):
        super().__init__(csv_path, image_dir, transform=transform, only_downloaded=True)
        self.artist_to_idx = artist_to_idx or {}

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        artist = str(sample["artist_hint"])
        label = self.artist_to_idx.get(artist, -1)
        sample["label"] = label
        return sample


def train_standard(args, device):
    # standard training path — uses pre-mined triplets from triplets.csv
    triplets_csv = config.PROCESSED_DIR / "triplets.csv"

    if not triplets_csv.exists() or args.rebuild_triplets:
        print("Building triplets from portrait metadata...")
        triplet_df = build_triplets_from_metadata()
        save_triplets(triplet_df, triplets_csv)
        print(f"Built {len(triplet_df)} triplets -> {triplets_csv}")
    else:
        print(f"Using existing triplets at {triplets_csv}")

    transform = build_transform()
    dataset = TripletDataset(
        triplets_csv=str(triplets_csv),
        image_dir=str(config.IMAGE_DIR),
        transform=transform,
    )
    print(f"Triplet dataset size: {len(dataset)}")

    loader = DataLoader(  # pyright: ignore[reportArgumentType]
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = build_model(model_name=args.model, embed_dim=args.embed_dim, device=device)
    criterion = CosineTripletLoss(margin=args.margin)  # bumped 0.2 -> 0.3 after loss collapsed early

    backbone_params = [p for n, p in model.named_parameters()
                       if "proj" not in n and p.requires_grad]
    head_params = list(model.proj.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_loss = float("inf")
    checkpoint_path = config.MODELS_DIR / f"portrait_{args.model}_triplet.pth"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "embed_dim": args.embed_dim,
                "loss_type": "standard",
                "state_dict": model.state_dict(),
                "loss": avg_loss,
            }, checkpoint_path)
            print(f"  -> saved best checkpoint (loss={avg_loss:.4f})")

    return best_loss, checkpoint_path


def train_hard_negative(args, device):
    # TODO: merge with train_standard — same loop structure
    metadata_csv = config.FILTERED_METADATA_CSV

    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Portrait metadata CSV not found: {metadata_csv}\n"
            "Run run_prepare.py first."
        )

    _, artist_to_idx = _build_label_map(metadata_csv)
    print(f"Found {len(artist_to_idx)} unique artists for label-based mining")

    transform = build_transform()
    dataset = LabeledPortraitDataset(
        csv_path=str(metadata_csv),
        image_dir=str(config.IMAGE_DIR),
        transform=transform,
        artist_to_idx=artist_to_idx,
    )
    print(f"Dataset size: {len(dataset)} portraits")

    # larger batch for hard negative mining — more negatives to mine from
    batch_size = max(args.batch_size, 32)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = build_model(model_name=args.model, embed_dim=args.embed_dim, device=device)
    criterion = HardNegativeTripletLoss(margin=args.margin)

    backbone_params = [p for n, p in model.named_parameters()
                       if "proj" not in n and p.requires_grad]
    head_params = list(model.proj.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_loss = float("inf")
    checkpoint_path = config.MODELS_DIR / f"portrait_{args.model}_triplet.pth"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs} [hard_negative]")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = torch.tensor(batch["label"], dtype=torch.long).to(device)

            # skip batches with no valid labels (shouldn't happen but be safe)
            if (labels == -1).all():
                continue

            embeddings = model(images)
            loss = criterion(embeddings, labels)

            if loss.item() == 0.0:
                # all triplets satisfied — still do a backward pass so
                # gradients don't go stale, but don't count this batch
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", active=n_batches)

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        print(f"Epoch {epoch} avg loss: {avg_loss:.4f} ({n_batches} active batches)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "embed_dim": args.embed_dim,
                "loss_type": "hard_negative",
                "state_dict": model.state_dict(),
                "loss": avg_loss,
            }, checkpoint_path)
            print(f"  -> saved best checkpoint (loss={avg_loss:.4f})")

    return best_loss, checkpoint_path


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Loss type: {args.loss}")

    if args.loss == "standard":
        best_loss, ckpt = train_standard(args, device)
    else:
        best_loss, ckpt = train_hard_negative(args, device)

    print(f"\nTraining done. Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved at: {ckpt}")


if __name__ == "__main__":
    main()
