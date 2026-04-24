# portrait retrieval training — PK sampler, supcon + triplet + CE

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import src.config as config
from src.models.embedding_model import build_model, get_device


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.04),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class AllSourceDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

        artists = sorted(self.df["artist"].unique())
        self.artist_to_idx = {a: i for i, a in enumerate(artists)}
        self.labels = torch.tensor(
            [self.artist_to_idx[a] for a in self.df["artist"]], dtype=torch.long
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["image_dir"]) / row["filename"]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


class PKSampler(Sampler):
    # P artists x K images per batch

    def __init__(self, labels, batch_size, samples_per_class=4):
        self.labels = np.array(labels, dtype=np.int64)
        self.batch_size = batch_size
        self.samples_per_class = max(2, samples_per_class)

        self.class_to_indices = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            self.class_to_indices[int(lab)].append(idx)

        self.classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) >= 2]
        if len(self.classes) < 2:
            raise ValueError("not enough classes - need at least 2 artists with 2+ images each")

        self.num_classes_per_batch = max(2, batch_size // self.samples_per_class)
        self.effective_batch_size = self.num_classes_per_batch * self.samples_per_class

        self.num_batches = max(1, len(self.labels) // self.effective_batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_classes = random.sample(
                self.classes,
                k=min(self.num_classes_per_batch, len(self.classes))
            )

            batch = []
            for c in chosen_classes:
                idxs = self.class_to_indices[c]
                if len(idxs) >= self.samples_per_class:
                    batch.extend(random.sample(idxs, self.samples_per_class))
                else:
                    batch.extend(random.choices(idxs, k=self.samples_per_class))

            random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.num_batches * self.effective_batch_size


class SupConLoss(nn.Module):
    # supcon loss, embeddings should be l2-normalized

    def __init__(self, temperature=0.10):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        B = embeddings.size(0)

        sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask) - torch.eye(B, device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_count + 1e-12)

        valid = pos_count > 0
        if valid.any():
            loss = -mean_log_prob_pos[valid].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss


class BatchHardTripletLoss(nn.Module):
    # batch-hard triplet with cosine distance

    def __init__(self, margin=0.20):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        sim = embeddings @ embeddings.T
        dist = 1.0 - sim

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        pos_mask = labels_eq & (~eye)
        neg_mask = ~labels_eq

        hardest_pos = torch.where(pos_mask, dist, torch.full_like(dist, -1e9)).max(dim=1).values
        hardest_neg = torch.where(neg_mask, dist, torch.full_like(dist, 1e9)).min(dim=1).values

        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device)

        loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)
        return loss.mean()


def freeze_vit_blocks(model, keep_last_n=6):
    backbone = model.backbone
    if not hasattr(backbone, "encoder") or not hasattr(backbone.encoder, "layers"):
        print("  ViT freeze skipped: unexpected backbone structure")
        return

    layers = list(backbone.encoder.layers)
    freeze_up_to = max(0, len(layers) - keep_last_n)

    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad = i >= freeze_up_to

    if hasattr(backbone, "conv_proj"):
        for p in backbone.conv_proj.parameters():
            p.requires_grad = False

    if hasattr(backbone, "class_token") and isinstance(backbone.class_token, torch.nn.Parameter):
        backbone.class_token.requires_grad = False

    print(
        f"  ViT: frozen first {freeze_up_to} / {len(layers)} blocks, "
        f"training last {keep_last_n} + projection head"
    )


def freeze_resnet_layers(model, keep_layer: str = "layer3"):
    order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    freeze = True
    for name in order:
        if name == keep_layer:
            freeze = False
        layer = getattr(model.backbone, name, None)
        if layer is None:
            continue
        for p in layer.parameters():
            p.requires_grad = not freeze
    print(f"  ResNet: unfreezing from {keep_layer} onwards + projection head")


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = max(self.warmup_epochs + 1, total_epochs)
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch <= self.warmup_epochs:
            scale = epoch / self.warmup_epochs
            lrs = [max(self.min_lr, base * scale) for base in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lrs = [self.min_lr + (base - self.min_lr) * cosine for base in self.base_lrs]

        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def train(args):
    seed_everything(args.seed)

    device = get_device()
    print(f"Device: {device}")

    meta_csv = Path(args.metadata_csv) if args.metadata_csv else (config.PROCESSED_DIR / "all_images_metadata.csv")
    if not meta_csv.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {meta_csv}"
        )

    df = pd.read_csv(meta_csv)
    df["artist"] = df["artist"].fillna("").astype(str).str.strip()
    unknown_mask = df["artist"].str.lower().isin({"", "unknown", "nan", "none", "unknown artist"})
    if unknown_mask.any():
        print(f"Dropping {int(unknown_mask.sum())} rows with unknown artist labels")
        df = df[~unknown_mask].reset_index(drop=True)

    counts = df["artist"].value_counts()
    valid_artists = counts[counts >= args.min_per_artist].index
    df = df[df["artist"].isin(valid_artists)].reset_index(drop=True)

    print(f"Training images: {len(df)} | Artists: {df['artist'].nunique()}")
    print("Source distribution after filtering:")
    print(df["source"].value_counts().to_string())

    dataset = AllSourceDataset(df, transform=train_transform())

    sampler = PKSampler(
        labels=dataset.labels.tolist(),
        batch_size=args.batch,
        samples_per_class=args.samples_per_class,
    )

    loader = DataLoader(
        dataset,
        batch_size=sampler.effective_batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    print(
        f"Batches per epoch: {len(loader)} | "
        f"Effective batch: {sampler.effective_batch_size} "
        f"({sampler.num_classes_per_batch} artists x {sampler.samples_per_class} imgs)"
    )

    model = build_model(model_name=args.model, embed_dim=args.embed_dim, device=device)

    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {init_path}")
        checkpoint = torch.load(init_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded init checkpoint: {init_path}")
        print(f"  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    if args.model == "vit":
        freeze_vit_blocks(model, keep_last_n=args.unfreeze_blocks)
    elif args.model in ("resnet50", "resnet18"):
        freeze_resnet_layers(model, keep_layer="layer3")

    num_artists = df["artist"].nunique()
    classifier = nn.Linear(args.embed_dim, num_artists).to(device)

    supcon_loss = SupConLoss(temperature=args.temperature)
    triplet_loss = BatchHardTripletLoss(margin=args.margin)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    backbone_params = [p for n, p in model.named_parameters() if "proj" not in n and p.requires_grad]
    head_params = list(model.proj.parameters())
    cls_params = list(classifier.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
        {"params": cls_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=1e-6,
    )

    best_loss = float("inf")
    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
    else:
        ckpt_path = config.MODELS_DIR / f"portrait_{args.model}_improved.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        classifier.train()
        scheduler.step(epoch)

        total_loss = 0.0
        total_supcon = 0.0
        total_triplet = 0.0
        total_ce = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            logits = classifier(embeddings)

            loss_supcon = supcon_loss(embeddings, labels)
            loss_triplet = triplet_loss(embeddings, labels)
            loss_ce = ce_loss(logits, labels)

            loss = (
                args.supcon_weight * loss_supcon
                + args.triplet_weight * loss_triplet
                + args.ce_weight * loss_ce
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_supcon += loss_supcon.item()
            total_triplet += loss_triplet.item()
            total_ce += loss_ce.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                sc=f"{loss_supcon.item():.3f}",
                tri=f"{loss_triplet.item():.3f}",
                ce=f"{loss_ce.item():.3f}",
                lr=f"{scheduler.get_last_lr()[-1]:.2e}",
            )

        avg_loss = total_loss / max(n_batches, 1)
        avg_sc = total_supcon / max(n_batches, 1)
        avg_tri = total_triplet / max(n_batches, 1)
        avg_ce = total_ce / max(n_batches, 1)

        print(
            f"Epoch {epoch}: "
            f"avg_loss={avg_loss:.4f}  "
            f"supcon={avg_sc:.4f}  triplet={avg_tri:.4f}  ce={avg_ce:.4f}  "
            f"lr={scheduler.get_last_lr()[-1]:.2e}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "embed_dim": args.embed_dim,
                    "loss": avg_loss,
                    "state_dict": model.state_dict(),
                },
                ckpt_path,
            )
            print(f"  saved checkpoint (loss={avg_loss:.4f})")

    print(f"\nTraining done. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    return ckpt_path


def parse_args():
    p = argparse.ArgumentParser(description="Improved retrieval-oriented training on 2100+ images")
    p.add_argument("--model", default="vit", choices=["vit", "resnet50", "resnet18"])
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--metadata_csv", type=str, default=None)
    p.add_argument("--init_checkpoint", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, default=None)

    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--samples_per_class", type=int, default=4)

    p.add_argument("--lr_head", type=float, default=3e-4)
    p.add_argument("--lr_backbone", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)

    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--margin", type=float, default=0.20)
    p.add_argument("--label_smoothing", type=float, default=0.10)

    p.add_argument("--supcon_weight", type=float, default=1.0)
    p.add_argument("--triplet_weight", type=float, default=0.5)
    p.add_argument("--ce_weight", type=float, default=0.5)

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--min_per_artist", type=int, default=3)
    p.add_argument("--unfreeze_blocks", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
