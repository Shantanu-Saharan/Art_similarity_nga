"""
Triplet sampling from NGA portrait metadata.

Positives = same artist, negatives = different artist. Weak labels but
good enough at this scale. Tried date ranges too but artist gives cleaner signal.
"""

import random
from pathlib import Path

import pandas as pd

import src.config as config


def build_triplets_from_metadata(csv_path=None, min_per_artist=3, seed=42):
    # builds (anchor, positive, negative) filename triplets from portrait_subset.csv
    if csv_path is None:
        csv_path = config.FILTERED_METADATA_CSV

    df = pd.read_csv(csv_path)

    # only keep successfully downloaded images
    if "downloaded" in df.columns:
        df = df[df["downloaded"] == True].copy()

    df = df.reset_index(drop=True)

    # group by artist — this is our "class" for triplet sampling
    # artist_hint column was set in nga_loader from the attribution field
    artist_groups = {}
    for _, row in df.iterrows():
        artist = str(row["artist_hint"]).strip().lower()
        if artist in {"unknown", "nan", "", "none"}:
            continue
        if artist not in artist_groups:
            artist_groups[artist] = []
        artist_groups[artist].append(row["filename"])

    # filter out artists with too few paintings to form triplets
    artist_groups = {
        k: v for k, v in artist_groups.items()
        if len(v) >= min_per_artist
    }

    if len(artist_groups) < 2:
        raise ValueError(
            f"Not enough artists with >= {min_per_artist} paintings to build triplets. "
            "Try lowering min_per_artist or downloading more data."
        )

    all_artists = list(artist_groups.keys())
    rng = random.Random(seed)

    triplets = []

    for artist, paintings in artist_groups.items():
        # sample as many triplets as there are paintings for this artist
        # don't want huge artists to dominate the training set
        n_triplets = len(paintings)

        for _ in range(n_triplets):
            # pick anchor and positive from same artist (different images)
            if len(paintings) < 2:
                continue
            anchor, positive = rng.sample(paintings, 2)

            # pick a negative from a different artist
            neg_artist = rng.choice([a for a in all_artists if a != artist])
            negative = rng.choice(artist_groups[neg_artist])

            triplets.append({
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
            })

    triplet_df = pd.DataFrame(triplets)
    triplet_df = triplet_df.drop_duplicates().reset_index(drop=True)

    return triplet_df


def save_triplets(triplet_df, output_path=None):
    if output_path is None:
        output_path = config.PROCESSED_DIR / "triplets.csv"
    triplet_df.to_csv(output_path, index=False)
    return output_path


class TripletDataset:
    # reads triplets.csv and returns (anchor, positive, negative) image tensors
    from torch.utils.data import Dataset

    def __init__(self, triplets_csv, image_dir, transform=None):
        import pandas as pd
        from pathlib import Path
        from torchvision import transforms as T

        self.df = pd.read_csv(triplets_csv)
        self.image_dir = Path(image_dir)
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image

        row = self.df.iloc[idx]
        a = Image.open(self.image_dir / row["anchor"]).convert("RGB")
        p = Image.open(self.image_dir / row["positive"]).convert("RGB")
        n = Image.open(self.image_dir / row["negative"]).convert("RGB")

        return self.transform(a), self.transform(p), self.transform(n)
