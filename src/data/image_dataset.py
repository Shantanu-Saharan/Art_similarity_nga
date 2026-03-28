from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PortraitDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, only_downloaded=True):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.df = pd.read_csv(self.csv_path)

        if only_downloaded and "downloaded" in self.df.columns:
            self.df = self.df[self.df["downloaded"] == True].copy()

        self.df = self.df.reset_index(drop=True)

        # explicit check — silent path bugs are hard to debug
        if "filename" not in self.df.columns:
            raise ValueError("Expected a 'filename' column in the metadata CSV")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / row["filename"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        sample = {
            "image": image,
            "filename": row["filename"],
            "objectid": int(row["objectid"]),
            "title": str(row.get("title", "")),
            "artist_hint": str(row.get("artist_hint", "")),
            "displaydate": str(row.get("displaydate", "")),
        }
        return sample
