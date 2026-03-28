import re
import time
from io import BytesIO

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import src.config as config


def _make_dirs():
    config.NGA_DIR.mkdir(parents=True, exist_ok=True)
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_csvs():
    if not config.OBJECTS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {config.OBJECTS_CSV}. Put objects.csv inside data/raw/nga/"
        )

    if not config.PUBLISHED_IMAGES_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {config.PUBLISHED_IMAGES_CSV}. Put published_images.csv inside data/raw/nga/"
        )

    objects_df = pd.read_csv(config.OBJECTS_CSV, low_memory=False)
    images_df = pd.read_csv(config.PUBLISHED_IMAGES_CSV, low_memory=False)

    objects_df.columns = [c.strip().lower() for c in objects_df.columns]
    images_df.columns = [c.strip().lower() for c in images_df.columns]

    return objects_df, images_df


def _portrait_regex():
    escaped = [re.escape(x) for x in config.TITLE_KEYWORDS]
    return "|".join(escaped)


def _filter_objects(objects_df):
    df = objects_df.copy()

    if "classification" not in df.columns:
        raise ValueError("objects.csv does not contain 'classification' column")

    if "title" not in df.columns:
        raise ValueError("objects.csv does not contain 'title' column")

    df = df[df["classification"].isin(config.ALLOWED_CLASSIFICATIONS)]

    if "isvirtual" in df.columns:
        df = df[df["isvirtual"].fillna(0) == 0]

    pattern = _portrait_regex()
    df = df[df["title"].astype(str).str.contains(pattern, case=False, na=False, regex=True)]

    if "objectid" not in df.columns:
        raise ValueError("objects.csv does not contain 'objectid' column")

    df = df.drop_duplicates(subset=["objectid"]).copy()
    return df


def _keep_primary_images(images_df):
    df = images_df.copy()

    if "viewtype" in df.columns:
        df = df[df["viewtype"].astype(str).str.lower() == "primary"]

    return df


def _merge_objects_and_images(objects_df, images_df):
    if "depictstmsobjectid" not in images_df.columns:
        raise ValueError("published_images.csv does not contain 'depictstmsobjectid' column")

    if "iiifurl" not in images_df.columns:
        raise ValueError("published_images.csv does not contain 'iiifurl' column")

    merged = objects_df.merge(
        images_df,
        left_on="objectid",
        right_on="depictstmsobjectid",
        how="inner"
    )

    merged = merged[merged["iiifurl"].notna()].copy()
    merged = merged.drop_duplicates(subset=["objectid"]).reset_index(drop=True)
    return merged


def _build_download_url(iiif_url):
    iiif_url = str(iiif_url).strip().rstrip("/")
    return f"{iiif_url}/full/{config.DOWNLOAD_SIZE}/0/default.jpg"


def _safe_artist_name(row):
    for col in ["attribution", "displayname", "constituentid", "schoolorstyle"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return "unknown"


def _download_one_image(iiif_url, save_path):
    url = _build_download_url(iiif_url)

    try:
        response = requests.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(save_path)
        return True, None

    except (requests.RequestException, UnidentifiedImageError, OSError) as e:
        return False, str(e)


def prepare_portrait_subset(limit=None, redownload=False):
    _make_dirs()

    objects_df, images_df = _read_csvs()

    filtered_objects = _filter_objects(objects_df)
    primary_images = _keep_primary_images(images_df)
    merged = _merge_objects_and_images(filtered_objects, primary_images)

    if limit is not None:
        merged = merged.sample(
            n=min(limit, len(merged)),
            random_state=config.RANDOM_SEED
        ).reset_index(drop=True)

    rows_for_csv = []

    print(f"Filtered portrait-like paintings with image links: {len(merged)}")

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Downloading portraits"):
        object_id = int(row["objectid"])
        filename = f"painting_{object_id}.jpg"
        save_path = config.IMAGE_DIR / filename

        downloaded = False
        error_message = None

        if save_path.exists() and not redownload:
            downloaded = True
        else:
            downloaded, error_message = _download_one_image(row["iiifurl"], save_path)

        rows_for_csv.append(
            {
                "objectid": object_id,
                "filename": filename,
                "title": str(row.get("title", "")),
                "classification": str(row.get("classification", "")),
                "displaydate": str(row.get("displaydate", "")),
                "artist_hint": _safe_artist_name(row),
                "iiifurl": str(row.get("iiifurl", "")),
                "downloaded": downloaded,
                "error": "" if error_message is None else error_message,
            }
        )

        time.sleep(config.SLEEP_BETWEEN_DOWNLOADS)

    out_df = pd.DataFrame(rows_for_csv)
    out_df.to_csv(config.FILTERED_METADATA_CSV, index=False)

    success_count = int(out_df["downloaded"].sum())
    print(f"Saved metadata to: {config.FILTERED_METADATA_CSV}")
    print(f"Images downloaded successfully: {success_count}/{len(out_df)}")

    return out_df