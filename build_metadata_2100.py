# builds the unified metadata CSV from all local image sources
# outputs: data/processed/all_images_metadata.csv

import html
import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
import src.config as config

def _clean(text: str) -> str:
    return str(text).strip() if text and str(text).lower() not in ("nan", "none", "") else ""


TITLE_STOPWORDS = {
    "portrait", "self", "study", "head", "man", "woman", "lady", "gentleman",
    "girl", "boy", "a", "an", "the", "of", "with", "in", "and", "young",
    "old", "saint", "st", "bust", "madonna", "virgin", "pope", "king",
    "queen", "christ", "holy", "family", "child", "children", "donor",
}
ARTIST_PARTICLES = {
    "van", "von", "de", "del", "della", "da", "di", "der", "den", "du",
    "la", "le", "il", "y", "the",
}
UNKNOWN_ARTIST_VALUES = {"", "unknown", "nan", "none", "unknown artist"}


def _clean_filename_token(token: str) -> str:
    token = html.unescape(str(token))
    token = token.replace("&#39;", "'")
    token = re.sub(r"(?<=[A-Za-z])39$", "", token)
    token = token.strip("._- ")
    return token


def _normalize_artist_name(text: str) -> str:
    text = html.unescape(str(text)).replace("_", " ")
    text = text.replace("&#39;", "'")
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace(",", " ")
    text = re.sub(r"(?<=[A-Za-z])39(?=\s|$)", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -_")

    if text.lower() in UNKNOWN_ARTIST_VALUES:
        return "Unknown"

    alias_map = {
        "Sebastiano del Piomb": "Sebastiano del Piombo",
        "Francesco de Rossi": "Francesco de' Rossi",
        "Francesco de Rossi Rossi": "Francesco de' Rossi",
        "El Greco Domenikos Theotokopoulos": "El Greco",
        "Goya Francisco de Goya y Lucientes": "Goya (Francisco de Goya y Lucientes)",
        "Jacometto Jacometto Veneziano": "Jacometto (Jacometto Veneziano)",
        "Giovanni Battista Gaulli Il Baciccio": "Giovanni Battista Gaulli (Il Baciccio)",
        "Anthony van Dyck": "Anthony van Dyck",
        "Gerard van Honthorst": "Gerard van Honthorst",
        "Jusepe de Ribera": "Jusepe de Ribera",
        "Cornelis de Vos": "Cornelis de Vos",
        "Bernard van Orley": "Bernard van Orley",
        "Maerten van Heemsker": "Maerten van Heemskerck",
        "Lucas Cranach the El": "Lucas Cranach the Elder",
        "Hans Holbein the You": "Hans Holbein the Younger",
    }
    return alias_map.get(text, text)


def _filename_tokens(fname: str, source: str) -> list[str]:
    parts = Path(fname).stem.split("_")
    if source == "wikiart":
        parts = parts[1:]
    elif source == "met":
        parts = parts[1:]
        if parts and parts[0].isdigit():
            parts = parts[1:]
    else:
        return []

    tokens = []
    for part in parts:
        clean = _clean_filename_token(part)
        if not clean:
            continue
        if re.match(r"^\d{4}", clean):
            break
        tokens.append(clean)
        if len(tokens) >= 8:
            break
    return tokens


def _build_prefix_frequency(files: list[str], source: str, max_prefix_tokens: int = 6) -> dict[str, int]:
    prefix_counts: dict[str, int] = {}
    for fname in files:
        tokens = _filename_tokens(fname, source)
        limit = min(len(tokens), max_prefix_tokens)
        for idx in range(limit):
            last = tokens[idx].lower()
            if last in TITLE_STOPWORDS:
                continue
            prefix = " ".join(tokens[: idx + 1])
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    return prefix_counts


def _artist_from_filename_with_frequency(fname: str, source: str, prefix_counts: dict[str, int], min_count: int = 2) -> str:
    tokens = _filename_tokens(fname, source)
    limit = min(len(tokens), 6)

    for idx in range(limit, 0, -1):
        last = tokens[idx - 1].lower()
        if last in TITLE_STOPWORDS:
            continue
        prefix = " ".join(tokens[:idx])
        if prefix_counts.get(prefix, 0) >= min_count:
            return _normalize_artist_name(prefix)

    artist_parts = []
    for token in tokens:
        lowered = token.lower()
        if lowered in TITLE_STOPWORDS and lowered not in ARTIST_PARTICLES:
            break
        artist_parts.append(token)
        if len(artist_parts) >= 4 and lowered not in ARTIST_PARTICLES:
            break

    return _normalize_artist_name(" ".join(artist_parts)) if artist_parts else "Unknown"

def load_nga(rows):
    # scans image dir, pulls title/artist from objects.csv where available
    if not config.IMAGE_DIR.exists():
        print("  NGA image dir not found, skipping")
        return

    meta_map = {}
    try:
        import pandas as _pd
        obj = _pd.read_csv(config.RAW_DIR / "nga" / "objects.csv", low_memory=False,
                           usecols=["objectid", "title", "displaydate", "attribution"])
        for _, r in obj.iterrows():
            oid = str(int(r["objectid"])) if not _pd.isna(r["objectid"]) else None
            if oid:
                meta_map[oid] = {
                    "title": _clean(str(r.get("title", ""))),
                    "date": _clean(str(r.get("displaydate", ""))),
                    "artist": _clean(str(r.get("attribution", "NGA Portrait"))) or "NGA Portrait",
                }
    except Exception:
        pass

    files = sorted(config.IMAGE_DIR.glob("*.jpg"))
    for fp in files:
        stem = fp.stem.replace("painting_", "")
        meta = meta_map.get(stem, {})
        rows.append({
            "filename": fp.name,
            "image_dir": str(config.IMAGE_DIR),
            "artist": meta.get("artist", "NGA Portrait"),
            "title": meta.get("title", ""),
            "source": "NGA",
            "date": meta.get("date", ""),
        })
    print(f"  NGA: {len(files)} images")


def load_wikiart(rows):
    wikiart_dir = config.RAW_DIR / "wikiart"
    if not wikiart_dir.exists():
        print("  WikiArt dir not found, skipping")
        return

    artist_map = {}
    wikiart_csv = wikiart_dir / "wikiart_portraits.csv"
    if wikiart_csv.exists():
        df = pd.read_csv(wikiart_csv)
        for _, r in df.iterrows():
            fn = _clean(r.get("filename", ""))
            art = _clean(r.get("artist", ""))
            if fn and art:
                artist_map[fn] = art

    files = sorted(wikiart_dir.glob("*.jpg"))
    prefix_counts = _build_prefix_frequency([fp.name for fp in files], source="wikiart")
    added = 0
    for fp in files:
        fn = fp.name
        artist = artist_map.get(fn) or _artist_from_filename_with_frequency(fn, "wikiart", prefix_counts)
        rows.append({
            "filename": fn,
            "image_dir": str(wikiart_dir),
            "artist": _normalize_artist_name(artist),
            "title": "",
            "source": "WikiArt",
            "date": "",
        })
        added += 1
    print(f"  WikiArt: {added} images ({len(artist_map)} with CSV metadata)")


def load_met(rows):
    met_dir = config.RAW_DIR / "met"
    if not met_dir.exists():
        print("  Met dir not found, skipping")
        return

    artist_map = {}
    met_csv = met_dir / "met_portraits.csv"
    if met_csv.exists():
        df = pd.read_csv(met_csv)
        for _, r in df.iterrows():
            fn = _clean(r.get("filename", ""))
            art = _clean(r.get("artist", ""))
            if fn and art:
                artist_map[fn] = art

    files = sorted(met_dir.glob("*.jpg"))
    prefix_counts = _build_prefix_frequency([fp.name for fp in files], source="met")
    added = 0
    for fp in files:
        fn = fp.name
        artist = artist_map.get(fn) or _artist_from_filename_with_frequency(fn, "met", prefix_counts)
        rows.append({
            "filename": fn,
            "image_dir": str(met_dir),
            "artist": _normalize_artist_name(artist),
            "title": "",
            "source": "Met",
            "date": "",
        })
        added += 1
    print(f"  Met: {added} images")


def load_alternative(rows):
    alt_dir = config.RAW_DIR / "alternative"
    if not alt_dir.exists():
        return
    artist_map = {}
    alt_csv = alt_dir / "alternative_portraits.csv"
    if alt_csv.exists():
        df = pd.read_csv(alt_csv)
        for _, r in df.iterrows():
            fn = _clean(r.get("filename", ""))
            art = _clean(r.get("artist", ""))
            if fn and art:
                artist_map[fn] = _normalize_artist_name(art)
    files = sorted(alt_dir.glob("*.jpg")) + sorted(alt_dir.glob("*.png"))
    for fp in files:
        rows.append({
            "filename": fp.name,
            "image_dir": str(alt_dir),
            "artist": artist_map.get(fp.name, "Unknown"),
            "title": "",
            "source": "Alternative",
            "date": "",
        })
    print(f"  Alternative: {len(files)} images")

def main():
    print("Building unified metadata for all images...")
    rows = []

    load_nga(rows)
    load_wikiart(rows)
    load_met(rows)
    load_alternative(rows)

    df = pd.DataFrame(rows)

    df["artist"] = df["artist"].map(_normalize_artist_name)
    df["artist"] = df["artist"].str.strip().replace("", "Unknown")
    df.loc[df["artist"].str.lower().isin(UNKNOWN_ARTIST_VALUES), "artist"] = "Unknown"

    def exists(row):
        return (Path(row["image_dir"]) / row["filename"]).exists()

    before = len(df)
    df = df[df.apply(exists, axis=1)].reset_index(drop=True)
    print(f"\nVerified on disk: {len(df)} / {before} images exist")

    print(f"\nSource distribution:")
    print(df["source"].value_counts().to_string())
    print(f"\nArtist distribution (top 20):")
    print(df["artist"].value_counts().head(20).to_string())
    usable = df[df["artist"] != "Unknown"]["artist"].nunique()
    print(f"\nUnique labelled artists: {usable}")
    print(f"Images with artist label: {(df['artist'] != 'Unknown').sum()}")

    out = config.PROCESSED_DIR / "all_images_metadata.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    return df


if __name__ == "__main__":
    main()
