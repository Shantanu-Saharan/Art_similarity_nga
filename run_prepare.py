import argparse
import sys
from pathlib import Path

# repo root on path so src.* imports work when running directly
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.data.nga_loader import prepare_portrait_subset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare NGA portrait-painting subset")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of portrait paintings to download"
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Force image download again even if file already exists"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = prepare_portrait_subset(limit=args.limit, redownload=args.redownload)

    downloaded = int(df["downloaded"].sum())
    print()
    print("Preparation stage finished.")
    print(f"Rows in subset CSV: {len(df)}")
    print(f"Downloaded images : {downloaded}")


if __name__ == "__main__":
    main()
