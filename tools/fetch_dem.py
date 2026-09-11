"""Download a one-time SRTM HGT tile for the current Dehradun demo area."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default="N30E078")
    parser.add_argument("--output", type=Path, default=Path("data/dem"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"{args.tile}.hgt"
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{args.tile[:3]}/{args.tile}.hgt.gz"
    gz = target.with_suffix(".hgt.gz")
    print(f"Downloading {url}")
    urlretrieve(url, gz)
    import gzip
    with gzip.open(gz, "rb") as source, target.open("wb") as destination:
        destination.write(source.read())
    gz.unlink()
    print(target)


if __name__ == "__main__":
    main()
