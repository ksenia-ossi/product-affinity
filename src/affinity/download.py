"""Download the raw Instacart CSVs into data/raw (skips files that already exist)."""
import urllib.request

from .config import MIRROR, RAW_DIR, RAW_FILES


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        dest = RAW_DIR / name
        if dest.exists():
            print(f"skip {name} (exists)")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(MIRROR + name, dest)
    print("done")


if __name__ == "__main__":
    main()
