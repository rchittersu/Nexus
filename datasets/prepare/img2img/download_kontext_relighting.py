"""
Download kontext-community/relighting and write images + pairs.json for img2img prepare.py.
Requires: pip install datasets

  python download_kontext_relighting.py --out_dir ./img2img/kontext_relighting
  python prepare.py --pairs_json <out_dir>/pairs.json --local_mds_dir <out_dir>/mds/ ...
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./img2img/kontext_relighting")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset("kontext-community/relighting", split="train")
    out_dir = Path(args.out_dir)
    (out_dir / "sources").mkdir(parents=True, exist_ok=True)
    (out_dir / "targets").mkdir(parents=True, exist_ok=True)

    row0 = dataset[0]
    image_cols = [k for k, v in row0.items() if hasattr(v, "save")]
    if len(image_cols) < 2:
        raise ValueError(f"Dataset has only one image column: {image_cols}. Need source + target.")
    source_col, target_col = image_cols[0], image_cols[1]
    caption_col = next((k for k, v in row0.items() if isinstance(v, str)), None)

    pairs = []
    for i in range(len(dataset)):
        row = dataset[i]
        src = row[source_col]
        tgt = row[target_col]
        cap = row.get(caption_col, "") or ""
        src_path = str(out_dir / "sources" / f"{i:04d}.png")
        tgt_path = str(out_dir / "targets" / f"{i:04d}.png")
        src.save(src_path)
        tgt.save(tgt_path)
        pairs.append({"source_path": src_path, "target_path": tgt_path, "caption": cap})

    with open(out_dir / "pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Wrote {len(pairs)} pairs to {out_dir}, pairs.json ready for prepare.py")


if __name__ == "__main__":
    main()
