from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.data.splits import GuideSplitConfig, build_guide_split, write_split_artifacts  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build guide-level split artifacts.")
    parser.add_argument("--config", required=True, help="Path to data config YAML.")
    parser.add_argument("--output-dir", default="outputs/splits", help="Directory for split artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split search.")
    parser.add_argument("--search-iterations", type=int, default=50_000, help="Number of candidate guide splits to score.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    dataset = config.get("dataset", {})
    raw_path = ROOT / Path(str(dataset.get("raw_path", "")))
    if not raw_path.exists():
        print(f"Dataset not found: {raw_path}")
        return 1

    df = pd.read_parquet(raw_path)
    split_config = GuideSplitConfig(seed=args.seed, search_iterations=args.search_iterations)
    split, summary = build_guide_split(df, split_config)
    manifest_path, summary_path = write_split_artifacts(split, summary, ROOT / args.output_dir)

    print(f"Split manifest written: {manifest_path.relative_to(ROOT)}")
    print(f"Split summary written: {summary_path.relative_to(ROOT)}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
