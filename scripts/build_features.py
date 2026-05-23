from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.data.splits import assign_measured_splits  # noqa: E402
from crispr_gnn.data.splits import GuideSplit, GuideSplitConfig  # noqa: E402
from crispr_gnn.features.tabular import summarize_feature_sets, write_feature_catalog  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sprint 2 feature catalog artifacts.")
    parser.add_argument("--config", required=True, help="Path to data config YAML.")
    parser.add_argument("--split-manifest", default="outputs/splits/sprint2_guides.json", help="Locked split manifest.")
    parser.add_argument("--output-dir", default="outputs/features", help="Directory for feature artifacts.")
    return parser.parse_args()


def load_split(path: Path) -> GuideSplit:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config_data = manifest["config"]
    config = GuideSplitConfig(
        split_id=config_data["split_id"],
        seed=int(config_data["seed"]),
        guide_column=config_data["guide_column"],
        label_threshold=float(config_data["label_threshold"]),
        train_fraction=float(config_data["train_fraction"]),
        val_fraction=float(config_data["val_fraction"]),
        test_fraction=float(config_data["test_fraction"]),
        exclude_experiment_id=config_data["exclude_experiment_id"],
        search_iterations=int(config_data["search_iterations"]),
    )
    return GuideSplit(config=config, guides=manifest["guides"], score=float(manifest["score"]))


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    dataset = config.get("dataset", {})
    raw_path = ROOT / Path(str(dataset.get("raw_path", "")))
    split_path = ROOT / args.split_manifest
    if not raw_path.exists():
        print(f"Dataset not found: {raw_path}")
        return 1
    if not split_path.exists():
        print(f"Split manifest not found: {split_path}")
        return 1

    df = pd.read_parquet(raw_path)
    split = load_split(split_path)
    assigned = assign_measured_splits(df, split)
    catalog_path, summary_path = write_feature_catalog(assigned, ROOT / args.output_dir)

    print(f"Feature catalog written: {catalog_path.relative_to(ROOT)}")
    print(f"Feature summary written: {summary_path.relative_to(ROOT)}")
    print(summarize_feature_sets(assigned).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
