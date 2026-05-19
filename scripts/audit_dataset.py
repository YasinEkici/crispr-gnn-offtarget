from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the configured dataset.")
    parser.add_argument("--config", required=True, help="Path to data config YAML.")
    parser.add_argument("--sample", action="store_true", help="Run a lightweight sample audit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    dataset = config.get("dataset", {})
    raw_path = ROOT / dataset.get("raw_path", "")

    print(f"Dataset: {dataset.get('name', 'unknown')}")
    print(f"Raw path: {raw_path}")
    if args.sample:
        print("Sample audit mode: no full dataset required.")
    if not raw_path.exists():
        print("Raw dataset not found; this is expected for Sprint 0 scaffold smoke runs.")
    print("Audit placeholder completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
