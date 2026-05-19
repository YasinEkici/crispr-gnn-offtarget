from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a run or configured experiment.")
    parser.add_argument("--config", help="Path to experiment config YAML.")
    parser.add_argument("--run", help="Path to a run directory.")
    parser.add_argument("--latest", action="store_true", help="Evaluate the latest run placeholder.")
    parser.add_argument("--debug", action="store_true", help="Run lightweight evaluation checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config:
        config = load_yaml(args.config)
        print(f"Evaluation config: {config.get('experiment_name', Path(args.config).stem)}")
    if args.run:
        print(f"Run path: {args.run}")
    if args.latest:
        print("Latest-run lookup is a later-sprint feature.")
    if args.debug:
        print("Debug evaluation placeholder completed.")
    if not any([args.config, args.run, args.latest, args.debug]):
        print("No evaluation target provided; placeholder completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
