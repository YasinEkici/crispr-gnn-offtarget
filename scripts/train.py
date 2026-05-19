from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configured experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--debug", action="store_true", help="Run a lightweight smoke training path.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    experiment_name = config.get("experiment_name", Path(args.config).stem)
    max_epochs = args.max_epochs or config.get("training", {}).get("max_epochs", 1)

    print(f"Experiment: {experiment_name}")
    print(f"Debug mode: {args.debug}")
    print(f"Max epochs: {max_epochs}")
    print("Training placeholder: real ML models are deferred until later sprints.")

    if args.debug:
        run_dir = ROOT / "outputs" / "runs" / f"{experiment_name}_debug"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"status": "placeholder", "max_epochs": max_epochs}
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
