from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.graph import GRAPH_A, GRAPH_B, GRAPH_C  # noqa: E402
from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader  # noqa: E402


GRAPH_SCHEMAS = [GRAPH_A, GRAPH_B, GRAPH_C]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Sprint 3 graph artifacts and write a checksum/provenance record."
    )
    parser.add_argument(
        "--artifact-dir",
        default="data/processed/graphs/sprint3",
        help="Sprint 3 graph artifact directory.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path for the checksum/provenance record.",
    )
    parser.add_argument(
        "--approved-source",
        default="local_sprint3_handoff",
        help="Human-readable source label for the copied artifact set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_dir = (ROOT / Path(args.artifact_dir)).resolve()
    if not artifact_dir.exists():
        print(f"Sprint 3 graph artifact directory not found: {artifact_dir}", file=sys.stderr)
        return 1

    try:
        record = build_provenance_record(artifact_dir, approved_source=str(args.approved_source))
    except Exception as exc:  # pragma: no cover - exercised through CLI failure behavior
        print(f"Sprint 3 graph artifact validation failed: {exc}", file=sys.stderr)
        return 1

    if args.output:
        output_path = (ROOT / Path(args.output)).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Graph artifact provenance written: {output_path}")
    else:
        print(json.dumps(record, indent=2, sort_keys=True))

    print(
        "Validated Sprint 3 graph artifacts: "
        + ", ".join(f"{schema}={record['schemas'][schema]['candidate_pair_edges']} candidate edges" for schema in GRAPH_SCHEMAS)
    )
    return 0


def build_provenance_record(
    artifact_dir: Path,
    *,
    approved_source: str,
    loader_factory: Callable[[Path], Sprint3HeteroDataLoader] = Sprint3HeteroDataLoader,
) -> dict[str, Any]:
    """Validate Sprint 3 artifact contracts and compute file checksums.

    This function intentionally consumes serialized graph tables and manifests
    through the Sprint 4 loader. It does not rebuild topology from raw data and
    does not modify graph artifacts.
    """
    loader = loader_factory(artifact_dir)
    schemas: dict[str, dict[str, Any]] = {}
    for graph_schema in GRAPH_SCHEMAS:
        materialized = loader.load(graph_schema)
        manifest = _read_manifest(artifact_dir / graph_schema / "manifest.json")
        schemas[graph_schema] = {
            "graph_name": materialized.graph_name,
            "split_id": manifest.get("split_id"),
            "label_scheme": manifest.get("label_scheme"),
            "label_definition": manifest.get("label_definition"),
            "visibility_policy": manifest.get("metadata", {}).get("visibility_policy"),
            "nodes": manifest.get("nodes", {}),
            "relations": manifest.get("relations", {}),
            "candidate_pair_edges": int(manifest.get("relations", {}).get("candidate_pair", 0)),
        }

    return {
        "provenance_type": "sprint3_graph_artifacts_sha256",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "approved_source": approved_source,
        "artifact_dir": _safe_relative_path(artifact_dir),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
        "schemas": schemas,
        "files": _artifact_file_hashes(artifact_dir),
    }


def _artifact_file_hashes(artifact_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(artifact_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not rows:
        raise ValueError(f"No artifact files found under {artifact_dir}")
    return rows


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _safe_relative_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
