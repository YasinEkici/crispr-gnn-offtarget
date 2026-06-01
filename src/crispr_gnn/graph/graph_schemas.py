"""Schema definitions for dependency-light Sprint 3 graph artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


GRAPH_A = "graph_a_minimal_physical_target"
GRAPH_B = "graph_b_guide_similarity_control"
GRAPH_C = "graph_c_context_observation"
GRAPH_ORDER = (GRAPH_A, GRAPH_B, GRAPH_C)

PHYSICAL_TARGET_KEY_FIELDS = (
    "genome",
    "target_chr",
    "target_start",
    "target_end",
    "target_strand",
)


@dataclass(frozen=True)
class SimilarityConfig:
    top_k: int = 5
    metric: str = "euclidean"


@dataclass(frozen=True)
class GraphBuildConfig:
    artifact_dir: Path = Path("data/processed/graphs/sprint3")
    report_path: Path = Path("outputs/sprint3/graph_schema_report.md")
    max_length: int = 23
    graph_b: SimilarityConfig = SimilarityConfig(metric="hamming")
    graph_c: SimilarityConfig = SimilarityConfig(metric="euclidean")
    visibility_policy: str = "strict_inductive_primary"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "GraphBuildConfig":
        paths = mapping.get("paths", {})
        representation = mapping.get("representation", {})
        graph_b = mapping.get("graph_b", {})
        graph_c = mapping.get("graph_c", {})
        return cls(
            artifact_dir=Path(str(paths.get("artifact_dir", cls.artifact_dir))),
            report_path=Path(str(paths.get("report_path", cls.report_path))),
            max_length=int(representation.get("max_length", cls.max_length)),
            graph_b=SimilarityConfig(
                top_k=int(graph_b.get("top_k", 5)),
                metric=str(graph_b.get("metric", "hamming")),
            ),
            graph_c=SimilarityConfig(
                top_k=int(graph_c.get("top_k", 5)),
                metric=str(graph_c.get("metric", "euclidean")),
            ),
            visibility_policy=str(mapping.get("visibility_policy", cls.visibility_policy)),
        )

