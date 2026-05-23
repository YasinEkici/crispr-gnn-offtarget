import pandas as pd

from crispr_gnn.data.splits import GuideSplitConfig, assign_measured_splits, build_guide_split


def make_leakage_df() -> pd.DataFrame:
    rows = []
    for guide_index in range(24):
        guide = f"guide-{guide_index:02d}"
        for row_index in range(6):
            rows.append(
                {
                    "grna_target_id": guide,
                    "measured": 1,
                    "experiment_id": 1,
                    "cleavage_freq": 2e-5 if row_index < 3 else 0.0,
                    "genome": "hg19",
                }
            )
        rows.append(
            {
                "grna_target_id": guide,
                "measured": 0,
                "experiment_id": 1,
                "cleavage_freq": 0.0,
                "genome": "hg19",
            }
        )
    rows.append(
        {
            "grna_target_id": "experiment-18-guide",
            "measured": 1,
            "experiment_id": 18,
            "cleavage_freq": 2e-5,
            "genome": "hg19",
        }
    )
    return pd.DataFrame(rows)


def test_guide_level_split_has_no_guide_or_putative_row_leakage() -> None:
    df = make_leakage_df()
    split, _ = build_guide_split(df, GuideSplitConfig(seed=11, search_iterations=500))
    assigned = assign_measured_splits(df, split)

    guide_sets = {name: set(guides) for name, guides in split.guides.items()}
    assert guide_sets["train"].isdisjoint(guide_sets["val"])
    assert guide_sets["train"].isdisjoint(guide_sets["test"])
    assert guide_sets["val"].isdisjoint(guide_sets["test"])

    assert (assigned["measured"] == 1).all()
    assert (assigned["experiment_id"] != 18).all()
    assert "experiment-18-guide" not in guide_sets["train"] | guide_sets["val"] | guide_sets["test"]
