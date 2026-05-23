import pandas as pd
import pytest

from crispr_gnn.data.splits import (
    LABEL_COLUMN,
    GuideSplitConfig,
    assign_measured_splits,
    build_guide_split,
    main_clean_frame,
    validate_guide_split,
)


def make_split_df() -> pd.DataFrame:
    rows = []
    for guide_index in range(30):
        guide = f"g{guide_index:02d}"
        for row_index in range(4):
            rows.append(
                {
                    "grna_target_id": guide,
                    "measured": 1,
                    "experiment_id": 1,
                    "cleavage_freq": 2e-5 if row_index in {0, 1} else 0.0,
                    "genome": "hg19" if guide_index % 2 else "rn5",
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
            "grna_target_id": "g_exp18",
            "measured": 1,
            "experiment_id": 18,
            "cleavage_freq": 0.0,
            "genome": "hg19",
        }
    )
    rows.append(
        {
            "grna_target_id": "g_nan",
            "measured": 1,
            "experiment_id": 1,
            "cleavage_freq": None,
            "genome": "hg19",
        }
    )
    return pd.DataFrame(rows)


def test_main_clean_frame_excludes_nan_labels_and_experiment_18() -> None:
    clean = main_clean_frame(make_split_df(), GuideSplitConfig(search_iterations=100))

    assert clean[LABEL_COLUMN].notna().all()
    assert (clean["experiment_id"] == 18).sum() == 0
    assert "g_nan" not in set(clean["grna_target_id"])


def test_build_guide_split_has_disjoint_guides_and_measured_eval_rows() -> None:
    df = make_split_df()
    split, summary = build_guide_split(df, GuideSplitConfig(seed=7, search_iterations=500))
    assigned = assign_measured_splits(df, split)

    train = set(split.guides["train"])
    val = set(split.guides["val"])
    test = set(split.guides["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    assert set(assigned["split"]) == {"train", "val", "test"}
    assert (assigned["measured"] == 1).all()
    assert (assigned["experiment_id"] != 18).all()
    assert set(summary["split"]) == {"train", "val", "test"}
    assert (summary["positives"] > 0).all()
    assert (summary["negatives"] > 0).all()


def test_validate_guide_split_rejects_overlapping_guides() -> None:
    df = make_split_df()
    split, _ = build_guide_split(df, GuideSplitConfig(seed=7, search_iterations=500))
    bad_guides = {name: list(guides) for name, guides in split.guides.items()}
    bad_guides["test"].append(bad_guides["train"][0])
    bad_split = type(split)(config=split.config, guides=bad_guides, score=split.score)

    with pytest.raises(ValueError, match="overlapping"):
        validate_guide_split(df, bad_split)
