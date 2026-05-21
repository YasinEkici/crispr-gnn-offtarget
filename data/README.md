# Data

Large data files are not committed. Keep this folder in the structure defined by `PROJECT_FOLDER_STRUCTURE.md`.

## Folders

- `raw/`: original downloaded files. Do not commit.
- `interim/`: partially cleaned intermediate files. Do not commit unless tiny and explicitly reviewed.
- `processed/`: model-ready tables or graph objects. Do not commit large files.
- `splits/`: train/validation/test split manifests. Small split manifests may be committed if safe.
- `sample/`: tiny committed fixtures for tests and smoke commands only.

## Primary Dataset

Expected local raw path:

```text
data/raw/260520_putative_nucleosomal.parquet
```

Run the Sprint 1 audit in sample mode without the full dataset:

```bash
uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml --sample
```
