# Dataset Audit

Sprint 0 skeleton. Sprint 1 will fill this file with verified results.

## Dataset

- Primary: Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset.
- Expected local raw path: `data/raw/260520_putative_nucleosomal.parquet`.
- Large raw files are not committed.

## Required audit outputs

- Row and column counts.
- Unique sgRNA and target counts.
- `measured=1` vs `measured=0` distribution.
- Missingness table for epigenetic features.
- Label distributions for Schemes A, B, and C.
- Outlier handling decisions for `cleavage_freq`.
- Dataset access methodology and citation notes.
