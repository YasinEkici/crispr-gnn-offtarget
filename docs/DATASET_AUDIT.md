# Dataset Audit

This document is the canonical Sprint 1 dataset audit summary for the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset.

Generated report artifact:

- `outputs/reports/dataset_audit.md`

## Source And Access

- Primary dataset: Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset.
- Local raw path: `data/raw/260520_putative_nucleosomal.parquet`.
- Original URL: `https://crisprsql.com/downloads/260520_putative_nucleosomal.parquet.gz`.
- Project source method: Internet Archive Wayback snapshot of the original Mak 2022 URL.
- Citation: cite Mak et al. 2022, BMC Genomics, DOI `10.1186/s12864-022-09012-7`, CC-BY 4.0, plus the Wayback snapshot URL used to recover the dataset.

The original URL is unavailable, so this project treats the Wayback snapshot of the same original URL as the settled source. Sprint 1 does not reopen source discovery.

## Reference Check Summary

The audit independently loaded the local Parquet file and compared actual values against the expected reference snapshot. All reference-backed checks below passed.

| check | actual | expected | status |
| --- | ---: | ---: | --- |
| Rows | 310,142 | 310,142 | PASS |
| Columns | 45 | 45 | PASS |
| `measured=1` rows | 25,632 | 25,632 | PASS |
| `measured=0` rows | 284,510 | 284,510 | PASS |
| Unique sgRNAs using `grna_target_id` | 154 | 154 | PASS |
| Unique target locations using `target_chr`, `target_start`, `target_end`, `target_strand` | 138,747 | 138,747 | PASS |
| Genome names | `hg19`, `hg38`, `mm10`, `mm9`, `rn5` | same set | PASS |
| Cell lines excluding missing | 7 | 7 | PASS |
| Missing `cell_line` rows | 14,108 | approximately 14,108 | PASS |
| Rows missing at least one computed nucleosome feature | 15,153 | approximately 15,153 | PASS |
| CA-like transformed columns | none | none | PASS |

## Dataset Composition

Measured distribution:

| measured | rows | interpretation |
| ---: | ---: | --- |
| 1 | 25,632 | experimentally measured crisprSQL rows |
| 0 | 284,510 | putative off-target rows; optional training negatives only |

Genome counts:

| genome | rows |
| --- | ---: |
| hg19 | 244,602 |
| rn5 | 51,133 |
| hg38 | 7,252 |
| mm10 | 6,407 |
| mm9 | 748 |

Cell-line counts:

| cell_line | rows |
| --- | ---: |
| U2OS | 128,789 |
| embryo | 57,540 |
| HEK293 | 56,898 |
| HeLa | 31,592 |
| K562 | 18,859 |
| missing | 14,108 |
| HAP1 | 1,608 |
| N2A | 748 |

All missing `cell_line` rows are concentrated in `experiment_id=18`.

## Schema And Feature Groups

All expected feature groups are present:

- 6 experimental epigenetic scalar features:
  - `epigen_ctcf`
  - `epigen_dnase`
  - `epigen_rrbs`
  - `epigen_h3k4me3`
  - `epigen_drip`
  - `MNase`
- 13 computed nucleosome features stored as string-formatted 23-value arrays:
  - `GCContent`
  - `WSScore`
  - `YRScore`
  - `NucleotideBDM`
  - `StrongWeakBDM`
  - `NuPoP_Occup_147_human`
  - `NuPoP_Viterbi_147_human`
  - `NuPoP_Affinity_147_human`
  - `nuCpos_Occup_147_yeast`
  - `nuCpos_Viterbi_147_yeast`
  - `nuCpos_Affinity_147_yeast`
  - `VanDerHeijden`
  - `LeNupH3Q85C`
- 5 binding-energy scalar features:
  - `energy_1`
  - `energy_2`
  - `energy_3`
  - `energy_4`
  - `energy_5`

`cleavage_freq` is present and is the available raw label column. A transformed `CA` column is absent.

## Cleavage Frequency Quality

| check | actual | status |
| --- | ---: | --- |
| Minimum | -0.001531 | PASS |
| Maximum | 4.52863 | PASS |
| NaN values | 78 | PASS |
| Negative values | 685 | PASS |
| Zero values | 285,241 | PASS |
| Values above 1 | 298 | PASS |

Policy implications are documented in `docs/LABEL_SCHEMES.md` and `docs/DECISIONS.md`.

The 78 NaN rows are label-ineligible for supervised binary training/evaluation. Excluding them leaves 310,064 label-eligible rows overall and 25,554 label-eligible `measured=1` rows.

## Label Threshold Sensitivity

Main threshold results on the full dataset:

| scheme | definition | positives | negatives | imbalance | role |
| --- | --- | ---: | ---: | ---: | --- |
| Scheme A | `cleavage_freq > 1e-5` | 21,365 | 288,699 | 13.51:1 | primary binary label |
| Scheme C | `cleavage_freq > 1e-3` | 8,280 | 301,784 | 36.45:1 | later robustness sensitivity |
| High threshold | `cleavage_freq > 0.1` | 1,184 | 308,880 | 260.88:1 | audit-only sensitivity |

These threshold counts exclude NaN `cleavage_freq` rows. All positives under these thresholds come from `measured=1` rows in this snapshot. `measured=0` rows are putative negatives and must not be used as test ground truth.

## Guide-Level Split Risk

Unique target locations per sgRNA are highly uneven:

| statistic | value |
| --- | ---: |
| count | 154 |
| mean | 1,932.857 |
| median | 406 |
| 95th percentile | 9,725.650 |
| 99th percentile | 25,394.960 |
| max | 50,422 |

Later split code must account for large guides. Random edge split is debug-only; final evaluation requires guide-level splits.

Top guides by unique target locations:

| guide | unique target locations |
| --- | ---: |
| 1325 | 50,422 |
| 6478 | 26,915 |
| 541 | 24,047 |
| 10753 | 20,985 |
| 11107 | 18,435 |
| 788 | 13,486 |
| 11493 | 10,596 |
| 11506 | 10,596 |
| 11487 | 9,257 |
| 1018 | 8,381 |

## Modeling Caveats

- Main project track remains binary off-target classification with AUPRC as the primary metric.
- Mak CA / Box-Cox reproduction is a later-only paper-comparison track, not the center of the project.
- Test sets must contain only `measured=1` rows.
- Validation should prefer `measured=1` rows.
- `measured=0` rows may be used only as optional training negatives with an explicit label-noise caveat.
- `experiment_id=18` should be kept out of main evaluation or reported as a separate no-cell-line sensitivity subset.
- Non-`hg19` genomes should not be dropped by default; report per-genome breakdowns and avoid human-only overclaims.
- Computed nucleosome features have shared missingness in 15,153 rows, mostly concentrated in `experiment_id=18`.
