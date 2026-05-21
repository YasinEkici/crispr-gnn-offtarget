from __future__ import annotations

from pathlib import Path


EXPECTED_RAW_PATH = Path("data/raw/260520_putative_nucleosomal.parquet")
EXPECTED_SHAPE = (310_142, 45)
EXPECTED_MEASURED_COUNTS = {1: 25_632, 0: 284_510}
EXPECTED_UNIQUE_GUIDES = 154
EXPECTED_UNIQUE_TARGETS = 138_747
EXPECTED_GENOMES = {
    "hg19": 244_000,
    "rn5": 51_000,
    "hg38": 7_000,
}
EXPECTED_GENOME_NAMES = {"hg19", "rn5", "hg38", "mm10", "mm9"}
EXPECTED_CELL_LINES = 7
EXPECTED_MISSING_CELL_LINE = 14_108
EXPECTED_COMPUTED_FEATURE_MISSING_ROWS = 15_153
EXPECTED_CLEAVAGE_FREQ = {
    "min": -0.0015,
    "max": 4.53,
    "nan": 78,
    "negative": 685,
}
EXPECTED_THRESHOLDS = {
    "scheme_a": {"threshold": 1e-5, "positives": 21_365, "imbalance": 14.0},
    "scheme_c": {"threshold": 1e-3, "positives": 8_280, "imbalance": 36.0},
    "high_0.1": {"threshold": 0.1, "positives": 1_184, "imbalance": 261.0},
}

EXPERIMENTAL_EPIGENETIC_FEATURES = [
    "epigen_ctcf",
    "epigen_dnase",
    "epigen_rrbs",
    "epigen_h3k4me3",
    "epigen_drip",
    "MNase",
]

COMPUTED_NUCLEOSOME_FEATURES = [
    "GCContent",
    "WSScore",
    "YRScore",
    "NucleotideBDM",
    "StrongWeakBDM",
    "NuPoP_Occup_147_human",
    "NuPoP_Viterbi_147_human",
    "NuPoP_Affinity_147_human",
    "nuCpos_Occup_147_yeast",
    "nuCpos_Viterbi_147_yeast",
    "nuCpos_Affinity_147_yeast",
    "VanDerHeijden",
    "LeNupH3Q85C",
]

BINDING_ENERGY_FEATURES = [
    "energy_1",
    "energy_2",
    "energy_3",
    "energy_4",
    "energy_5",
]

REQUIRED_FIELDS = [
    "measured",
    "experiment_id",
    "cell_line",
    "cleavage_freq",
]

GENOME_CANDIDATE_FIELDS = [
    "genome",
    "assembly",
    "target_genome",
    "genome_assembly",
]

TARGET_KEY_FIELDS = [
    "target_chr",
    "target_start",
    "target_end",
    "target_strand",
]

GUIDE_KEY_CANDIDATE_FIELDS = [
    "grna_target_id",
    "grna_target_chr",
    "grna_target_start",
    "grna_target_end",
    "grna",
    "sgRNA",
    "sgRNA_seq",
    "grna_seq",
]

GUIDE_KEY = "grna_target_id"
