# AGENTS.md

This repository implements a context-aware GNN workflow for CRISPR-Cas9 off-target prediction.

## Read first

1. `docs/PROJECT_CONTEXT.md`
2. `CRISPR_GNN_PROJECT_PLAN.md`
3. `PROJECT_FOLDER_STRUCTURE.md`
4. `docs/DATASET_AUDIT.md`
5. `docs/LABEL_SCHEMES.md`
6. `docs/EVALUATION_PROTOCOL.md`
7. `docs/literature/literature_index.md`

For feature parsing or epigenetic/nucleosome feature work, also read:

- `docs/FEATURE_PARSING.md`

For implementation decisions and previous rationale, also check:

- `docs/DECISIONS.md`

## Ground rules

- Use `uv sync` for setup and `uv run ...` for commands.
- Do not manually maintain `requirements.txt`.
- Keep core logic under `src/crispr_gnn/`.
- Colab is a runner only; notebooks must not contain final model logic.
- Do not commit large data, checkpoints, or run artifacts.
- Do not add PyTorch or PyTorch Geometric until the graph-model sprint.
- Final evaluation must use guide-level splits and AUPRC as the primary metric.
- Test rows must be `measured=1` only.
- Update `docs/DECISIONS.md` when changing labels, splits, datasets, or evaluation rules.
- Implementation choices must follow the current project plan, evaluation protocol, and documented best practices.
- When implementing a model, feature, loss, graph schema, or evaluation rule, consult the task-relevant literature notes under `docs/literature/`.
- Do not copy or reproduce a paper’s method blindly. If a design is inspired by a paper, document the adaptation clearly in code comments, configs, or relevant docs.
- Do not claim that an implementation reproduces a paper unless the dataset, split, target, metrics, and architecture match the paper’s setup.

## Task workflow

For non-trivial tasks:

1. Read the relevant docs listed above.
2. Read the active execution plan under `docs/exec-plans/active/`.
3. Before editing, summarize:
   - task goal,
   - expected file changes,
   - risks,
   - acceptance criteria.
4. Wait for user approval if the task is broad or changes core behavior.
5. Keep changes limited to the current task.
6. Run the required `uv run ...` commands and tests.
7. Update relevant docs when behavior, labels, paths, or evaluation rules change.