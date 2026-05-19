# Colab Runner Notes

Colab is only a runner for commands from this repository.

Typical setup:

```python
!pip install uv
!git clone https://github.com/<user>/crispr-gnn-offtarget.git
%cd crispr-gnn-offtarget
!uv sync
!uv run pytest -q
```

Keep reusable preprocessing, model, training, and evaluation code under `src/crispr_gnn/` and `scripts/`.
