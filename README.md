# Neuro Golf workspace

## Directory layout

- `neurogolf-2026/`: competition task data and official utilities.
- `scripts/`: ONNX generator scripts, one file per solved task.
- `outputs/`: current submission-ready ONNX files and submission zip.
- `notebooks/`: visualization and validation notebooks.
- `docs/`: Japanese overview notes.
- `profiles/`: notebook validation profiler traces.
- `artifacts/`: official verification outputs moved out of the project root.
  - `artifacts/official-onnx/`: ONNX files written by the official `verify_network`.
  - `artifacts/official-profiles/`: profiler JSON traces written by the official `verify_network`.
  - `artifacts/official-work/`: temporary working directory used to prevent root-level output.

## Common commands

Generate a task ONNX:

```bash
python3 scripts/create_task009.py
```

Task scripts use `scripts/neurogolf_common.py` for output paths, local verification, and official verification artifact routing. Current generated ONNX files are written to `outputs/`.

Validate generated ONNX files:

```bash
jupyter notebook notebooks/neurogolf_onnx_validation.ipynb
```
