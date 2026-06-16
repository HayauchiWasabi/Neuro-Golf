# Conventions

- Python style in repo: `from __future__ import annotations` in reusable scripts, `Path` for paths, type hints on helpers, small pure functions, minimal comments except for ONNX graph intent.
- Reusable helper modules are function-only scripts with no CLI parser; preserve importability and avoid notebook-only dependencies in `scripts/neurogolf_score.py` and `scripts/neurogolf_onnx_analysis.py`.
- ONNX builders are task-specific executable scripts under `onnx_work/`; they construct graphs explicitly with `onnx.helper.make_node` and write generated models under `outputs/gpt_workbench/<task>/`.
- ONNX graph invariants for generated candidates:
  - Single input named `input`, single output named `output`.
  - Static tensor shapes only.
  - Target input shape `[1, 10, 30, 30]`.
  - Avoid banned ops: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, `Compress`; also avoid Sequence operations.
  - Avoid custom domains, subgraphs, model functions, initializer names colliding with graph input/output names.
  - Keep each ONNX file under 1.44 MiB.
- Scoring-sensitive edits: cost is `memory + params`; removing initializers, intermediate tensors, value_info footprint, casts, pads, and redundant nodes can matter even if runtime outputs are unchanged.
- Correctness-sensitive edits: validate against `train`, `test`, and `arc-gen` examples before keeping a candidate; example outputs must match one-hot tensors exactly after thresholding in local analysis helper.
- Generated artifacts in `outputs/`, `profiles/`, and `solution/` may be large/binary; avoid broad rewrites or cleanup unless the user asks.