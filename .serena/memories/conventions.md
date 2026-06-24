# Conventions

- Prefer editing reusable Python helpers in `scripts/` or task builders in `onnx_work/`; notebooks are exploratory and may contain copied older logic.
- Task builders construct ONNX directly with `onnx.helper`/`numpy_helper`; small helper functions for tensor initializers (`t_i64`, `t_f16`, `vec_i64`, etc.) are normal local style.
- Builder outputs should land in generated locations such as `outputs/gpt_workbench/<task>/`; final candidate folders live under `solution/<score-or-candidate>/` with direct `taskXXX.onnx` names.
- ONNX graph naming matters to scoring: `sanitize_model` rewrites internal names to safe names but preserves `input` and `output`; avoid names containing `kernel_time`.
- Use static shapes everywhere. The local scorer rejects dynamic dims, graph attributes/subgraphs, sequences, multiple inputs/outputs, functions, and non-default domains.
- Cost optimization is `memory + params`; builders commonly choose FLOAT16 intermediate/output tensors and compact initializers/names to reduce cost and file size.
- Scoring statuses considered usable are `ok` and `ok_static`; `ok_static` means runtime profiling failed but static memory calculation succeeded, so preserve the error string for context.
- Do not treat `outputs/` and notebooks as authoritative source without checking freshness; they are frequently generated or modified during experiments.
- Existing docs may be Japanese/English mixed; code identifiers are English, while exploratory comments/docstrings can be Japanese in task-specific builders.