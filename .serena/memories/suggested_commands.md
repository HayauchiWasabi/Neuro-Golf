# Suggested Commands

- Build current task001 candidates:
  - `python3 onnx_work/task001.py`
  - Emits ONNX files into `outputs/gpt_workbench/task001/`.
- Use scoring/analysis helpers from repo root with `scripts` on `PYTHONPATH` when importing `neurogolf_score` or `neurogolf_onnx_analysis`:
  - `PYTHONPATH=scripts python3 -c "from neurogolf_score import score_model_file; print(score_model_file('solution/6411.33/task001.onnx', 'neurogolf-2026'))"`
  - `PYTHONPATH=scripts python3 -c "from neurogolf_score import score_submission_folder; import json; print(json.dumps(score_submission_folder('solution/6411.33', 'neurogolf-2026'), indent=2)[:4000])"`
- Validate a built candidate on public examples:
  - `PYTHONPATH=scripts python3 -c "from neurogolf_onnx_analysis import run_model_on_examples; import json; print(json.dumps(run_model_on_examples('outputs/gpt_workbench/task001/task001_ct_block_only.onnx', 'task001', 'neurogolf-2026'), indent=2)[:4000])"`
- Inspect model cost and status:
  - `PYTHONPATH=scripts python3 -c "from neurogolf_score import score_model_file; print(score_model_file('outputs/gpt_workbench/task001/task001_ct_block_only.onnx', 'neurogolf-2026'))"`
- Repository discovery on Darwin/zsh:
  - Use `rg --files` and `rg -n` first; avoid dumping generated ONNX/output trees unless needed.
  - `git status --short` is important because notebooks/outputs are often dirty/generated during experiments.