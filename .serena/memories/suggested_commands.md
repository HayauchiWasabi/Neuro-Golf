# Suggested Commands

- Build current task001 candidate:
  - `python3 onnx_work/build_task001.py`
- Quick syntax/import check for local helper modules:
  - `python3 -m py_compile scripts/neurogolf_score.py scripts/neurogolf_onnx_analysis.py onnx_work/build_task001.py`
- Score a single ONNX file from repo root:
  - `PYTHONPATH=scripts python3 -c "from neurogolf_score import score_model_file; print(score_model_file('outputs/gpt_workbench/task001/task001_candidate.onnx', 'neurogolf-2026'))"`
- Validate task examples for a candidate ONNX:
  - `PYTHONPATH=scripts python3 -c "from neurogolf_onnx_analysis import run_model_on_examples; print(run_model_on_examples('outputs/gpt_workbench/task001/task001_candidate.onnx', 'task001', 'neurogolf-2026'))"`
- Score a candidate folder:
  - `PYTHONPATH=scripts python3 -c "from neurogolf_score import score_submission_folder; print(score_submission_folder('solution/6406.18', 'neurogolf-2026'))"`
- Inspect model summary for GPT/debugging:
  - `PYTHONPATH=scripts python3 -c "from neurogolf_onnx_analysis import summarize_model; import pprint; pprint.pp(summarize_model('outputs/gpt_workbench/task001/task001_candidate.onnx'))"`
- Build/score/zip workflows also exist as notebooks under `notebooks/`: `analyze_onnx_for_gpt.ipynb`, `score_submission_folder.ipynb`, `build_optimized_submission_zip.ipynb`, `compare_task_script_outputs.ipynb`.
- Darwin note: filenames may include `.DS_Store` in generated/output directories; scoring helpers glob only `task*.onnx` where relevant.