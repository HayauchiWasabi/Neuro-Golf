# Task Completion

- For Python source edits, run at minimum:
  - `python3 -m py_compile scripts/neurogolf_score.py scripts/neurogolf_onnx_analysis.py onnx_work/build_task001.py`
- For edits to a task builder, run the builder, then validate the produced ONNX:
  - `python3 onnx_work/build_task001.py`
  - `PYTHONPATH=scripts python3 -c "from neurogolf_onnx_analysis import run_model_on_examples; print(run_model_on_examples('outputs/gpt_workbench/task001/task001_candidate.onnx', 'task001', 'neurogolf-2026'))"`
  - `PYTHONPATH=scripts python3 -c "from neurogolf_score import score_model_file; print(score_model_file('outputs/gpt_workbench/task001/task001_candidate.onnx', 'neurogolf-2026'))"`
- For submission-folder or zip-selection changes, score/validate with `scripts/neurogolf_score.py` helpers: `score_submission_folder`, `select_best_by_task`, `write_submission_zip`, `validate_submission_zip` as applicable.
- Completion criterion for ONNX optimization work: report example validation status, score/cost/memory/params, generated file path, and whether a candidate beats the baseline being compared.
- If dependencies are missing locally (`onnx`, `onnxruntime`, etc.), state that verification could not run and include the exact failed command/error.