# Task Completion

- For ONNX builder changes, run the relevant builder, e.g. `python3 onnx_work/task001.py`, and confirm expected files are emitted under `outputs/gpt_workbench/<task>/`.
- Score each changed/generated model with `score_model_file` using `PYTHONPATH=scripts` and `data_dir='neurogolf-2026'`; require status `ok` or intentionally accepted `ok_static`.
- Run `run_model_on_examples` for the affected task/model to check `train`, `test`, and `arc-gen` public examples; investigate any `fail` or runtime `error` rows.
- If assembling or modifying a candidate folder/zip, use `score_submission_folder`, `select_best_by_task`/`write_submission_zip` as appropriate, then `validate_submission_zip`.
- Always check `git status --short` before final response; generated notebooks/outputs may already be dirty and should not be reverted unless the user explicitly asks.