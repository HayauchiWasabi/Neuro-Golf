# Core

- Kaggle NeuroGolf 2026 workspace for building tiny ONNX models for ARC-style `task001`..`task400` transformations.
- Source map:
  - `neurogolf-2026/taskXXX.json`: competition task data; each JSON has `train`, `test`, and `arc-gen` example pairs.
  - `neurogolf-2026/neurogolf_utils/neurogolf_utils.py`: downloaded competition utility; keep as reference/validator-style code, not primary app source.
  - `scripts/neurogolf_score.py`: reusable local scoring helpers for ONNX files/folders/candidate selection/zip validation.
  - `scripts/neurogolf_onnx_analysis.py`: task/model summarization, GPT prompt generation, and public-example execution helpers.
  - `onnx_work/`: task-specific ONNX builders/experiments; builders usually emit candidate `.onnx` files under `outputs/gpt_workbench/<task>/`.
  - `notebooks/`: exploratory workflows; scripts are cleaner reusable versions of notebook scoring/analysis logic.
  - `solution/<candidate>/`: candidate submission folders containing direct `taskXXX.onnx` files.
  - `outputs/`: generated score CSVs, analysis prompts, and workbench ONNX artifacts; often dirty/generated.
- Competition invariants:
  - Model input name `input`, shape `[1, 10, 30, 30]`, float32 one-hot color channels 0..9; outside the grid is zero-hot.
  - Model output name `output`, shape `[1, 10, 30, 30]`; local builders often use FLOAT16 output to reduce memory.
  - ONNX must have static tensor shapes, one input, one output, default/ai.onnx opsets only, no functions/subgraphs.
  - Banned ops include `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, `Compress`; sequence ops are treated as invalid locally.
  - Per-file ONNX size limit is 1.44 MiB.
- Read `mem:tech_stack` for dependencies/runtime assumptions, `mem:suggested_commands` for practical commands, `mem:conventions` for builder/scoring patterns, and `mem:task_completion` for done checks.