# Tech Stack

- Language: Python 3.11.9 observed locally on Darwin.
- No `requirements.txt`, `pyproject.toml`, `environment.yml`, `setup.py`, or `Makefile` currently present.
- Required runtime libraries inferred from imports:
  - `numpy`
  - `onnx`
  - `onnxruntime`
  - `onnx_tool` for official Kaggle utility/notebook flows
  - `matplotlib`, `IPython` for visualization/notebook utility flows
- Local scripts use standard library modules only beyond the ONNX/Numpy stack: `json`, `math`, `pathlib`, `tempfile`, `zipfile`, `collections.Counter`.
- ONNX model builders use `onnx.helper`, `onnx.TensorProto`, `onnx.numpy_helper`; current task builder sets `ir_version = 8` and opset 9 for task001 candidate.
- Official utility defaults in `neurogolf-2026/neurogolf_utils/neurogolf_utils.py`: grid shape `[1, 10, 30, 30]`, data type float, official `_IR_VERSION = 10`, default opset 10.
- Local scorer disables ONNX Runtime graph optimization while scoring/profiling: `ORT_DISABLE_ALL`; do not assume optimized ORT graph behavior when comparing costs.
- Local imports from `scripts/` are plain module imports, not packaged; one-off commands often need `PYTHONPATH=scripts` when importing `neurogolf_score` or `neurogolf_onnx_analysis` from repo root.