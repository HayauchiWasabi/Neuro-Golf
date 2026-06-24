# Tech Stack

- Python workspace; no `pyproject.toml`, `requirements.txt`, or Makefile is present, so dependencies are implicit.
- Core runtime packages used by scripts/builders:
  - `numpy`
  - `onnx`
  - `onnxruntime`
  - `onnx.numpy_helper` / `onnx.helper` / `TensorProto`
- Competition utility additionally imports:
  - `onnx_tool`
  - `matplotlib`
  - `IPython.display`
- Standard-library usage is lightweight: `pathlib`, `json`, `math`, `collections.Counter`, `tempfile`, `zipfile`.
- Notebooks use Python 3 / IPython kernels; reusable logic should usually move into `scripts/` or `onnx_work/` instead of staying notebook-only.
- Local scorer uses `onnxruntime.SessionOptions` with `ORT_DISABLE_ALL` graph optimizations when profiling or checking public examples, matching the intent to measure authored graphs rather than optimized runtime rewrites.