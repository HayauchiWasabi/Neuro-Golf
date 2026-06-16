from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import tempfile
import zipfile

import numpy as np
import onnx
import onnxruntime


FILESIZE_LIMIT_IN_BYTES = 1.44 * 1024 * 1024
EXCLUDED_OP_TYPES = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
SCOREABLE_STATUSES = {"ok", "ok_static"}


def fallback_input() -> np.ndarray:
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    x[0, 0, :, :] = 1.0
    return x


def grid_to_input_tensor(grid: list[list[int]]) -> np.ndarray:
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            x[0, color, r, c] = 1.0
    return x


def profiling_input_for_task(task_name: str, data_dir: Path) -> np.ndarray:
    task_path = Path(data_dir) / f"{task_name}.json"
    if not task_path.exists():
        return fallback_input()
    with open(task_path) as f:
        task = json.load(f)
    return grid_to_input_tensor(task["train"][0]["input"])


def calculate_params(model: onnx.ModelProto) -> int | None:
    params = 0
    for init in model.graph.initializer:
        if any(d <= 0 for d in init.dims):
            return None
        params += math.prod(init.dims)
    for sparse_init in model.graph.sparse_initializer:
        if any(d <= 0 for d in sparse_init.values.dims):
            return None
        params += math.prod(sparse_init.values.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(d <= 0 for d in attr.t.dims):
                    return None
                params += math.prod(attr.t.dims)
            elif attr.name == "sparse_value":
                if any(d <= 0 for d in attr.sparse_tensor.values.dims):
                    return None
                params += math.prod(attr.sparse_tensor.values.dims)
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return params


def sanitize_model(model: onnx.ModelProto) -> onnx.ModelProto | None:
    for node in model.graph.node:
        if node.output:
            node.name = node.output[0]
            if "kernel_time" in node.output[0]:
                return None

    name_map: dict[str, str] = {}
    counter = 0

    def get_safe_name(old_name: str) -> str:
        nonlocal counter
        if not old_name or old_name in {"input", "output"}:
            return old_name
        if old_name not in name_map:
            name_map[old_name] = f"safe_name_{counter}"
            counter += 1
        return name_map[old_name]

    for inp in model.graph.input:
        inp.name = get_safe_name(inp.name)
    for init in model.graph.initializer:
        init.name = get_safe_name(init.name)
    for node in model.graph.node:
        for i in range(len(node.input)):
            node.input[i] = get_safe_name(node.input[i])
        for i in range(len(node.output)):
            node.output[i] = get_safe_name(node.output[i])
        if node.output and node.output[0]:
            node.name = node.output[0]
    for out in model.graph.output:
        out.name = get_safe_name(out.name)
    for vi in model.graph.value_info:
        vi.name = get_safe_name(vi.name)
    for node in model.graph.node:
        if node.output:
            node.name = node.output[0]
    return model


def _validate_static_graph(model: onnx.ModelProto) -> onnx.GraphProto | None:
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None

    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names):
        return None
    if model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None

    seen = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)
    return graph


def _tensor_memory_maps(graph: onnx.GraphProto) -> tuple[dict[str, int], dict[str, np.dtype]] | tuple[None, None]:
    tensor_names = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return None, None
        for output_name in node.output:
            if output_name:
                tensor_names.add(output_name)

    tensor_memory = {}
    tensor_dtypes = {}
    tensor_map = {t.name: t for t in list(graph.input) + list(graph.value_info) + list(graph.output)}
    tensor_names.update(tensor_map.keys())
    for tensor_name in tensor_names:
        item = tensor_map.get(tensor_name)
        if not item:
            return None, None
        if item.type.HasField("sequence_type"):
            return None, None
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None, None
        num_elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                return None, None
            num_elements *= dim.dim_value
        if tensor_name in ["input", "output"]:
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_memory[tensor_name] = num_elements * np.dtype(np_dtype).itemsize
        tensor_dtypes[tensor_name] = np_dtype

    for node in graph.node:
        for output_name in node.output:
            if output_name and output_name != "output":
                item = tensor_map.get(output_name)
                if item is None or not item.type.HasField("tensor_type"):
                    return None, None
    return tensor_memory, tensor_dtypes


def calculate_static_memory(model: onnx.ModelProto) -> int | None:
    graph = _validate_static_graph(model)
    if graph is None:
        return None
    tensor_memory, _ = _tensor_memory_maps(graph)
    if tensor_memory is None:
        return None
    return sum(tensor_memory.values())


def calculate_memory(model: onnx.ModelProto, trace_path: str | Path) -> int | None:
    graph = _validate_static_graph(model)
    if graph is None:
        return None

    node_outputs = {node.name: list(node.output) for node in graph.node}
    tensor_memory, tensor_dtypes = _tensor_memory_maps(graph)
    if tensor_memory is None or tensor_dtypes is None:
        return None

    with open(trace_path) as f:
        trace_data = json.load(f)
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = event.get("name").replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for i, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if i >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][i]
            if output_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory[output_name], mem)
    return sum(tensor_memory.values())


def score_from_cost(memory: int, params: int) -> float:
    cost = max(1.0, float(memory + params))
    return max(1.0, 25.0 - math.log(cost))


def _finalize_score_result(
    result: dict,
    memory: int | None,
    params: int | None,
    status: str,
    error: str = "",
) -> dict:
    if memory is None or params is None or memory < 0 or params < 0:
        result.update(status="error", error="cost could not be measured")
        return result
    cost = max(1.0, float(memory + params))
    result.update(
        status=status,
        memory=memory,
        params=params,
        cost=cost,
        score=score_from_cost(memory, params),
        error=error,
    )
    return result


def score_model_file(path: str | Path, data_dir: str | Path) -> dict:
    path = Path(path)
    result = {
        "task": path.stem,
        "candidate": path.parent.name,
        "file": str(path),
        "filesize": path.stat().st_size,
    }
    if result["filesize"] > FILESIZE_LIMIT_IN_BYTES:
        result.update(status="error", error="filesize limit exceeded")
        return result

    try:
        model = sanitize_model(onnx.load(path))
        if model is None:
            result.update(status="error", error="model sanitization failed")
            return result
        for node in model.graph.node:
            op_type = node.op_type.upper()
            if op_type in EXCLUDED_OP_TYPES or "Sequence" in node.op_type:
                result.update(status="error", error=f"excluded op: {node.op_type}")
                return result

        with tempfile.TemporaryDirectory() as tmpdir:
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.profile_file_prefix = str(Path(tmpdir) / path.stem)
            session = onnxruntime.InferenceSession(model.SerializeToString(), options)
            session.run(["output"], {"input": profiling_input_for_task(path.stem, Path(data_dir))})
            trace_path = session.end_profiling()

            memory = calculate_memory(model, trace_path)
            params = calculate_params(model)
        return _finalize_score_result(result, memory, params, "ok")
    except Exception as exc:
        try:
            model = sanitize_model(onnx.load(path))
            if model is None:
                result.update(status="error", error="model sanitization failed")
                return result
            memory = calculate_static_memory(model)
            params = calculate_params(model)
            return _finalize_score_result(result, memory, params, "ok_static", repr(exc))
        except Exception as static_exc:
            result.update(status="error", error=f"runtime={repr(exc)}; static={repr(static_exc)}")
            return result


def is_scoreable(row: dict) -> bool:
    return row.get("status") in SCOREABLE_STATUSES


def score_submission_folder(folder: str | Path, data_dir: str | Path) -> dict:
    folder = Path(folder)
    rows = [score_model_file(path, data_dir) for path in sorted(folder.glob("task*.onnx"))]
    ok_rows = [row for row in rows if is_scoreable(row)]
    return {
        "folder": str(folder),
        "candidate": folder.name,
        "num_files": len(rows),
        "num_ok": len(ok_rows),
        "num_errors": len(rows) - len(ok_rows),
        "total_score": sum(row["score"] for row in ok_rows),
        "rows": rows,
    }


def discover_candidate_dirs(solution_root: str | Path, candidate_names: list[str] | None = None) -> list[Path]:
    solution_root = Path(solution_root)
    if candidate_names is not None:
        return [solution_root / name for name in candidate_names]
    return sorted(path for path in solution_root.iterdir() if path.is_dir())


def score_candidate_dirs(candidate_dirs: list[str | Path], data_dir: str | Path) -> tuple[list[dict], list[dict]]:
    summaries = []
    rows = []
    for folder in candidate_dirs:
        summary = score_submission_folder(folder, data_dir)
        summaries.append({k: v for k, v in summary.items() if k != "rows"})
        rows.extend(summary["rows"])
    return summaries, rows


def select_best_by_task(rows: list[dict]) -> list[dict]:
    best_by_task = {}
    for row in rows:
        if not is_scoreable(row):
            continue
        current = best_by_task.get(row["task"])
        is_better = current is None
        if current is not None:
            is_better = (
                row["score"] > current["score"]
                or (row["score"] == current["score"] and row["cost"] < current["cost"])
                or (
                    row["score"] == current["score"]
                    and row["cost"] == current["cost"]
                    and row["filesize"] < current["filesize"]
                )
                or (
                    row["score"] == current["score"]
                    and row["cost"] == current["cost"]
                    and row["filesize"] == current["filesize"]
                    and row["candidate"] < current["candidate"]
                )
            )
        if is_better:
            best_by_task[row["task"]] = row
    return [best_by_task[task] for task in sorted(best_by_task)]


def optimized_summary(selected_rows: list[dict], all_rows: list[dict]) -> dict:
    return {
        "selected_tasks": len(selected_rows),
        "total_score": sum(row["score"] for row in selected_rows),
        "selected_by_candidate": dict(Counter(row["candidate"] for row in selected_rows)),
        "selected_by_status": dict(Counter(row["status"] for row in selected_rows)),
        "error_rows": [row for row in all_rows if not is_scoreable(row)],
    }


def write_submission_zip(selected_rows: list[dict], output_zip: str | Path) -> Path:
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in sorted(selected_rows, key=lambda r: r["task"]):
            zf.write(row["file"], arcname=f"{row['task']}.onnx")
    return output_zip


def validate_submission_zip(output_zip: str | Path) -> dict:
    output_zip = Path(output_zip)
    with zipfile.ZipFile(output_zip) as zf:
        infos = zf.infolist()
    names = [info.filename for info in infos]
    task_names = [name for name in names if name.startswith("task") and name.endswith(".onnx")]
    return {
        "zip": str(output_zip),
        "exists": output_zip.exists(),
        "size_bytes": output_zip.stat().st_size if output_zip.exists() else None,
        "num_entries": len(names),
        "num_task_files": len(task_names),
        "has_directories": any(name.endswith("/") for name in names),
        "has_duplicate_names": len(names) != len(set(names)),
        "invalid_names": [name for name in names if not (name.startswith("task") and name.endswith(".onnx"))],
        "oversized_entries": [
            info.filename for info in infos if info.file_size > FILESIZE_LIMIT_IN_BYTES
        ],
    }
