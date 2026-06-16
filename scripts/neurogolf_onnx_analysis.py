from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper

from neurogolf_score import sanitize_model


def load_task(task_name: str, data_dir: str | Path) -> dict:
    task_path = Path(data_dir) / f"{task_name}.json"
    with open(task_path) as f:
        return json.load(f)


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def color_counts_for_examples(examples: list[dict]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for example in examples:
        for key in ["input", "output"]:
            for row in example[key]:
                counts.update(row)
    return dict(sorted(counts.items()))


def summarize_task(task_name: str, data_dir: str | Path) -> dict:
    task = load_task(task_name, data_dir)
    subsets = {name: task.get(name, []) for name in ["train", "test", "arc-gen"]}
    rows = []
    for subset_name, examples in subsets.items():
        for idx, example in enumerate(examples):
            input_h, input_w = grid_shape(example["input"])
            output_h, output_w = grid_shape(example["output"])
            rows.append(
                {
                    "subset": subset_name,
                    "index": idx,
                    "input_shape": f"{input_h}x{input_w}",
                    "output_shape": f"{output_h}x{output_w}",
                    "input_cells": input_h * input_w,
                    "output_cells": output_h * output_w,
                }
            )
    return {
        "task": task_name,
        "num_train": len(subsets["train"]),
        "num_test": len(subsets["test"]),
        "num_arc_gen": len(subsets["arc-gen"]),
        "color_counts": color_counts_for_examples(rows and sum(subsets.values(), []) or []),
        "examples": rows,
        "raw": task,
    }


def _tensor_type_summary(value_info: onnx.ValueInfoProto) -> dict:
    item = {
        "name": value_info.name,
        "elem_type": None,
        "shape": None,
    }
    tensor_type = value_info.type.tensor_type
    if tensor_type.HasField("elem_type"):
        item["elem_type"] = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    if tensor_type.HasField("shape"):
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        item["shape"] = dims
    return item


def _attribute_summary(attr: onnx.AttributeProto, max_items: int = 8) -> object:
    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    if attr.type == onnx.AttributeProto.INT:
        return attr.i
    if attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="replace")
    if attr.type == onnx.AttributeProto.FLOATS:
        values = list(attr.floats)
        return values[:max_items] + (["..."] if len(values) > max_items else [])
    if attr.type == onnx.AttributeProto.INTS:
        values = list(attr.ints)
        return values[:max_items] + (["..."] if len(values) > max_items else [])
    if attr.type == onnx.AttributeProto.STRINGS:
        values = [s.decode("utf-8", errors="replace") for s in attr.strings]
        return values[:max_items] + (["..."] if len(values) > max_items else [])
    if attr.type == onnx.AttributeProto.TENSOR:
        dims = list(attr.t.dims)
        return {"tensor_dtype": onnx.TensorProto.DataType.Name(attr.t.data_type), "dims": dims, "numel": math.prod(dims) if dims else 1}
    if attr.type == onnx.AttributeProto.SPARSE_TENSOR:
        dims = list(attr.sparse_tensor.dims)
        return {"sparse_tensor_dims": dims}
    if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
        return "<graph attribute>"
    return f"<attribute type {attr.type}>"


def summarize_initializers(model: onnx.ModelProto, max_values: int = 12) -> list[dict]:
    rows = []
    for init in model.graph.initializer:
        dims = list(init.dims)
        row = {
            "name": init.name,
            "dtype": onnx.TensorProto.DataType.Name(init.data_type),
            "dims": dims,
            "numel": math.prod(dims) if dims else 1,
            "bytes": 0,
            "sample": [],
        }
        try:
            arr = numpy_helper.to_array(init)
            row["bytes"] = arr.nbytes
            flat = arr.reshape(-1)
            sample = flat[:max_values].tolist()
            row["sample"] = [x.item() if isinstance(x, np.generic) else x for x in sample]
        except Exception as exc:
            row["sample"] = [f"<unavailable: {exc!r}>"]
        rows.append(row)
    return rows


def summarize_nodes(model: onnx.ModelProto) -> list[dict]:
    rows = []
    for idx, node in enumerate(model.graph.node):
        rows.append(
            {
                "index": idx,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {attr.name: _attribute_summary(attr) for attr in node.attribute},
            }
        )
    return rows


def summarize_model(path: str | Path) -> dict:
    path = Path(path)
    model = onnx.load(path)
    graph = model.graph
    nodes = summarize_nodes(model)
    initializers = summarize_initializers(model)
    return {
        "path": str(path),
        "filename": path.name,
        "filesize": path.stat().st_size,
        "ir_version": model.ir_version,
        "opset_imports": [{"domain": opset.domain, "version": opset.version} for opset in model.opset_import],
        "producer_name": model.producer_name,
        "graph_name": graph.name,
        "inputs": [_tensor_type_summary(x) for x in graph.input],
        "outputs": [_tensor_type_summary(x) for x in graph.output],
        "value_info": [_tensor_type_summary(x) for x in graph.value_info],
        "num_nodes": len(nodes),
        "num_initializers": len(initializers),
        "op_counts": dict(sorted(Counter(row["op_type"] for row in nodes).items())),
        "nodes": nodes,
        "initializers": initializers,
    }


def compact_model_markdown(summary: dict, score_row: dict | None = None, max_nodes: int = 120) -> str:
    lines = [
        f"# ONNX analysis: {summary['filename']}",
        "",
        "## Basic",
        f"- path: {summary['path']}",
        f"- filesize: {summary['filesize']} bytes",
        f"- ir_version: {summary['ir_version']}",
        f"- opset_imports: {summary['opset_imports']}",
        f"- inputs: {summary['inputs']}",
        f"- outputs: {summary['outputs']}",
        f"- nodes: {summary['num_nodes']}",
        f"- initializers: {summary['num_initializers']}",
        f"- op_counts: {summary['op_counts']}",
    ]
    if score_row:
        lines.extend(
            [
                "",
                "## Score",
                f"- status: {score_row.get('status')}",
                f"- memory: {score_row.get('memory')}",
                f"- params: {score_row.get('params')}",
                f"- cost: {score_row.get('cost')}",
                f"- score: {score_row.get('score')}",
                f"- score_error: {score_row.get('error')}",
            ]
        )
    lines.extend(["", "## Initializers"])
    for row in summary["initializers"]:
        lines.append(
            f"- {row['name']}: dtype={row['dtype']}, dims={row['dims']}, numel={row['numel']}, bytes={row['bytes']}, sample={row['sample']}"
        )
    lines.extend(["", "## Nodes"])
    for row in summary["nodes"][:max_nodes]:
        lines.append(
            f"- #{row['index']} {row['op_type']} name={row['name']} inputs={row['inputs']} outputs={row['outputs']} attrs={row['attributes']}"
        )
    if len(summary["nodes"]) > max_nodes:
        lines.append(f"- ... {len(summary['nodes']) - max_nodes} more nodes omitted")
    return "\n".join(lines)


def gpt_analysis_prompt(task_summary: dict, model_markdown: str) -> str:
    return "\n".join(
        [
            "You are helping optimize a NeuroGolf ONNX model.",
            "",
            "Rules:",
            "- Input tensor: float32 [1, 10, 30, 30], name 'input'.",
            "- Output tensor: float32 [1, 10, 30, 30], name 'output'.",
            "- Static tensor shapes only.",
            "- One input and one output only.",
            "- Banned ops: Loop, Scan, NonZero, Unique, Script, Function, Compress.",
            "- Optimize score by reducing memory + params.",
            "- Assume public examples pass unless told otherwise.",
            "",
            "Requested output:",
            "1. Summarize what the current ONNX appears to do.",
            "2. Identify redundant or expensive parts.",
            "3. Propose safe rewrite candidates ranked by risk and likely score gain.",
            "4. Do not write code yet.",
            "",
            "Task summary:",
            f"- task: {task_summary['task']}",
            f"- train examples: {task_summary['num_train']}",
            f"- test examples: {task_summary['num_test']}",
            f"- arc-gen examples: {task_summary['num_arc_gen']}",
            f"- color counts: {task_summary['color_counts']}",
            f"- example shapes: {task_summary['examples']}",
            "",
            model_markdown,
        ]
    )


def convert_example_to_tensors(example: dict) -> dict[str, np.ndarray] | None:
    tensors = {}
    for mode in ["input", "output"]:
        tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
        grid = example[mode]
        if max(len(grid), len(grid[0])) > 30:
            return None
        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                tensor[0, color, r, c] = 1.0
        tensors[mode] = tensor
    return tensors


def output_tensor_to_grid(output: np.ndarray) -> list[list[int]]:
    grid = []
    _, channels, height, width = output.shape
    for row in range(height):
        cells = []
        for col in range(width):
            colors = [c for c in range(channels) if output[0, c, row, col] == 1]
            cells.append(colors[0] if len(colors) == 1 else (11 if colors else 10))
        while cells and cells[-1] == 10:
            cells.pop()
        grid.append(cells)
    while grid and not grid[-1]:
        grid.pop()
    return grid


def run_model_on_examples(model_path: str | Path, task_name: str, data_dir: str | Path) -> dict:
    task = load_task(task_name, data_dir)
    model = sanitize_model(onnx.load(model_path))
    if model is None:
        return {"status": "error", "error": "model sanitization failed", "rows": []}

    try:
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = onnxruntime.InferenceSession(model.SerializeToString(), options)
    except Exception as exc:
        return {"status": "error", "error": repr(exc), "rows": []}

    rows = []
    first_failure = None
    for subset in ["train", "test", "arc-gen"]:
        for idx, example in enumerate(task.get(subset, [])):
            tensors = convert_example_to_tensors(example)
            if tensors is None:
                rows.append({"subset": subset, "index": idx, "status": "skipped_large_grid"})
                continue
            try:
                raw_output = session.run(["output"], {"input": tensors["input"]})[0]
                output = (raw_output > 0.0).astype(np.float32)
                passed = bool(np.array_equal(output, tensors["output"]))
                row = {"subset": subset, "index": idx, "status": "pass" if passed else "fail"}
                if not passed:
                    actual_grid = output_tensor_to_grid(output)
                    row["expected_shape"] = grid_shape(example["output"])
                    row["actual_shape"] = grid_shape(actual_grid)
                    if first_failure is None:
                        first_failure = {
                            "subset": subset,
                            "index": idx,
                            "input": example["input"],
                            "expected": example["output"],
                            "actual": actual_grid,
                        }
                rows.append(row)
            except Exception as exc:
                row = {"subset": subset, "index": idx, "status": "error", "error": repr(exc)}
                rows.append(row)
                if first_failure is None:
                    first_failure = {"subset": subset, "index": idx, "error": repr(exc)}

    counts = dict(Counter(row["status"] for row in rows))
    ok = all(row["status"] in {"pass", "skipped_large_grid"} for row in rows)
    return {
        "status": "ok" if ok else "fail",
        "counts": counts,
        "rows": rows,
        "first_failure": first_failure,
    }


def compare_score_rows(base: dict, candidate: dict) -> dict:
    keys = ["filesize", "memory", "params", "cost", "score"]
    comparison = {}
    for key in keys:
        if base.get(key) is None or candidate.get(key) is None:
            comparison[f"{key}_delta"] = None
        else:
            comparison[f"{key}_delta"] = candidate[key] - base[key]
    comparison["base_status"] = base.get("status")
    comparison["candidate_status"] = candidate.get("status")
    return comparison
