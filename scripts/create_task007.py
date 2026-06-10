from pathlib import Path
import importlib.util
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from neurogolf_common import run_task, verify_official as common_verify_official

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "neurogolf-2026"
TASK_NUM = 7
TASK_PATH = DATA_DIR / f"task{TASK_NUM:03d}.json"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / f"task{TASK_NUM:03d}.onnx"


def tensor(name, array, dtype=np.float32):
    return numpy_helper.from_array(np.asarray(array, dtype=dtype), name=name)


def const_node(name, array, dtype=np.float32):
    return helper.make_node("Constant", [], [name], value=tensor(f"{name}_value", array, dtype=dtype))


def slice_node(output, data, starts, ends, axes):
    return [
        const_node(f"{output}_starts", starts, np.int64),
        const_node(f"{output}_ends", ends, np.int64),
        const_node(f"{output}_axes", axes, np.int64),
        helper.make_node("Slice", [data, f"{output}_starts", f"{output}_ends", f"{output}_axes"], [output]),
    ]


def grid_to_tensor(grid):
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def tensor_to_grid(output):
    output = (output > 0.5).astype(np.float32)
    rows = []
    for r in range(output.shape[2]):
        row = []
        for c in range(output.shape[3]):
            colors = np.flatnonzero(output[0, :, r, c] == 1.0)
            row.append(int(colors[0]) if len(colors) == 1 else None)
        while row and row[-1] is None:
            row.pop()
        rows.append(row)
    while rows and not rows[-1]:
        rows.pop()
    return rows


def class_mask(k):
    arr = np.zeros((1, 1, 30, 30), dtype=np.float32)
    for r in range(7):
        for c in range(7):
            if (r + c) % 3 == k:
                arr[0, 0, r, c] = 1.0
    return arr


def build_model():
    nodes = []
    nodes.append(const_node("zero_scalar", np.array([0.0], dtype=np.float32)))
    for k in range(3):
        nodes.append(const_node(f"class{k}", class_mask(k)))

    color_outputs = []
    for color in range(1, 10):
        nodes += slice_node(f"ch{color}", "input", [0, color, 0, 0], [1, color + 1, 30, 30], [0, 1, 2, 3])
        parts = []
        for k in range(3):
            nodes.append(helper.make_node("Mul", [f"ch{color}", f"class{k}"], [f"evidence_pixels_{color}_{k}"]))
            nodes.append(
                helper.make_node(
                    "ReduceSum",
                    [f"evidence_pixels_{color}_{k}"],
                    [f"evidence_sum_{color}_{k}"],
                    axes=[0, 1, 2, 3],
                    keepdims=1,
                )
            )
            nodes.append(helper.make_node("Clip", [f"evidence_sum_{color}_{k}"], [f"evidence_{color}_{k}"], min=0.0, max=1.0))
            nodes.append(helper.make_node("Mul", [f"class{k}", f"evidence_{color}_{k}"], [f"part_{color}_{k}"]))
            parts.append(f"part_{color}_{k}")
        nodes.append(helper.make_node("Add", [parts[0], parts[1]], [f"sum_{color}_01"]))
        nodes.append(helper.make_node("Add", [f"sum_{color}_01", parts[2]], [f"out_ch{color}"]))
        color_outputs.append(f"out_ch{color}")

    nodes.append(helper.make_node("Mul", [color_outputs[0], "zero_scalar"], ["out_ch0"]))
    nodes.append(helper.make_node("Concat", ["out_ch0"] + color_outputs, ["output"], axis=1))
    graph = helper.make_graph(
        nodes,
        "task007_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_onnx(path, task):
    session = ort.InferenceSession(path.read_bytes())
    passed = {}
    for split, pairs in task.items():
        passed[split] = 0
        for idx, pair in enumerate(pairs):
            actual = tensor_to_grid(session.run(["output"], {"input": grid_to_tensor(pair["input"])})[0])
            if actual != pair["output"]:
                raise AssertionError(f"{split}[{idx}] actual={actual} expected={pair['output']}")
            passed[split] += 1
    return passed


def verify_official(model, task):
    common_verify_official(model, TASK_NUM, task)


def main():
    run_task(TASK_NUM, build_model)


if __name__ == "__main__":
    main()
