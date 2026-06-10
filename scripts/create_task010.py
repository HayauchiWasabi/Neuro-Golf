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
TASK_NUM = 10
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


def col_mask(c):
    arr = np.zeros((1, 1, 30, 30), dtype=np.float32)
    arr[0, 0, :9, c] = 1.0
    return arr


def build_model():
    nodes = []
    active_cols = [1, 3, 5, 7]
    nodes.append(const_node("zero_scalar", np.array([0.0], dtype=np.float32)))
    nodes.append(const_node("one_valid", np.pad(np.ones((1, 1, 9, 9), dtype=np.float32), ((0, 0), (0, 0), (0, 21), (0, 21)))))
    for c in active_cols:
        nodes.append(const_node(f"colmask{c}", col_mask(c)))
    nodes += slice_node("bars", "input", [0, 5, 0, 0], [1, 6, 30, 30], [0, 1, 2, 3])

    heights = {}
    present = {}
    bar_cols = {}
    for c in active_cols:
        nodes.append(helper.make_node("Mul", ["bars", f"colmask{c}"], [f"bar_col{c}"]))
        nodes.append(helper.make_node("ReduceSum", [f"bar_col{c}"], [f"height{c}"], axes=[0, 1, 2, 3], keepdims=1))
        nodes.append(helper.make_node("Clip", [f"height{c}"], [f"present{c}"], min=0.0, max=1.0))
        heights[c] = f"height{c}"
        present[c] = f"present{c}"
        bar_cols[c] = f"bar_col{c}"

    color_channels = {1: [], 2: [], 3: [], 4: []}
    rank_to_color = {0: 1, 1: 2, 2: 3, 3: 4}  # number of taller bars -> color
    for c in active_cols:
        greater_terms = []
        for j in active_cols:
            nodes.append(helper.make_node("Sub", [heights[j], heights[c]], [f"diff_{c}_{j}"]))
            nodes.append(helper.make_node("Clip", [f"diff_{c}_{j}"], [f"gt_raw_{c}_{j}"], min=0.0, max=1.0))
            nodes.append(helper.make_node("Mul", [f"gt_raw_{c}_{j}", present[j]], [f"gt_{c}_{j}"]))
            greater_terms.append(f"gt_{c}_{j}")
        acc = greater_terms[0]
        for j, term in enumerate(greater_terms[1:], start=1):
            nodes.append(helper.make_node("Add", [acc, term], [f"gt_sum_{c}_{j}"]))
            acc = f"gt_sum_{c}_{j}"
        for rank, color in rank_to_color.items():
            nodes.append(const_node(f"rank_const_{c}_{rank}", np.array([[[[float(rank)]]]], dtype=np.float32)))
            nodes.append(helper.make_node("Sub", [acc, f"rank_const_{c}_{rank}"], [f"rank_diff_{c}_{rank}"]))
            nodes.append(helper.make_node("Abs", [f"rank_diff_{c}_{rank}"], [f"rank_abs_{c}_{rank}"]))
            nodes.append(helper.make_node("Sub", ["one_scalar", f"rank_abs_{c}_{rank}"], [f"rank_score_{c}_{rank}"]))
            nodes.append(helper.make_node("Clip", [f"rank_score_{c}_{rank}"], [f"rank_mask_{c}_{rank}"], min=0.0, max=1.0))
            nodes.append(helper.make_node("Mul", [bar_cols[c], f"rank_mask_{c}_{rank}"], [f"colored_{c}_{rank}"]))
            color_channels[color].append(f"colored_{c}_{rank}")
    nodes.insert(0, const_node("one_scalar", np.array([[[[1.0]]]], dtype=np.float32)))

    outputs = []
    for color in range(1, 5):
        parts = color_channels[color]
        acc = parts[0]
        for idx, part in enumerate(parts[1:], start=1):
            nodes.append(helper.make_node("Add", [acc, part], [f"out{color}_sum_{idx}"]))
            acc = f"out{color}_sum_{idx}"
        nodes.append(helper.make_node("Clip", [acc], [f"out_ch{color}"], min=0.0, max=1.0))
        outputs.append(f"out_ch{color}")
    nodes.append(helper.make_node("Add", ["out_ch1", "out_ch2"], ["area12"]))
    nodes.append(helper.make_node("Add", ["out_ch3", "out_ch4"], ["area34"]))
    nodes.append(helper.make_node("Add", ["area12", "area34"], ["color_area"]))
    nodes.append(helper.make_node("Sub", ["one_valid", "color_area"], ["out_ch0_raw"]))
    nodes.append(helper.make_node("Clip", ["out_ch0_raw"], ["out_ch0"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Mul", ["out_ch1", "zero_scalar"], ["zero_ch"]))
    nodes.append(helper.make_node("Concat", ["out_ch0"] + outputs + ["zero_ch"] * 5, ["output"], axis=1))
    graph = helper.make_graph(
        nodes,
        "task010_graph",
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
