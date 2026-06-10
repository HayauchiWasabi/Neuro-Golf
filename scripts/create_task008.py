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
TASK_NUM = 8
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


def row_mask(r):
    arr = np.zeros((1, 1, 30, 30), dtype=np.float32)
    arr[:, :, r, :] = 1.0
    return arr


def col_mask(c):
    arr = np.zeros((1, 1, 30, 30), dtype=np.float32)
    arr[:, :, :, c] = 1.0
    return arr


def bbox_coord_nodes(nodes, prefix, mask):
    row_present = []
    col_present = []
    for i in range(30):
        nodes.append(const_node(f"{prefix}_rowmask{i}", row_mask(i)))
        nodes.append(helper.make_node("Mul", [mask, f"{prefix}_rowmask{i}"], [f"{prefix}_rowpix{i}"]))
        nodes.append(helper.make_node("ReduceSum", [f"{prefix}_rowpix{i}"], [f"{prefix}_rowsum{i}"], axes=[0, 1, 2, 3], keepdims=1))
        nodes.append(helper.make_node("Clip", [f"{prefix}_rowsum{i}"], [f"{prefix}_rowpres{i}"], min=0.0, max=1.0))
        row_present.append(f"{prefix}_rowpres{i}")

        nodes.append(const_node(f"{prefix}_colmask{i}", col_mask(i)))
        nodes.append(helper.make_node("Mul", [mask, f"{prefix}_colmask{i}"], [f"{prefix}_colpix{i}"]))
        nodes.append(helper.make_node("ReduceSum", [f"{prefix}_colpix{i}"], [f"{prefix}_colsum{i}"], axes=[0, 1, 2, 3], keepdims=1))
        nodes.append(helper.make_node("Clip", [f"{prefix}_colsum{i}"], [f"{prefix}_colpres{i}"], min=0.0, max=1.0))
        col_present.append(f"{prefix}_colpres{i}")

    def first_last(axis, present):
        first_terms = []
        last_terms = []
        for i in range(30):
            if i == 0:
                nodes.append(helper.make_node("Mul", [present[i], "one_scalar"], [f"{prefix}_{axis}_first_ind{i}"]))
            else:
                acc = present[0]
                for j in range(1, i):
                    nodes.append(helper.make_node("Add", [acc, present[j]], [f"{prefix}_{axis}_prev{i}_{j}"]))
                    acc = f"{prefix}_{axis}_prev{i}_{j}"
                nodes.append(helper.make_node("Clip", [acc], [f"{prefix}_{axis}_prevclip{i}"], min=0.0, max=1.0))
                nodes.append(helper.make_node("Sub", ["one_scalar", f"{prefix}_{axis}_prevclip{i}"], [f"{prefix}_{axis}_noprev{i}"]))
                nodes.append(helper.make_node("Mul", [present[i], f"{prefix}_{axis}_noprev{i}"], [f"{prefix}_{axis}_first_ind{i}"]))
            if i == 29:
                nodes.append(helper.make_node("Mul", [present[i], "one_scalar"], [f"{prefix}_{axis}_last_ind{i}"]))
            else:
                acc = present[i + 1]
                for j in range(i + 2, 30):
                    nodes.append(helper.make_node("Add", [acc, present[j]], [f"{prefix}_{axis}_next{i}_{j}"]))
                    acc = f"{prefix}_{axis}_next{i}_{j}"
                nodes.append(helper.make_node("Clip", [acc], [f"{prefix}_{axis}_nextclip{i}"], min=0.0, max=1.0))
                nodes.append(helper.make_node("Sub", ["one_scalar", f"{prefix}_{axis}_nextclip{i}"], [f"{prefix}_{axis}_nonext{i}"]))
                nodes.append(helper.make_node("Mul", [present[i], f"{prefix}_{axis}_nonext{i}"], [f"{prefix}_{axis}_last_ind{i}"]))
            nodes.append(const_node(f"{prefix}_{axis}_coord{i}", np.array([[[[float(i)]]]], dtype=np.float32)))
            nodes.append(helper.make_node("Mul", [f"{prefix}_{axis}_first_ind{i}", f"{prefix}_{axis}_coord{i}"], [f"{prefix}_{axis}_first_term{i}"]))
            nodes.append(helper.make_node("Mul", [f"{prefix}_{axis}_last_ind{i}", f"{prefix}_{axis}_coord{i}"], [f"{prefix}_{axis}_last_term{i}"]))
            first_terms.append(f"{prefix}_{axis}_first_term{i}")
            last_terms.append(f"{prefix}_{axis}_last_term{i}")
        first = first_terms[0]
        last = last_terms[0]
        for i in range(1, 30):
            nodes.append(helper.make_node("Add", [first, first_terms[i]], [f"{prefix}_{axis}_first_acc{i}"]))
            first = f"{prefix}_{axis}_first_acc{i}"
            nodes.append(helper.make_node("Add", [last, last_terms[i]], [f"{prefix}_{axis}_last_acc{i}"]))
            last = f"{prefix}_{axis}_last_acc{i}"
        return first, last

    return (*first_last("row", row_present), *first_last("col", col_present))


def shift_nodes(output, data, dr, dc, step):
    r_shift = dr * step
    c_shift = dc * step
    r0 = max(0, -r_shift)
    r1 = min(30, 30 - r_shift)
    c0 = max(0, -c_shift)
    c1 = min(30, 30 - c_shift)
    top = max(0, r_shift)
    left = max(0, c_shift)
    bottom = 30 - top - (r1 - r0)
    right = 30 - left - (c1 - c0)
    return [
        *slice_node(f"{output}_slice", data, [0, 0, r0, c0], [1, 1, r1, c1], [0, 1, 2, 3]),
        helper.make_node("Pad", [f"{output}_slice"], [output], mode="constant", pads=[0, 0, top, left, 0, 0, bottom, right], value=0.0),
    ]


def build_model():
    nodes = []
    nodes.append(const_node("one_scalar", np.array([[[[1.0]]]], dtype=np.float32)))
    nodes.append(const_node("zero_scalar", np.array([[[[0.0]]]], dtype=np.float32)))
    nodes += slice_node("black", "input", [0, 0, 0, 0], [1, 1, 30, 30], [0, 1, 2, 3])
    nodes += slice_node("red", "input", [0, 2, 0, 0], [1, 3, 30, 30], [0, 1, 2, 3])
    nodes += slice_node("cyan", "input", [0, 8, 0, 0], [1, 9, 30, 30], [0, 1, 2, 3])

    dir_parts = []
    dirs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    for d_idx, (dr, dc) in enumerate(dirs):
        step_parts = []
        prev_overlap = None
        for step in range(30):
            nodes += shift_nodes(f"shift_{d_idx}_{step}", "red", dr, dc, step)
            nodes.append(helper.make_node("Mul", [f"shift_{d_idx}_{step}", "cyan"], [f"overlap_pix_{d_idx}_{step}"]))
            nodes.append(helper.make_node("ReduceSum", [f"overlap_pix_{d_idx}_{step}"], [f"overlap_sum_{d_idx}_{step}"], axes=[0, 1, 2, 3], keepdims=1))
            nodes.append(helper.make_node("Clip", [f"overlap_sum_{d_idx}_{step}"], [f"overlap_{d_idx}_{step}"], min=0.0, max=1.0))
            if step > 0:
                nodes.append(helper.make_node("Sub", ["one_scalar", prev_overlap], [f"safe_{d_idx}_{step - 1}"]))
                nodes.append(helper.make_node("Mul", [f"safe_{d_idx}_{step - 1}", f"overlap_{d_idx}_{step}"], [f"stepgate_{d_idx}_{step - 1}"]))
                nodes.append(helper.make_node("Mul", [f"shift_{d_idx}_{step - 1}", f"stepgate_{d_idx}_{step - 1}"], [f"steppart_{d_idx}_{step - 1}"]))
                step_parts.append(f"steppart_{d_idx}_{step - 1}")
            prev_overlap = f"overlap_{d_idx}_{step}"
        acc = step_parts[0]
        for i, part in enumerate(step_parts[1:], start=1):
            nodes.append(helper.make_node("Add", [acc, part], [f"stepsum_{d_idx}_{i}"]))
            acc = f"stepsum_{d_idx}_{i}"
        dir_parts.append(acc)

    red_acc = dir_parts[0]
    for i, part in enumerate(dir_parts[1:], start=1):
        nodes.append(helper.make_node("Add", [red_acc, part], [f"redout_acc{i}"]))
        red_acc = f"redout_acc{i}"
    nodes.append(helper.make_node("Clip", [red_acc], ["out_ch2"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Add", ["black", "red"], ["valid_br"]))
    nodes.append(helper.make_node("Add", ["valid_br", "cyan"], ["valid"]))
    nodes.append(helper.make_node("Add", ["out_ch2", "cyan"], ["objects"]))
    nodes.append(helper.make_node("Sub", ["valid", "objects"], ["out_ch0_raw"]))
    nodes.append(helper.make_node("Clip", ["out_ch0_raw"], ["out_ch0"], min=0.0, max=1.0))
    zero_channels = []
    for color in [1, 3, 4, 5, 6, 7, 9]:
        nodes.append(helper.make_node("Mul", ["out_ch2", "zero_scalar"], [f"out_ch{color}"]))
        zero_channels.append(f"out_ch{color}")
    nodes.append(helper.make_node("Concat", ["out_ch0", "out_ch1", "out_ch2", "out_ch3", "out_ch4", "out_ch5", "out_ch6", "out_ch7", "cyan", "out_ch9"], ["output"], axis=1))

    graph = helper.make_graph(
        nodes,
        "task008_graph",
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
