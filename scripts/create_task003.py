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
TASK_NUM = 3
TASK_PATH = DATA_DIR / f"task{TASK_NUM:03d}.json"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / f"task{TASK_NUM:03d}.onnx"


def tensor(name, array, dtype=np.float32):
    return numpy_helper.from_array(np.asarray(array, dtype=dtype), name=name)


def const_node(name, array, dtype=np.float32):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=tensor(f"{name}_value", array, dtype=dtype),
    )


def slice_node(output, data, starts, ends, axes):
    return [
        const_node(f"{output}_starts", starts, dtype=np.int64),
        const_node(f"{output}_ends", ends, dtype=np.int64),
        const_node(f"{output}_axes", axes, dtype=np.int64),
        helper.make_node(
            "Slice",
            inputs=[data, f"{output}_starts", f"{output}_ends", f"{output}_axes"],
            outputs=[output],
        ),
    ]


def load_task():
    with TASK_PATH.open("r") as f:
        return json.load(f)


def find_period(grid):
    rows = [tuple(row) for row in grid]
    for period in range(1, 7):
        if all(rows[i] == rows[i % period] for i in range(len(rows))):
            return period
    return 6


def solve_grid(grid):
    period = find_period(grid)
    rows = [grid[i % period] for i in range(9)]
    return [[2 if cell == 1 else cell for cell in row] for row in rows]


def grid_to_tensor(grid):
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    grid = np.asarray(grid, dtype=np.int64)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            arr[0, grid[r, c], r, c] = 1.0
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


def make_row_slices(nodes):
    rows = []
    for row_idx in range(6):
        name = f"row{row_idx}"
        nodes += slice_node(name, "ch1", [0, 0, row_idx, 0], [1, 1, row_idx + 1, 3], [0, 1, 2, 3])
        rows.append(name)
    return rows


def period_raw_mask(nodes, rows, period):
    if period == 6:
        return "one_scalar"

    actual_rows = rows[period:6]
    expected_rows = [rows[i % period] for i in range(period, 6)]
    nodes.append(
        helper.make_node(
            "Concat",
            inputs=actual_rows,
            outputs=[f"p{period}_actual"],
            axis=2,
        )
    )
    nodes.append(
        helper.make_node(
            "Concat",
            inputs=expected_rows,
            outputs=[f"p{period}_expected"],
            axis=2,
        )
    )
    nodes.append(
        helper.make_node(
            "Sub",
            inputs=[f"p{period}_actual", f"p{period}_expected"],
            outputs=[f"p{period}_diff"],
        )
    )
    nodes.append(helper.make_node("Abs", inputs=[f"p{period}_diff"], outputs=[f"p{period}_abs"]))
    nodes.append(
        helper.make_node(
            "ReduceSum",
            inputs=[f"p{period}_abs"],
            outputs=[f"p{period}_sum"],
            axes=[2, 3],
            keepdims=1,
        )
    )
    nodes.append(helper.make_node("Clip", inputs=[f"p{period}_sum"], outputs=[f"p{period}_any"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Sub", inputs=["one_scalar", f"p{period}_any"], outputs=[f"p{period}_raw"]))
    return f"p{period}_raw"


def candidate_output(nodes, rows, period):
    row_names = [rows[i % period] for i in range(9)]
    cand_93 = f"cand{period}_93"
    cand_3030 = f"cand{period}_3030"
    nodes.append(helper.make_node("Concat", inputs=row_names, outputs=[cand_93], axis=2))
    nodes.append(
        helper.make_node(
            "Pad",
            inputs=[cand_93],
            outputs=[cand_3030],
            pads=[0, 0, 0, 0, 0, 0, 21, 27],
            mode="constant",
            value=0.0,
        )
    )
    return cand_3030


def build_model():
    nodes = []

    nodes.append(const_node("one_scalar", np.array([[[[1.0]]]], dtype=np.float32)))
    nodes.append(const_node("valid_area", np.pad(np.ones((1, 1, 9, 3), dtype=np.float32), ((0, 0), (0, 0), (0, 21), (0, 27)))))

    nodes += slice_node("ch1", "input", [0, 1, 0, 0], [1, 2, 6, 3], [0, 1, 2, 3])
    rows = make_row_slices(nodes)

    raw_masks = {period: period_raw_mask(nodes, rows, period) for period in range(1, 6)}

    remaining = "one_scalar"
    selected_masks = {}
    for period in range(1, 6):
        selected = f"p{period}_selected"
        nodes.append(helper.make_node("Mul", inputs=[remaining, raw_masks[period]], outputs=[selected]))
        selected_masks[period] = selected
        not_raw = f"p{period}_not_raw"
        next_remaining = f"remaining_after_p{period}"
        nodes.append(helper.make_node("Sub", inputs=["one_scalar", raw_masks[period]], outputs=[not_raw]))
        nodes.append(helper.make_node("Mul", inputs=[remaining, not_raw], outputs=[next_remaining]))
        remaining = next_remaining
    selected_masks[6] = remaining

    masked_candidates = []
    for period in range(1, 7):
        cand = candidate_output(nodes, rows, period)
        masked = f"cand{period}_masked"
        nodes.append(helper.make_node("Mul", inputs=[cand, selected_masks[period]], outputs=[masked]))
        masked_candidates.append(masked)

    running = masked_candidates[0]
    for idx, candidate in enumerate(masked_candidates[1:], start=2):
        summed = f"ch2_sum_{idx}"
        nodes.append(helper.make_node("Add", inputs=[running, candidate], outputs=[summed]))
        running = summed

    nodes.append(helper.make_node("Sub", inputs=["valid_area", running], outputs=["out_ch0"]))
    nodes.append(helper.make_node("Sub", inputs=["valid_area", "valid_area"], outputs=["zero_ch"]))
    nodes.append(
        helper.make_node(
            "Concat",
            inputs=[
                "out_ch0",
                "zero_ch",
                running,
                "zero_ch",
                "zero_ch",
                "zero_ch",
                "zero_ch",
                "zero_ch",
                "zero_ch",
                "zero_ch",
            ],
            outputs=["output"],
            axis=1,
        )
    )

    graph = helper.make_graph(
        nodes,
        "task003_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_python_rule(task):
    for split, pairs in task.items():
        for idx, pair in enumerate(pairs):
            actual = solve_grid(pair["input"])
            if actual != pair["output"]:
                raise AssertionError(f"python rule mismatch: {split}[{idx}]")


def verify_onnx(path, task):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(path.read_bytes(), options)

    passed = {}
    for split, pairs in task.items():
        passed[split] = 0
        for idx, pair in enumerate(pairs):
            result = session.run(["output"], {"input": grid_to_tensor(pair["input"])})[0]
            actual = tensor_to_grid(result)
            if actual != pair["output"]:
                raise AssertionError(f"onnx mismatch: {split}[{idx}] actual={actual} expected={pair['output']}")
            passed[split] += 1
    return passed


def verify_with_official_utils(model, task):
    common_verify_official(model, TASK_NUM, task)


def main():
    run_task(TASK_NUM, build_model, verify_python_rule)


if __name__ == "__main__":
    main()
