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
TASK_NUM = 5
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
        helper.make_node("Slice", inputs=[data, f"{output}_starts", f"{output}_ends", f"{output}_axes"], outputs=[output]),
    ]


def shift_node(nodes, output, data, dr, dc):
    top = max(dr, 0)
    bottom = max(-dr, 0)
    left = max(dc, 0)
    right = max(-dc, 0)
    padded = f"{output}_padded"
    nodes.append(
        helper.make_node(
            "Pad",
            inputs=[data],
            outputs=[padded],
            pads=[0, 0, top, left, 0, 0, bottom, right],
            mode="constant",
            value=0.0,
        )
    )
    start_r = max(-dr, 0)
    start_c = max(-dc, 0)
    nodes += slice_node(output, padded, [0, 0, start_r, start_c], [1, 1, start_r + 30, start_c + 30], [0, 1, 2, 3])
    return output


def load_task():
    with TASK_PATH.open("r") as f:
        return json.load(f)


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


def solve_grid_with_onnx_rule(grid):
    a = grid_to_tensor(grid)
    # Filled by ONNX validation path; this helper is intentionally omitted to avoid
    # maintaining two separate implementations for this more complex task.
    raise NotImplementedError


def build_model():
    nodes = []
    initializers = []

    count_w = np.ones((9, 1, 3, 3), dtype=np.float32)
    expand_w = np.ones((9, 1, 3, 3), dtype=np.float32)
    initializers.append(tensor("count_w", count_w))
    initializers.append(tensor("expand_w", expand_w))

    nodes.append(const_node("one_scalar", np.array([[[[1.0]]]], dtype=np.float32)))
    nodes.append(const_node("zero_scalar", np.array([0.0], dtype=np.float32)))

    nodes += slice_node("color_input", "input", [0, 1, 0, 0], [1, 10, 30, 30], [0, 1, 2, 3])
    nodes.append(helper.make_node("Conv", inputs=["color_input", "count_w"], outputs=["window_counts"], group=9))
    nodes.append(
        helper.make_node(
            "ReduceMax",
            inputs=["window_counts"],
            outputs=["max_count"],
            axes=[1, 2, 3],
            keepdims=1,
        )
    )
    nodes.append(helper.make_node("Sub", inputs=["window_counts", "max_count"], outputs=["count_minus_max"]))
    nodes.append(helper.make_node("Add", inputs=["count_minus_max", "one_scalar"], outputs=["base_window_score"]))
    nodes.append(helper.make_node("Clip", inputs=["base_window_score"], outputs=["base_window_select"], min=0.0, max=1.0))
    nodes.append(
        helper.make_node(
            "ConvTranspose",
            inputs=["base_window_select", "expand_w"],
            outputs=["base_window_area"],
            group=9,
        )
    )
    nodes.append(helper.make_node("Mul", inputs=["color_input", "base_window_area"], outputs=["base_by_color"]))
    nodes.append(
        helper.make_node(
            "ReduceSum",
            inputs=["base_by_color"],
            outputs=["base_mask"],
            axes=[1],
            keepdims=1,
        )
    )

    repeated_masks = {}
    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        parts = []
        for step in range(1, 4):
            name = f"repeat_{dr}_{dc}_{step}".replace("-", "m")
            shift_node(nodes, name, "base_mask", dr * 4 * step, dc * 4 * step)
            parts.append(name)
        acc = parts[0]
        for idx, part in enumerate(parts[1:], start=2):
            summed = f"repeat_sum_{dr}_{dc}_{idx}".replace("-", "m")
            nodes.append(helper.make_node("Add", inputs=[acc, part], outputs=[summed]))
            acc = summed
        clipped = f"repeat_mask_{dr}_{dc}".replace("-", "m")
        nodes.append(helper.make_node("Clip", inputs=[acc], outputs=[clipped], min=0.0, max=1.0))
        repeated_masks[(dr, dc)] = clipped

    nodes.append(
        helper.make_node(
            "ReduceSum",
            inputs=["input"],
            outputs=["original_area_sum"],
            axes=[1],
            keepdims=1,
        )
    )
    nodes.append(helper.make_node("Clip", inputs=["original_area_sum"], outputs=["original_area"], min=0.0, max=1.0))

    color_outputs = []
    for color in range(1, 10):
        ch = f"marker_ch{color}"
        nodes += slice_node(ch, "input", [0, color, 0, 0], [1, color + 1, 30, 30], [0, 1, 2, 3])
        nodes += slice_node(f"base_ch{color}", "base_by_color", [0, color - 1, 0, 0], [1, color, 30, 30], [0, 1, 2, 3])

        contributions = [f"base_ch{color}"]
        for dr, dc in repeated_masks:
            adj = f"adj_{color}_{dr}_{dc}".replace("-", "m")
            shift_node(nodes, adj, "base_mask", dr * 4, dc * 4)
            nodes.append(helper.make_node("Mul", inputs=[ch, adj], outputs=[f"evidence_pixels_{color}_{dr}_{dc}".replace("-", "m")]))
            evidence_sum = f"evidence_sum_{color}_{dr}_{dc}".replace("-", "m")
            nodes.append(
                helper.make_node(
                    "ReduceSum",
                    inputs=[f"evidence_pixels_{color}_{dr}_{dc}".replace("-", "m")],
                    outputs=[evidence_sum],
                    axes=[0, 1, 2, 3],
                    keepdims=1,
                )
            )
            evidence = f"evidence_{color}_{dr}_{dc}".replace("-", "m")
            nodes.append(helper.make_node("Clip", inputs=[evidence_sum], outputs=[evidence], min=0.0, max=1.0))
            contrib = f"contrib_{color}_{dr}_{dc}".replace("-", "m")
            nodes.append(helper.make_node("Mul", inputs=[repeated_masks[(dr, dc)], evidence], outputs=[contrib]))
            contributions.append(contrib)

        acc = contributions[0]
        for idx, contrib in enumerate(contributions[1:], start=2):
            summed = f"color{color}_sum_{idx}"
            nodes.append(helper.make_node("Add", inputs=[acc, contrib], outputs=[summed]))
            acc = summed
        masked = f"out_ch{color}_masked"
        nodes.append(helper.make_node("Mul", inputs=[acc, "original_area"], outputs=[masked]))
        clipped = f"out_ch{color}"
        nodes.append(helper.make_node("Clip", inputs=[masked], outputs=[clipped], min=0.0, max=1.0))
        color_outputs.append(clipped)

    acc = color_outputs[0]
    for idx, out in enumerate(color_outputs[1:], start=2):
        summed = f"color_area_sum_{idx}"
        nodes.append(helper.make_node("Add", inputs=[acc, out], outputs=[summed]))
        acc = summed
    nodes.append(helper.make_node("Clip", inputs=[acc], outputs=["color_area"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Sub", inputs=["original_area", "color_area"], outputs=["out_ch0_raw"]))
    nodes.append(helper.make_node("Clip", inputs=["out_ch0_raw"], outputs=["out_ch0"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Concat", inputs=["out_ch0"] + color_outputs, outputs=["output"], axis=1))

    graph = helper.make_graph(
        nodes,
        "task005_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model


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
    run_task(TASK_NUM, build_model)


if __name__ == "__main__":
    main()
