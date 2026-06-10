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
TASK_NUM = 4
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


def reshape_node(output, data, shape):
    return [
        const_node(f"{output}_shape", shape, dtype=np.int64),
        helper.make_node("Reshape", inputs=[data, f"{output}_shape"], outputs=[output]),
    ]


def load_task():
    with TASK_PATH.open("r") as f:
        return json.load(f)


def components(grid):
    from collections import deque

    a = np.asarray(grid, dtype=np.int64)
    h, w = a.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = [(dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if dr or dc]
    result = []
    for r in range(h):
        for c in range(w):
            if a[r, c] == 0 or seen[r, c]:
                continue
            color = a[r, c]
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and a[nr, nc] == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            result.append((color, cells))
    return result


def solve_grid(grid):
    a = np.asarray(grid, dtype=np.int64)
    out = np.zeros_like(a)
    for color, cells in components(a):
        max_row = max(r for r, _ in cells)
        max_col = max(c for _, c in cells)
        for r, c in cells:
            nc = c if r == max_row else min(c + 1, max_col)
            out[r, nc] = color
    return out


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


def build_presence_last_mask(nodes, source, prefix, axis):
    if axis == "row":
        nodes.append(
            helper.make_node(
                "ReduceSum",
                inputs=[source],
                outputs=[f"{prefix}_presence_sum"],
                axes=[3],
                keepdims=0,
            )
        )
        mask_shape = [1, 1, 30, 1]
    elif axis == "col":
        nodes.append(
            helper.make_node(
                "ReduceSum",
                inputs=[source],
                outputs=[f"{prefix}_presence_sum"],
                axes=[2],
                keepdims=0,
            )
        )
        mask_shape = [1, 1, 1, 30]
    else:
        raise ValueError(axis)

    nodes.append(helper.make_node("Clip", inputs=[f"{prefix}_presence_sum"], outputs=[f"{prefix}_presence"], min=0.0, max=1.0))
    nodes += reshape_node(f"{prefix}_presence_2d", f"{prefix}_presence", [1, 30])
    nodes.append(helper.make_node("MatMul", inputs=[f"{prefix}_presence_2d", "after_matrix"], outputs=[f"{prefix}_after_count"]))
    nodes.append(helper.make_node("Clip", inputs=[f"{prefix}_after_count"], outputs=[f"{prefix}_has_after"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Sub", inputs=["one_2d", f"{prefix}_has_after"], outputs=[f"{prefix}_no_after"]))
    nodes.append(helper.make_node("Mul", inputs=[f"{prefix}_presence_2d", f"{prefix}_no_after"], outputs=[f"{prefix}_last_2d"]))
    nodes += reshape_node(f"{prefix}_last_mask", f"{prefix}_last_2d", mask_shape)
    return f"{prefix}_last_mask"


def build_model():
    nodes = []
    initializers = []

    after_matrix = np.zeros((30, 30), dtype=np.float32)
    for src in range(30):
        for dst in range(30):
            if src > dst:
                after_matrix[src, dst] = 1.0
    initializers.append(tensor("after_matrix", after_matrix))

    nodes.append(const_node("one_2d", np.ones((1, 30), dtype=np.float32)))
    nodes.append(const_node("one_scalar", np.array([[[[1.0]]]], dtype=np.float32)))
    nodes.append(const_node("zero_scalar", np.array([0.0], dtype=np.float32)))

    color_outputs = []
    for color in range(1, 10):
        ch = f"ch{color}"
        nodes += slice_node(ch, "input", [0, color, 0, 0], [1, color + 1, 30, 30], [0, 1, 2, 3])

        bottom_mask = build_presence_last_mask(nodes, ch, f"c{color}_bottom", "row")
        right_mask = build_presence_last_mask(nodes, ch, f"c{color}_right", "col")

        nodes.append(helper.make_node("Mul", inputs=[ch, bottom_mask], outputs=[f"c{color}_bottom_pixels"]))
        nodes.append(helper.make_node("Mul", inputs=[ch, right_mask], outputs=[f"c{color}_right_pixels"]))

        nodes.append(helper.make_node("Sub", inputs=["one_scalar", bottom_mask], outputs=[f"c{color}_not_bottom"]))
        nodes.append(helper.make_node("Sub", inputs=["one_scalar", right_mask], outputs=[f"c{color}_not_right"]))
        nodes.append(helper.make_node("Mul", inputs=[ch, f"c{color}_not_bottom"], outputs=[f"c{color}_nonbottom"]))
        nodes.append(helper.make_node("Mul", inputs=[f"c{color}_nonbottom", f"c{color}_not_right"], outputs=[f"c{color}_shift_source"]))
        nodes.append(
            helper.make_node(
                "Pad",
                inputs=[f"c{color}_shift_source"],
                outputs=[f"c{color}_padded_shift"],
                pads=[0, 0, 0, 1, 0, 0, 0, 0],
                mode="constant",
                value=0.0,
            )
        )
        nodes += slice_node(
            f"c{color}_shifted",
            f"c{color}_padded_shift",
            [0, 0, 0, 0],
            [1, 1, 30, 30],
            [0, 1, 2, 3],
        )

        nodes.append(helper.make_node("Add", inputs=[f"c{color}_shifted", f"c{color}_bottom_pixels"], outputs=[f"c{color}_shift_bottom"]))
        nodes.append(helper.make_node("Add", inputs=[f"c{color}_shift_bottom", f"c{color}_right_pixels"], outputs=[f"out_ch{color}"]))
        nodes.append(helper.make_node("Clip", inputs=[f"out_ch{color}"], outputs=[f"out_ch{color}_clip"], min=0.0, max=1.0))
        color_outputs.append(f"out_ch{color}_clip")

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

    running = color_outputs[0]
    for idx, output in enumerate(color_outputs[1:], start=2):
        summed = f"color_sum_{idx}"
        nodes.append(helper.make_node("Add", inputs=[running, output], outputs=[summed]))
        running = summed
    nodes.append(helper.make_node("Clip", inputs=[running], outputs=["color_area"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Sub", inputs=["original_area", "color_area"], outputs=["out_ch0"]))
    nodes.append(helper.make_node("Clip", inputs=["out_ch0"], outputs=["out_ch0_clip"], min=0.0, max=1.0))

    nodes.append(
        helper.make_node(
            "Concat",
            inputs=["out_ch0_clip"] + color_outputs,
            outputs=["output"],
            axis=1,
        )
    )

    graph = helper.make_graph(
        nodes,
        "task004_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_python_rule(task):
    for split, pairs in task.items():
        for idx, pair in enumerate(pairs):
            actual = solve_grid(pair["input"]).tolist()
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
