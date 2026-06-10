from pathlib import Path
import importlib.util
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from neurogolf_common import run_task, verify_official as common_verify_official


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "neurogolf-2026"
TASK_NUM = 2
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


def solve_grid(grid):
    a = np.asarray(grid, dtype=np.int64)
    h, w = a.shape
    wall = a == 3
    outside = np.zeros((h + 2, w + 2), dtype=bool)
    padded_wall = np.zeros((h + 2, w + 2), dtype=bool)
    padded_wall[1 : h + 1, 1 : w + 1] = wall

    queue = [(0, 0)]
    outside[0, 0] = True
    for r, c in queue:
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not outside[nr, nc] and not padded_wall[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))

    out = a.copy()
    inside = (a == 0) & ~outside[1 : h + 1, 1 : w + 1]
    out[inside] = 4
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


def clip_node(name, input_name):
    return helper.make_node("Clip", inputs=[input_name], outputs=[name], min=0.0, max=1.0)


def build_model():
    nodes = []
    initializers = []

    original_w = np.ones((1, 10, 1, 1), dtype=np.float32)
    cross_w = np.zeros((1, 1, 3, 3), dtype=np.float32)
    cross_w[0, 0, 1, 1] = 1.0
    cross_w[0, 0, 0, 1] = 1.0
    cross_w[0, 0, 1, 0] = 1.0
    cross_w[0, 0, 1, 2] = 1.0
    cross_w[0, 0, 2, 1] = 1.0

    initializers.append(tensor("original_w", original_w))
    initializers.append(tensor("cross_w", cross_w))

    nodes += slice_node("ch0", "input", [0, 0, 0, 0], [1, 1, 30, 30], [0, 1, 2, 3])
    nodes += slice_node("ch3", "input", [0, 3, 0, 0], [1, 4, 30, 30], [0, 1, 2, 3])

    nodes.append(helper.make_node("Conv", inputs=["input", "original_w"], outputs=["original_mask"]))
    nodes.append(helper.make_node("Sub", inputs=["original_mask", "ch3"], outputs=["open_mask"]))

    nodes.append(
        helper.make_node(
            "Conv",
            inputs=["original_mask", "cross_w"],
            outputs=["original_neighbors"],
            pads=[1, 1, 1, 1],
        )
    )
    nodes.append(const_node("five", np.array([[[[5.0]]]], dtype=np.float32)))
    nodes.append(helper.make_node("Sub", inputs=["five", "original_neighbors"], outputs=["boundary_score"]))
    nodes.append(clip_node("boundary_mask", "boundary_score"))
    nodes.append(helper.make_node("Mul", inputs=["open_mask", "boundary_mask"], outputs=["exterior_00"]))

    previous = "exterior_00"
    for i in range(1, 61):
        neighbor = f"exterior_neighbors_{i:02d}"
        clipped = f"exterior_reach_{i:02d}"
        current = f"exterior_{i:02d}"
        nodes.append(
            helper.make_node(
                "Conv",
                inputs=[previous, "cross_w"],
                outputs=[neighbor],
                pads=[1, 1, 1, 1],
            )
        )
        nodes.append(clip_node(clipped, neighbor))
        nodes.append(helper.make_node("Mul", inputs=["open_mask", clipped], outputs=[current]))
        previous = current

    nodes.append(const_node("one", np.array([[[[1.0]]]], dtype=np.float32)))
    nodes.append(helper.make_node("Sub", inputs=["one", previous], outputs=["not_exterior"]))
    nodes.append(helper.make_node("Mul", inputs=["open_mask", "not_exterior"], outputs=["inside"]))
    nodes.append(helper.make_node("Sub", inputs=["ch0", "inside"], outputs=["out_ch0"]))

    nodes.append(const_node("zero_scalar", np.array([0.0], dtype=np.float32)))
    nodes.append(helper.make_node("Mul", inputs=["ch0", "zero_scalar"], outputs=["zero_ch"]))
    nodes.append(
        helper.make_node(
            "Concat",
            inputs=[
                "out_ch0",
                "zero_ch",
                "zero_ch",
                "ch3",
                "inside",
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
        "task002_graph",
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
                raise AssertionError(f"onnx mismatch: {split}[{idx}]")
            passed[split] += 1
    return passed


def verify_with_official_utils(model, task):
    common_verify_official(model, TASK_NUM, task)


def main():
    run_task(TASK_NUM, build_model, verify_python_rule)


if __name__ == "__main__":
    main()
