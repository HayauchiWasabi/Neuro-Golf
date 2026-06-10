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
TASK_NUM = 6
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


def build_model():
    nodes = []
    nodes += slice_node("left", "input", [0, 1, 0, 0], [1, 2, 3, 3], [0, 1, 2, 3])
    nodes += slice_node("right", "input", [0, 1, 0, 4], [1, 2, 3, 7], [0, 1, 2, 3])
    nodes.append(helper.make_node("Mul", ["left", "right"], ["overlap"]))
    nodes.append(helper.make_node("Sub", ["one_33", "overlap"], ["out0_33"]))
    nodes.append(helper.make_node("Pad", ["out0_33"], ["out0"], pads=[0, 0, 0, 0, 0, 0, 27, 27], mode="constant", value=0.0))
    zero_chs = []
    for i in [1, 3, 4, 5, 6, 7, 8, 9]:
        name = f"zero{i}"
        nodes.append(helper.make_node("Mul", ["out0", "zero_scalar"], [name]))
        zero_chs.append(name)
    nodes.append(helper.make_node("Pad", ["overlap"], ["out2"], pads=[0, 0, 0, 0, 0, 0, 27, 27], mode="constant", value=0.0))
    nodes.append(helper.make_node("Concat", ["out0", zero_chs[0], "out2"] + zero_chs[1:], ["output"], axis=1))
    nodes.insert(0, const_node("zero_scalar", np.array([0.0], dtype=np.float32)))
    nodes.insert(0, const_node("one_33", np.ones((1, 1, 3, 3), dtype=np.float32)))
    graph = helper.make_graph(
        nodes,
        "task006_graph",
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
                raise AssertionError(f"{split}[{idx}]")
            passed[split] += 1
    return passed


def verify_official(model, task):
    common_verify_official(model, TASK_NUM, task)


def main():
    run_task(TASK_NUM, build_model)


if __name__ == "__main__":
    main()
