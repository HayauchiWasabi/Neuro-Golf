from pathlib import Path
import importlib.util
import json
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "neurogolf-2026"
OUTPUT_DIR = ROOT / "outputs"
ARTIFACTS_DIR = ROOT / "artifacts"
OFFICIAL_ONNX_DIR = ARTIFACTS_DIR / "official-onnx"
OFFICIAL_PROFILE_DIR = ARTIFACTS_DIR / "official-profiles"
OFFICIAL_WORK_DIR = ARTIFACTS_DIR / "official-work"


def task_path(task_num):
    return DATA_DIR / f"task{task_num:03d}.json"


def output_path(task_num):
    return OUTPUT_DIR / f"task{task_num:03d}.onnx"


def load_task(task_num):
    with task_path(task_num).open("r") as f:
        return json.load(f)


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


def reshape_node(output, data, shape):
    return [
        const_node(f"{output}_shape", shape, np.int64),
        helper.make_node("Reshape", [data, f"{output}_shape"], [output]),
    ]


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


def verify_official(model, task_num, task):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    spec = importlib.util.spec_from_file_location("neurogolf_utils", DATA_DIR / "neurogolf_utils" / "neurogolf_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._NEUROGOLF_DIR = str(DATA_DIR) + "/"

    OFFICIAL_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    OFFICIAL_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    OFFICIAL_WORK_DIR.mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(OFFICIAL_WORK_DIR)
        module.verify_network(model, task_num, task)
    finally:
        os.chdir(previous_cwd)
        generated_onnx = OFFICIAL_WORK_DIR / f"task{task_num:03d}.onnx"
        if generated_onnx.exists():
            generated_onnx.replace(OFFICIAL_ONNX_DIR / generated_onnx.name)
        for profile in OFFICIAL_WORK_DIR.glob(f"{task_num:03d}_*.json"):
            profile.replace(OFFICIAL_PROFILE_DIR / profile.name)


def compact_model(model, keep=("input", "output")):
    keep = set(keep)
    mapping = {}
    next_id = 0
    for node in model.graph.node:
        for idx, name in enumerate(node.output):
            if name and name not in keep:
                if name not in mapping:
                    mapping[name] = f"t{next_id}"
                    next_id += 1
                node.output[idx] = mapping[name]
        node.name = ""
        for attr in node.attribute:
            if attr.t.name:
                attr.t.name = ""
    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name in mapping:
                node.input[idx] = mapping[name]


def run_task(task_num, build_model, verify_python_rule=None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    task = load_task(task_num)
    if verify_python_rule is not None:
        verify_python_rule(task)
        print("python rule ok")
    model = build_model()
    path = output_path(task_num)
    onnx.save(model, path)
    print(f"saved: {path.relative_to(ROOT)}")
    print(f"filesize: {path.stat().st_size} bytes")
    print(f"onnx local ok: {verify_onnx(path, task)}")
    verify_official(model, task_num, task)
