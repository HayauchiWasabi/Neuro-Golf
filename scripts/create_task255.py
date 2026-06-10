from pathlib import Path

import onnx

from neurogolf_common import run_task


ROOT = Path(__file__).resolve().parents[1]
TASK_NUM = 255
SOURCE_PATH = ROOT / "Sample" / "submission" / f"task{TASK_NUM:03d}.onnx"


def _attr_key(node):
    parts = []
    for attr in node.attribute:
        parts.append((attr.name, attr.SerializeToString()))
    return tuple(sorted(parts))


def dedupe_identical_nodes(model):
    replacement = {}
    seen = {}
    keep_nodes = []

    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name in replacement:
                node.input[idx] = replacement[name]

        key = (node.op_type, tuple(node.input), _attr_key(node))
        if key in seen and len(node.output) == len(seen[key]):
            for old, new in zip(node.output, seen[key]):
                replacement[old] = new
            continue

        seen[key] = tuple(node.output)
        keep_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(keep_nodes)

    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name in replacement:
                node.input[idx] = replacement[name]


def build_model():
    model = onnx.load(SOURCE_PATH)
    dedupe_identical_nodes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    run_task(TASK_NUM, build_model)


if __name__ == "__main__":
    main()
