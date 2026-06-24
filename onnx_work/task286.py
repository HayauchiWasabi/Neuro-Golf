# scripts/task286.py
#
# run:
#   python scripts/task286.py
#
# Output ONNX files only:
#   outputs/gpt_workbench/task286/*.onnx
#
# This script starts from current best:
#   outputs/gpt_workbench/task286/task286_v19_c02_tail_seed_label_u8.onnx
#
# If that file does not exist, it falls back to:
#   outputs/gpt_workbench/task286/task286_v18_c01_full_label_active_then_slice.onnx

from pathlib import Path
import copy
import re
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

from task286_builder import (
    save_model,
    try_check,
    remove_dead_nodes,
    remove_unused_initializers,
    add_initializer_if_missing,
)


OUT_DIR = Path("outputs/gpt_workbench/task286")

BEST_PATHS = [
    OUT_DIR / "task286_v19_c02_tail_seed_label_u8.onnx",
    OUT_DIR / "task286_v18_c01_full_label_active_then_slice.onnx",
    OUT_DIR / "task286_v16_c03_notbg_and_wall_from_label.onnx",
]


def load_best() -> onnx.ModelProto:
    for p in BEST_PATHS:
        if p.exists():
            print(f"[load] {p}")
            return onnx.load(str(p))
    raise FileNotFoundError(
        "Could not find current best ONNX. Expected one of:\n"
        + "\n".join(str(p) for p in BEST_PATHS)
    )


def finalize(model: onnx.ModelProto) -> onnx.ModelProto:
    m = remove_dead_nodes(model)
    m = remove_unused_initializers(m)
    try_check(m)
    return m


def add_false_mask25(model: onnx.ModelProto) -> None:
    add_initializer_if_missing(
        model,
        numpy_helper.from_array(
            np.zeros((1, 1, 25, 25), dtype=np.bool_),
            name="false_mask25_bool",
        ),
    )


def add_eight_u8(model: onnx.ModelProto) -> None:
    add_initializer_if_missing(
        model,
        numpy_helper.from_array(np.array([8], dtype=np.uint8), name="eight_u8"),
    )


def replace_notwall_with_label_less8(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Candidate c01.

    Current v19 best:
      label_u8 == 8 -> ch8_b
      Not(ch8_b) -> notwall_b
      And(active_b, notwall_b) -> open_b

    Candidate:
      Less(label_u8, 8) -> notwall_b
      And(active_b, notwall_b) -> open_b

    This removes ch8_b and the Not output.

    Risk:
      If valid non-wall label 9 exists, Less(label, 8) incorrectly treats it as wall.
      If task only uses 0..8 with 8 as wall, this should pass.
    """
    m = copy.deepcopy(model)
    add_eight_u8(m)

    new_nodes = []
    removed_ch8_equal = False
    replaced_notwall = False

    for node in m.graph.node:
        outputs = set(node.output)

        # Remove Equal(label_u8, 8) -> ch8_b
        if "ch8_b" in outputs:
            removed_ch8_equal = True
            continue

        # Replace Not(ch8_b) -> notwall_b with Less(label_u8, 8) -> notwall_b
        if "notwall_b" in outputs:
            new_nodes.append(
                helper.make_node(
                    "Less",
                    inputs=["label_u8", "eight_u8"],
                    outputs=["notwall_b"],
                    name="less_label_u8_than_8_notwall",
                )
            )
            replaced_notwall = True
            continue

        new_nodes.append(node)

    if not removed_ch8_equal:
        raise RuntimeError("Could not remove ch8_b producer.")
    if not replaced_notwall:
        raise RuntimeError("Could not replace notwall_b producer.")

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    for node in m.graph.node:
        if "ch8_b" in node.input:
            raise RuntimeError(f"ch8_b still used by {node.name} / {node.op_type}")

    return finalize(m)


def replace_open_and_notwall_with_where_full_false(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Candidate c02 helper.

    Current:
      open_b = And(active_b, notwall_b)

    Candidate:
      open_b = Where(ch8_b, false_mask25_bool, active_b)

    In earlier v19, scalar false caused runtime error.
    Here we use full [1,1,25,25] false initializer to avoid scalar bool broadcast issue.

    This should remove notwall_b.
    """
    m = copy.deepcopy(model)
    add_false_mask25(m)

    new_nodes = []
    removed_notwall = False
    replaced_open = False

    for node in m.graph.node:
        outputs = set(node.output)

        if "notwall_b" in outputs:
            removed_notwall = True
            continue

        if "open_b" in outputs and node.op_type == "And":
            new_nodes.append(
                helper.make_node(
                    "Where",
                    inputs=["ch8_b", "false_mask25_bool", "active_b"],
                    outputs=["open_b"],
                    name="where_wall_false_else_active_open_b",
                )
            )
            replaced_open = True
            continue

        new_nodes.append(node)

    if not removed_notwall:
        raise RuntimeError("Could not remove notwall_b producer.")
    if not replaced_open:
        raise RuntimeError("Could not replace open_b producer.")

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    for node in m.graph.node:
        if "notwall_b" in node.input:
            raise RuntimeError(f"notwall_b still used by {node.name} / {node.op_type}")

    return finalize(m)


def replace_uint8_floodfill_with_bool_floodfill(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Candidate c03.

    Current:
      seed_b -> Cast -> seed_u8
      MaxPool uint8
      reach_i = Where(open_b, pool_i, zero_u8_ff)
      reach_58 -> Cast -> reach_bool

    Candidate:
      MaxPool bool directly from seed_b
      reach_i = Where(open_b, pool_i, false_mask25_bool)
      reach_58 used directly as reach_bool

    Risk:
      ONNXRuntime may not support MaxPool(bool).
      If it does, this can remove seed_u8 and reach_bool cast path.
    """
    m = copy.deepcopy(model)
    add_false_mask25(m)

    new_nodes = []

    for node in m.graph.node:
        outputs = set(node.output)

        # Remove Cast(seed_b -> seed_u8)
        if "seed_u8" in outputs and node.op_type == "Cast":
            continue

        # Remove Cast(reach_58 -> reach_bool); downstream will use reach_58 directly.
        if "reach_bool" in outputs and node.op_type == "Cast":
            continue

        node = copy.deepcopy(node)

        # MaxPool first input: seed_u8 -> seed_b
        if node.op_type == "MaxPool":
            for i, inp in enumerate(node.input):
                if inp == "seed_u8":
                    node.input[i] = "seed_b"

        # Where flood-fill: zero_u8_ff -> false_mask25_bool
        if node.op_type == "Where" and len(node.output) == 1:
            if re.fullmatch(r"reach_\d+", node.output[0]):
                for i, inp in enumerate(node.input):
                    if inp == "zero_u8_ff":
                        node.input[i] = "false_mask25_bool"

        # downstream reach_bool input -> reach_58
        for i, inp in enumerate(node.input):
            if inp == "reach_bool":
                node.input[i] = "reach_58"

        new_nodes.append(node)

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    for node in m.graph.node:
        if "seed_u8" in node.input or "reach_bool" in node.input:
            raise RuntimeError(
                f"old floodfill tensor still used by {node.name} / {node.op_type}"
            )

    return finalize(m)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best = load_best()

    # c00: current best copy.
    save_model(
        finalize(best),
        OUT_DIR / "task286_v20_c00_v19_best.onnx",
    )

    # c01: notwall from Less(label, 8).
    try:
        m = replace_notwall_with_label_less8(best)
        save_model(
            m,
            OUT_DIR / "task286_v20_c01_notwall_label_less8.onnx",
        )
    except Exception as e:
        print(f"[skip] c01 notwall_label_less8 failed: {e}")

    # c02: use full false mask for Where open_b to avoid scalar bool runtime error.
    try:
        m = replace_open_and_notwall_with_where_full_false(best)
        save_model(
            m,
            OUT_DIR / "task286_v20_c02_open_where_full_false_mask.onnx",
        )
    except Exception as e:
        print(f"[skip] c02 open_where_full_false_mask failed: {e}")

    # c03: bool flood-fill candidate.
    try:
        m = replace_uint8_floodfill_with_bool_floodfill(best)
        save_model(
            m,
            OUT_DIR / "task286_v20_c03_bool_floodfill.onnx",
        )
    except Exception as e:
        print(f"[skip] c03 bool_floodfill failed: {e}")

    # c04: combine c01 + c03.
    try:
        m = replace_notwall_with_label_less8(best)
        m = replace_uint8_floodfill_with_bool_floodfill(m)
        save_model(
            m,
            OUT_DIR / "task286_v20_c04_less8_bool_floodfill.onnx",
        )
    except Exception as e:
        print(f"[skip] c04 less8_bool_floodfill failed: {e}")


if __name__ == "__main__":
    main()