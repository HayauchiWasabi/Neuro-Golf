from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from neurogolf_common import load_task, verify_onnx  # noqa: E402


SRC = ROOT / "Sample" / "submission" / "task173.onnx"
OUT_DIR = ROOT / "outputs" / "task173_experiments"


def set_initializer(model, name, array):
    for idx, init in enumerate(model.graph.initializer):
        if init.name == name:
            model.graph.initializer[idx].CopyFrom(numpy_helper.from_array(np.asarray(array), name=name))
            return
    raise KeyError(name)


def shrink_region(n):
    model = onnx.load(SRC)
    del model.graph.value_info[:]
    area = n * n
    set_initializer(model, "slice_bg_ends", np.array([1, n, n], dtype=np.int64))
    set_initializer(model, "slice_x_ends", np.array([10, n, n], dtype=np.int64))
    set_initializer(model, "shape_9_625", np.array([9, area], dtype=np.int64))
    set_initializer(model, "shape_1_1_25_25", np.array([1, 1, n, n], dtype=np.int64))
    set_initializer(model, "pad_to_30", np.array([0, 0, 0, 0, 0, 0, 5, 5], dtype=np.int64))
    return model


def qlinear_label():
    model = onnx.load(SRC)
    del model.graph.value_info[:]
    nodes = []
    for node in model.graph.node:
        if node.output and node.output[0] == "label":
            nodes.append(
                helper.make_node(
                    "QLinearMatMul",
                    [
                        "flat_t_u8",
                        "ql_scale",
                        "_pad_uint8_zero",
                        "color_values_2d_u8",
                        "ql_scale",
                        "_pad_uint8_zero",
                        "ql_scale",
                        "_pad_uint8_zero",
                    ],
                    ["label_flat_u8"],
                    name="label_qmm",
                )
            )
            nodes.append(helper.make_node("Reshape", ["label_flat_u8", "shape_1_1_25_25"], ["label_u8"], name="label_reshape"))
            nodes.append(helper.make_node("Cast", ["label_u8"], ["label"], name="label_cast", to=onnx.TensorProto.FLOAT16))
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def add_init(model, name, array, dtype=None):
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(array), name=name))


def replace_h_detect_with_bool():
    model = onnx.load(SRC)
    del model.graph.value_info[:]
    add_init(model, "h_left_starts", [0, 0, 0, 0], np.int64)
    add_init(model, "h_left_ends", [1, 9, 25, 23], np.int64)
    add_init(model, "h_right_starts", [0, 0, 0, 2], np.int64)
    add_init(model, "h_right_ends", [1, 9, 25, 25], np.int64)
    add_init(model, "h_axes", [0, 1, 2, 3], np.int64)
    add_init(model, "h_steps", [1, 1, 1, 1], np.int64)
    add_init(model, "h_pad_lr", [0, 0, 0, 1, 0, 0, 0, 1], np.int64)
    add_init(model, "h_pad_false", np.array(False, dtype=np.bool_))

    nodes = []
    inserted = False
    for node in model.graph.node:
        if node.output and node.output[0] == "h_detected_sum":
            if not inserted:
                nodes.extend(
                    [
                        helper.make_node("Cast", ["x9_u8"], ["h_x_bool"], name="h_x_bool", to=onnx.TensorProto.BOOL),
                        helper.make_node("Slice", ["h_x_bool", "h_left_starts", "h_left_ends", "h_axes", "h_steps"], ["h_left_bool"], name="h_left"),
                        helper.make_node("Slice", ["h_x_bool", "h_right_starts", "h_right_ends", "h_axes", "h_steps"], ["h_right_bool"], name="h_right"),
                        helper.make_node("And", ["h_left_bool", "h_right_bool"], ["h_mid_bool"], name="h_and"),
                        helper.make_node("Pad", ["h_mid_bool", "h_pad_lr", "h_pad_false"], ["h_full_bool"], name="h_pad"),
                    ]
                )
                inserted = True
            continue
        if node.output and node.output[0] == "h_full_bool":
            continue
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def remove_initializers(model, names):
    names = set(names)
    kept = [init for init in model.graph.initializer if init.name not in names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def replace_h_detect_with_u8():
    model = onnx.load(SRC)
    for opset in model.opset_import:
        if opset.domain in {"", "ai.onnx"}:
            opset.version = max(opset.version, 18)
    del model.graph.value_info[:]
    remove_initializers(model, ["h_detect_weight"])
    add_init(model, "h_left_starts", [0, 0, 0, 0], np.int64)
    add_init(model, "h_left_ends", [1, 9, 25, 23], np.int64)
    add_init(model, "h_right_starts", [0, 0, 0, 2], np.int64)
    add_init(model, "h_right_ends", [1, 9, 25, 25], np.int64)
    add_init(model, "h_axes", [0, 1, 2, 3], np.int64)
    add_init(model, "h_steps", [1, 1, 1, 1], np.int64)
    add_init(model, "h_pad_lr", [0, 0, 0, 1, 0, 0, 0, 1], np.int64)
    add_init(model, "h_pad_zero_u8", np.array(0, dtype=np.uint8))

    nodes = []
    inserted = False
    for node in model.graph.node:
        if node.output and node.output[0] == "h_detected_sum":
            if not inserted:
                nodes.extend(
                    [
                        helper.make_node("Slice", ["x9_u8", "h_left_starts", "h_left_ends", "h_axes", "h_steps"], ["h_left_u8"], name="h_left"),
                        helper.make_node("Slice", ["x9_u8", "h_right_starts", "h_right_ends", "h_axes", "h_steps"], ["h_right_u8"], name="h_right"),
                        helper.make_node("BitwiseAnd", ["h_left_u8", "h_right_u8"], ["h_mid_u8"], name="h_bitand"),
                        helper.make_node("Pad", ["h_mid_u8", "h_pad_lr", "h_pad_zero_u8"], ["h_full"], name="h_pad"),
                    ]
                )
                inserted = True
            continue
        if node.output and node.output[0] in {"h_full_bool", "h_full"}:
            continue
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def replace_h_detect_with_where_u8():
    model = onnx.load(SRC)
    del model.graph.value_info[:]
    remove_initializers(model, ["h_detect_weight"])
    add_init(model, "h_left_starts", [0, 0, 0, 0], np.int64)
    add_init(model, "h_left_ends", [1, 9, 25, 23], np.int64)
    add_init(model, "h_right_starts", [0, 0, 0, 2], np.int64)
    add_init(model, "h_right_ends", [1, 9, 25, 25], np.int64)
    add_init(model, "h_axes", [0, 1, 2, 3], np.int64)
    add_init(model, "h_steps", [1, 1, 1, 1], np.int64)
    add_init(model, "h_pad_lr", [0, 0, 0, 1, 0, 0, 0, 1], np.int64)
    add_init(model, "h_zero_u8", np.array(0, dtype=np.uint8))

    nodes = []
    inserted = False
    for node in model.graph.node:
        if node.output and node.output[0] == "h_detected_sum":
            if not inserted:
                nodes.extend(
                    [
                        helper.make_node("Slice", ["x9_u8", "h_left_starts", "h_left_ends", "h_axes", "h_steps"], ["h_left_u8"], name="h_left"),
                        helper.make_node("Slice", ["x9_u8", "h_right_starts", "h_right_ends", "h_axes", "h_steps"], ["h_right_u8"], name="h_right"),
                        helper.make_node("Greater", ["h_left_u8", "h_zero_u8"], ["h_left_nonzero"], name="h_left_nonzero"),
                        helper.make_node("Where", ["h_left_nonzero", "h_right_u8", "h_zero_u8"], ["h_mid_u8"], name="h_where"),
                        helper.make_node("Pad", ["h_mid_u8", "h_pad_lr", "h_zero_u8"], ["h_full"], name="h_pad"),
                    ]
                )
                inserted = True
            continue
        if node.output and node.output[0] in {"h_full_bool", "h_full"}:
            continue
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def strip_value_info():
    model = onnx.load(SRC)
    del model.graph.value_info[:]
    return model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(173)
    for n in [24, 23, 22, 21, 20]:
        path = OUT_DIR / f"task173_region{n}.onnx"
        onnx.save(shrink_region(n), path)
        try:
            passed = verify_onnx(path, task)
            print(f"region {n}: ok {passed} size={path.stat().st_size}")
        except Exception as exc:
            print(f"region {n}: FAIL {type(exc).__name__}: {exc}")
    path = OUT_DIR / "task173_qlinear_label.onnx"
    onnx.save(qlinear_label(), path)
    try:
        passed = verify_onnx(path, task)
        print(f"qlinear_label: ok {passed} size={path.stat().st_size}")
    except Exception as exc:
        print(f"qlinear_label: FAIL {type(exc).__name__}: {exc}")
    path = OUT_DIR / "task173_h_bool.onnx"
    onnx.save(replace_h_detect_with_bool(), path)
    try:
        passed = verify_onnx(path, task)
        print(f"h_bool: ok {passed} size={path.stat().st_size}")
    except Exception as exc:
        print(f"h_bool: FAIL {type(exc).__name__}: {exc}")
    path = OUT_DIR / "task173_h_u8.onnx"
    onnx.save(replace_h_detect_with_u8(), path)
    try:
        passed = verify_onnx(path, task)
        print(f"h_u8: ok {passed} size={path.stat().st_size}")
    except Exception as exc:
        print(f"h_u8: FAIL {type(exc).__name__}: {exc}")
    path = OUT_DIR / "task173_no_value_info.onnx"
    onnx.save(strip_value_info(), path)
    try:
        passed = verify_onnx(path, task)
        print(f"no_value_info: ok {passed} size={path.stat().st_size}")
    except Exception as exc:
        print(f"no_value_info: FAIL {type(exc).__name__}: {exc}")
    path = OUT_DIR / "task173_h_where_u8.onnx"
    onnx.save(replace_h_detect_with_where_u8(), path)
    try:
        passed = verify_onnx(path, task)
        print(f"h_where_u8: ok {passed} size={path.stat().st_size}")
    except Exception as exc:
        print(f"h_where_u8: FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
