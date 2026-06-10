import numpy as np
import onnx
from onnx import TensorProto, helper

from neurogolf_common import compact_model, const_node, run_task, slice_node, tensor

TASK_NUM = 9


def add_sum(nodes, terms, name):
    acc = terms[0]
    for i, term in enumerate(terms[1:], start=1):
        nodes.append(helper.make_node("Add", [acc, term], [f"{name}_{i}"]))
        acc = f"{name}_{i}"
    return acc


def shift_10(nodes, output, data, dr, dc):
    r0 = max(0, -dr)
    r1 = min(10, 10 - dr)
    c0 = max(0, -dc)
    c1 = min(10, 10 - dc)
    top = max(0, dr)
    left = max(0, dc)
    bottom = 10 - top - (r1 - r0)
    right = 10 - left - (c1 - c0)
    nodes += slice_node(f"{output}_sl", data, [0, 0, r0, c0], [1, 9, r1, c1], [0, 1, 2, 3])
    nodes.append(helper.make_node("Pad", [f"{output}_sl"], [output], mode="constant", pads=[0, 0, top, left, 0, 0, bottom, right], value=0.0))
    return output


def build_model():
    nodes = []
    initializers = [
        tensor("cell_w", np.ones((9, 1, 2, 2), dtype=np.float32)),
        tensor("expand_w", np.ones((9, 1, 2, 2), dtype=np.float32)),
    ]
    nodes.append(const_node("zero_scalar", np.array([0.0], dtype=np.float32)))

    nodes += slice_node("black", "input", [0, 0, 0, 0], [1, 1, 30, 30], [0, 1, 2, 3])
    nodes += slice_node("objects", "input", [0, 1, 0, 0], [1, 10, 30, 30], [0, 1, 2, 3])
    nodes.append(helper.make_node("Conv", ["objects", "cell_w"], ["cell_sum"], group=9, strides=[3, 3]))
    nodes.append(helper.make_node("Clip", ["cell_sum"], ["present"], min=0.0, max=1.0))

    left_terms = ["present"]
    right_terms = ["present"]
    up_terms = ["present"]
    down_terms = ["present"]
    for step in range(1, 10):
        left_terms.append(shift_10(nodes, f"l{step}", "present", 0, step))
        right_terms.append(shift_10(nodes, f"r{step}", "present", 0, -step))
        up_terms.append(shift_10(nodes, f"u{step}", "present", step, 0))
        down_terms.append(shift_10(nodes, f"d{step}", "present", -step, 0))

    nodes.append(helper.make_node("Clip", [add_sum(nodes, left_terms, "ls")], ["left"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Clip", [add_sum(nodes, right_terms, "rs")], ["right"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Clip", [add_sum(nodes, up_terms, "us")], ["up"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Clip", [add_sum(nodes, down_terms, "ds")], ["down"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Mul", ["left", "right"], ["hgate"]))
    nodes.append(helper.make_node("Mul", ["up", "down"], ["vgate"]))
    nodes.append(helper.make_node("Add", ["hgate", "vgate"], ["gate_raw"]))
    nodes.append(helper.make_node("Clip", ["gate_raw"], ["gate"], min=0.0, max=1.0))

    nodes.append(helper.make_node("ConvTranspose", ["gate", "expand_w"], ["fill"], group=9, strides=[3, 3]))
    nodes.append(helper.make_node("Pad", ["fill"], ["fill30"], mode="constant", pads=[0, 0, 0, 0, 0, 0, 1, 1], value=0.0))
    nodes.append(helper.make_node("Add", ["objects", "fill30"], ["out_obj_raw"]))
    nodes.append(helper.make_node("Clip", ["out_obj_raw"], ["out_obj"], min=0.0, max=1.0))

    obj_channels = []
    for color in range(1, 10):
        nodes += slice_node(f"obj{color}", "out_obj", [0, color - 1, 0, 0], [1, color, 30, 30], [0, 1, 2, 3])
        obj_channels.append(f"obj{color}")
    nonzero = add_sum(nodes, obj_channels, "nz")
    nodes.append(helper.make_node("Clip", [nonzero], ["nz_clip"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Sub", ["black", "nz_clip"], ["out_ch0_raw"]))
    nodes.append(helper.make_node("Clip", ["out_ch0_raw"], ["out_ch0"], min=0.0, max=1.0))
    nodes.append(helper.make_node("Concat", ["out_ch0"] + obj_channels, ["output"], axis=1))

    graph = helper.make_graph(
        nodes,
        "task009_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    compact_model(model, keep=("input", "output", "cell_w", "expand_w"))
    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    run_task(TASK_NUM, build_model)


if __name__ == "__main__":
    main()
