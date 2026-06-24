# scripts/task286_builder.py

from __future__ import annotations

from pathlib import Path
import copy
import re
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def save_model(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def try_check(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model)


def add_initializer_if_missing(model: onnx.ModelProto, tensor: onnx.TensorProto) -> None:
    names = {init.name for init in model.graph.initializer}
    if tensor.name not in names:
        model.graph.initializer.append(tensor)


def get_initializer_array(model: onnx.ModelProto, name: str) -> np.ndarray:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    raise KeyError(f"initializer not found: {name}")


def remove_unused_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)

    used = set()
    for node in m.graph.node:
        used.update(node.input)

    kept = [init for init in m.graph.initializer if init.name in used]

    del m.graph.initializer[:]
    m.graph.initializer.extend(kept)

    return m


def remove_dead_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)
    graph_outputs = {o.name for o in m.graph.output}

    changed = True
    while changed:
        changed = False

        used = set()
        for node in m.graph.node:
            used.update(node.input)
        used.update(graph_outputs)

        new_nodes = []
        for node in m.graph.node:
            if any(out in used for out in node.output):
                new_nodes.append(node)
            else:
                changed = True

        del m.graph.node[:]
        m.graph.node.extend(new_nodes)

    return m


def find_insert_index_after(model: onnx.ModelProto, output_names: list[str]) -> int:
    idx = 0
    target = set(output_names)

    for i, node in enumerate(model.graph.node):
        if any(out in target for out in node.output):
            idx = max(idx, i + 1)

    return idx


def add_task286_common_initializers(model: onnx.ModelProto) -> None:
    """
    Common constants for the current best task286 rewrite.

    Params may increase slightly.
    The goal is reducing intermediate memory.
    """
    # QLinearConv constants.
    add_initializer_if_missing(
        model,
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="q_scale_1"),
    )
    add_initializer_if_missing(
        model,
        numpy_helper.from_array(np.array([0], dtype=np.uint8), name="q_zp_u8_0"),
    )

    # label_w_u8: [1,10,1,1] = 0..9
    label_w_u8 = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    add_initializer_if_missing(
        model,
        numpy_helper.from_array(label_w_u8, name="label_w_u8"),
    )

    # uint8 constants.
    add_initializer_if_missing(
        model,
        numpy_helper.from_array(np.array([0], dtype=np.uint8), name="zero_u8_ff"),
    )
    add_initializer_if_missing(
        model,
        numpy_helper.from_array(np.array([0], dtype=np.uint8), name="zero_u8_tail"),
    )
    add_initializer_if_missing(
        model,
        numpy_helper.from_array(np.array([10], dtype=np.uint8), name="ten_u8_tail"),
    )

    # bool checkerboard masks from existing fp16 masks.
    even = get_initializer_array(model, "even_mask")
    odd = get_initializer_array(model, "odd_mask")

    even_b = (even.astype(np.float32) > 0.5).astype(np.bool_)
    odd_b = (odd.astype(np.float32) > 0.5).astype(np.bool_)

    add_initializer_if_missing(
        model,
        numpy_helper.from_array(even_b, name="even_mask_bool"),
    )
    add_initializer_if_missing(
        model,
        numpy_helper.from_array(odd_b, name="odd_mask_bool"),
    )


def replace_frontend_with_u8(
    model: onnx.ModelProto,
    *,
    create_open_u8: bool = False,
    active_from_bool_reduce: bool = False,
) -> onnx.ModelProto:
    """
    Replace original fp16 frontend with uint8/bool frontend.

    Original path removed:
      x16, active_f, label,
      ch8_f, ch0_f, notwall_f, open_f, notbg_f, seed_f,
      open_u8, seed_u8

    New path:
      x_bool -> x_u8
      x_u8 -> active_u8 -> active_b
      x_u8 + QLinearConv -> label_u8
      x_bool Gather ch8/ch0 -> bool open/seed
      seed_b -> seed_u8

    create_open_u8=False is best for current Where-based flood-fill,
    because open_u8 is unused after Min is replaced with Where(open_b,...).
    """
    m = copy.deepcopy(model)
    add_task286_common_initializers(m)

    remove_outputs = {
        "x16",
        "active_f",
        "label",
        "ch8_f",
        "ch0_f",
        "notwall_f",
        "open_f",
        "notbg_f",
        "seed_f",
        "open_u8",
        "seed_u8",
    }

    kept_nodes = []
    for node in m.graph.node:
        if any(out in remove_outputs for out in node.output):
            continue
        kept_nodes.append(node)

    del m.graph.node[:]
    m.graph.node.extend(kept_nodes)

    insert_idx = find_insert_index_after(m, ["x_bool"])

    frontend_nodes = [
        helper.make_node(
            "Cast",
            inputs=["x_bool"],
            outputs=["x_u8"],
            to=TensorProto.UINT8,
            name="cast_x_bool_to_u8",
        ),
    ]

    if active_from_bool_reduce:
        frontend_nodes.append(
            helper.make_node(
                "ReduceMax",
                inputs=["x_bool"],
                outputs=["active_b"],
                axes=[1],
                keepdims=1,
                name="reduce_active_bool",
            )
        )
    else:
        frontend_nodes.extend(
            [
                helper.make_node(
                    "ReduceMax",
                    inputs=["x_u8"],
                    outputs=["active_u8"],
                    axes=[1],
                    keepdims=1,
                    name="reduce_active_u8",
                ),
                helper.make_node(
                    "Cast",
                    inputs=["active_u8"],
                    outputs=["active_b"],
                    to=TensorProto.BOOL,
                    name="cast_active_u8_to_bool",
                ),
            ]
        )

    frontend_nodes.extend(
        [
            helper.make_node(
                "QLinearConv",
                inputs=[
                    "x_u8",
                    "q_scale_1",
                    "q_zp_u8_0",
                    "label_w_u8",
                    "q_scale_1",
                    "q_zp_u8_0",
                    "q_scale_1",
                    "q_zp_u8_0",
                ],
                outputs=["label_u8"],
                kernel_shape=[1, 1],
                name="qconv_label_u8",
            ),
            helper.make_node(
                "Gather",
                inputs=["x_bool", "idx8"],
                outputs=["ch8_b"],
                axis=1,
                name="gather_ch8_bool",
            ),
            helper.make_node(
                "Gather",
                inputs=["x_bool", "idx0"],
                outputs=["ch0_b"],
                axis=1,
                name="gather_ch0_bool",
            ),
            helper.make_node(
                "Not",
                inputs=["ch8_b"],
                outputs=["notwall_b"],
                name="not_ch8_wall",
            ),
            helper.make_node(
                "And",
                inputs=["active_b", "notwall_b"],
                outputs=["open_b"],
                name="and_active_notwall",
            ),
            helper.make_node(
                "Not",
                inputs=["ch0_b"],
                outputs=["notbg_b"],
                name="not_ch0_bg",
            ),
            helper.make_node(
                "And",
                inputs=["open_b", "notbg_b"],
                outputs=["seed_b"],
                name="and_open_notbg",
            ),
        ]
    )

    if create_open_u8:
        frontend_nodes.append(
            helper.make_node(
                "Cast",
                inputs=["open_b"],
                outputs=["open_u8"],
                to=TensorProto.UINT8,
                name="cast_open_bool_to_u8",
            )
        )

    frontend_nodes.append(
        helper.make_node(
            "Cast",
            inputs=["seed_b"],
            outputs=["seed_u8"],
            to=TensorProto.UINT8,
            name="cast_seed_bool_to_u8",
        )
    )

    nodes = list(m.graph.node)
    nodes[insert_idx:insert_idx] = frontend_nodes

    del m.graph.node[:]
    m.graph.node.extend(nodes)

    return m


def replace_uint8_min_with_where_using_open_b(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Replace:
      reach_i = Min(pool_i, open_u8)

    with:
      reach_i = Where(open_b, pool_i, zero_u8_ff)

    This keeps uint8 flood-fill tensors and avoids unsupported uint8 Min.
    """
    m = copy.deepcopy(model)
    add_task286_common_initializers(m)

    for node in m.graph.node:
        if node.op_type != "Min":
            continue
        if len(node.output) != 1:
            continue

        out_name = node.output[0]
        if not re.fullmatch(r"reach_\d+", out_name):
            continue

        pool_input = None
        for inp in node.input:
            if re.fullmatch(r"pool_\d+", inp):
                pool_input = inp
                break

        if pool_input is None:
            continue

        node.op_type = "Where"
        node.domain = ""
        del node.input[:]
        node.input.extend(["open_b", pool_input, "zero_u8_ff"])
        del node.attribute[:]

    return m


def replace_tail_with_uint8_label_map(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Replace original fp16 tail with uint8 label-map tail.

    Assumes:
      label_u8 exists
      seed_b exists
      active_b exists
      reach_58 exists
    """
    m = copy.deepcopy(model)
    add_task286_common_initializers(m)

    remove_outputs = {
        "seed_label",
        "seed_even",
        "even_color",
        "seed_odd",
        "odd_color",
        "fill_even",
        "fill_odd",
        "fill_label",
        "reach_bool",
        "out_label_f",
        "inactive_f",
        "inactive_label",
        "out_label_masked",
        "eq_bool",
        "out25_u8",
        "output",
    }

    kept_nodes = []
    for node in m.graph.node:
        if any(out in remove_outputs for out in node.output):
            continue
        kept_nodes.append(node)

    del m.graph.node[:]
    m.graph.node.extend(kept_nodes)

    tail_nodes = [
        helper.make_node(
            "And",
            inputs=["seed_b", "even_mask_bool"],
            outputs=["even_seed_b"],
            name="and_seed_even_mask",
        ),
        helper.make_node(
            "Where",
            inputs=["even_seed_b", "label_u8", "zero_u8_tail"],
            outputs=["seed_even_u8"],
            name="where_seed_even_label_u8",
        ),
        helper.make_node(
            "ReduceMax",
            inputs=["seed_even_u8"],
            outputs=["even_color_u8"],
            axes=[2, 3],
            keepdims=1,
            name="reduce_even_color_u8",
        ),
        helper.make_node(
            "And",
            inputs=["seed_b", "odd_mask_bool"],
            outputs=["odd_seed_b"],
            name="and_seed_odd_mask",
        ),
        helper.make_node(
            "Where",
            inputs=["odd_seed_b", "label_u8", "zero_u8_tail"],
            outputs=["seed_odd_u8"],
            name="where_seed_odd_label_u8",
        ),
        helper.make_node(
            "ReduceMax",
            inputs=["seed_odd_u8"],
            outputs=["odd_color_u8"],
            axes=[2, 3],
            keepdims=1,
            name="reduce_odd_color_u8",
        ),
        helper.make_node(
            "Where",
            inputs=["even_mask_bool", "even_color_u8", "odd_color_u8"],
            outputs=["fill_label_u8"],
            name="where_even_odd_fill_u8",
        ),
        helper.make_node(
            "Cast",
            inputs=["reach_58"],
            outputs=["reach_bool"],
            to=TensorProto.BOOL,
            name="cast_reach_to_bool",
        ),
        helper.make_node(
            "Where",
            inputs=["reach_bool", "fill_label_u8", "label_u8"],
            outputs=["out_label_u8"],
            name="where_reach_fill_or_label_u8",
        ),
        helper.make_node(
            "Where",
            inputs=["active_b", "out_label_u8", "ten_u8_tail"],
            outputs=["out_label_masked_u8"],
            name="where_active_or_ten_u8",
        ),
        helper.make_node(
            "Equal",
            inputs=["label_w_u8", "out_label_masked_u8"],
            outputs=["eq_bool"],
            name="equal_label_u8",
        ),
        helper.make_node(
            "Cast",
            inputs=["eq_bool"],
            outputs=["out25_u8"],
            to=TensorProto.UINT8,
            name="cast_eq_to_u8",
        ),
        helper.make_node(
            "Pad",
            inputs=["out25_u8", "pad_to_30"],
            outputs=["output"],
            mode="constant",
            name="pad_out25_to_30",
        ),
    ]

    m.graph.node.extend(tail_nodes)

    return m


def force_output_float32_if_needed(model: onnx.ModelProto) -> onnx.ModelProto:
    m = copy.deepcopy(model)

    if len(m.graph.output) != 1:
        raise RuntimeError("Expected one graph output.")

    graph_out = m.graph.output[0]
    if graph_out.type.tensor_type.elem_type == TensorProto.FLOAT:
        return m

    old_out = graph_out.name
    internal_out = old_out + "_before_final_float_cast"

    renamed = False
    for node in m.graph.node:
        for j, out_name in enumerate(node.output):
            if out_name == old_out:
                node.output[j] = internal_out
                renamed = True

    if not renamed:
        raise RuntimeError(f"Could not find producer for graph output: {old_out}")

    cast_node = helper.make_node(
        "Cast",
        inputs=[internal_out],
        outputs=[old_out],
        to=TensorProto.FLOAT,
        name="final_cast_to_float",
    )

    m.graph.node.append(cast_node)
    graph_out.type.tensor_type.elem_type = TensorProto.FLOAT

    return m


def build_task286_current_best(
    base: onnx.ModelProto,
    *,
    create_open_u8: bool = False,
    active_from_bool_reduce: bool = False,
    float_output: bool = False,
) -> onnx.ModelProto:
    """
    Current best family.

    Known good:
      create_open_u8=False
      active_from_bool_reduce=False
      float_output=False

    Known error in current runtime:
      active_from_bool_reduce=True
    """
    m = replace_frontend_with_u8(
        base,
        create_open_u8=create_open_u8,
        active_from_bool_reduce=active_from_bool_reduce,
    )
    m = replace_uint8_min_with_where_using_open_b(m)
    m = replace_tail_with_uint8_label_map(m)
    m = remove_dead_nodes(m)

    if float_output:
        m = force_output_float32_if_needed(m)

    m = remove_dead_nodes(m)
    m = remove_unused_initializers(m)
    try_check(m)

    return m