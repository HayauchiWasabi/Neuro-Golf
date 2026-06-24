# scripts/gen_task101_gpt_workbench_v13_memory_from107.py
# Usage:
#   python scripts/gen_task101_gpt_workbench_v13_memory_from107.py
#
# Input:
#   outputs/gpt_workbench/task101/107_77_max_s123.onnx
#
# Output:
#   outputs/gpt_workbench/task101/*.onnx only

from pathlib import Path
import copy
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


BASE = Path("outputs/gpt_workbench/task101/107_77_max_s123.onnx")
OUT_DIR = Path("outputs/gpt_workbench/task101")


def init_i64(name, vals, shape=None):
    arr = np.asarray(vals, dtype=np.int64)
    if shape is not None:
        arr = arr.reshape(shape)
    return numpy_helper.from_array(arr, name=name)


def init_f16(name, vals, shape=None):
    arr = np.asarray(vals, dtype=np.float16)
    if shape is not None:
        arr = arr.reshape(shape)
    return numpy_helper.from_array(arr, name=name)


def remove_nodes_by_name(model, names):
    names = set(names)
    kept = [n for n in model.graph.node if n.name not in names]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def remove_nodes_producing(model, outputs):
    outputs = set(outputs)
    kept = []
    for n in model.graph.node:
        if any(o in outputs for o in n.output):
            continue
        kept.append(n)
    del model.graph.node[:]
    model.graph.node.extend(kept)


def replace_input_name(model, old, new):
    for n in model.graph.node:
        for i, inp in enumerate(n.input):
            if inp == old:
                n.input[i] = new


def used_tensor_names(model):
    used = set()
    for n in model.graph.node:
        used.update(i for i in n.input if i)
    for o in model.graph.output:
        used.add(o.name)
    return used


def prune_initializers(model):
    used = used_tensor_names(model)
    kept = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def prune_dead_nodes_fixed(model):
    wanted = {o.name for o in model.graph.output}
    keep = set()

    changed = True
    while changed:
        changed = False
        for idx, n in enumerate(model.graph.node):
            if idx in keep:
                continue
            if any(o in wanted for o in n.output):
                keep.add(idx)
                wanted.update(i for i in n.input if i)
                changed = True

    new_nodes = [n for idx, n in enumerate(model.graph.node) if idx in keep]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    prune_initializers(model)


def topo_sort(model):
    available = {x.name for x in model.graph.input}
    available.update(x.name for x in model.graph.initializer)

    remaining = list(model.graph.node)
    sorted_nodes = []

    while remaining:
        progressed = False
        nxt = []

        for n in remaining:
            if all((i == "") or (i in available) for i in n.input):
                sorted_nodes.append(n)
                available.update(n.output)
                progressed = True
            else:
                nxt.append(n)

        if not progressed:
            print("TOPO SORT FAILED")
            for n in nxt[:20]:
                miss = [i for i in n.input if i and i not in available]
                print(n.name, n.op_type, miss)
            return model

        remaining = nxt

    del model.graph.node[:]
    model.graph.node.extend(sorted_nodes)
    return model


def clear_metadata(model):
    model.doc_string = ""
    model.graph.doc_string = ""
    for n in model.graph.node:
        n.doc_string = ""
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        vi.doc_string = ""


def remove_kernel_shape_attrs(model):
    for n in model.graph.node:
        if n.op_type in ("Conv", "ConvTranspose"):
            kept_attrs = [a for a in n.attribute if a.name != "kernel_shape"]
            del n.attribute[:]
            n.attribute.extend(kept_attrs)


def set_attr_ints(node, name, vals):
    kept = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.extend([helper.make_attribute(name, vals)])


def finalize(model, remove_kernel_shape=False):
    clear_metadata(model)

    if remove_kernel_shape:
        remove_kernel_shape_attrs(model)

    prune_dead_nodes_fixed(model)
    topo_sort(model)

    del model.opset_import[:]
    model.opset_import.extend([helper.make_opsetid("", 11)])
    return model


def save(model, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, OUT_DIR / name)


# ----------------------------
# Candidate A:
# f16 template_mask -> bool template_mask + Where
# ----------------------------

def replace_template_mask_f16_with_bool_where(model):
    """
    Candidate 200.

    既存:
      template_mask = Sign(R_02)              # f16 [1,1,30,30]
      template_red  = mask_red  * template_mask
      template_blue = mask_blue * template_mask

    置換:
      template_mask_bool = Greater(R_02, 0)   # bool [1,1,30,30]
      template_red  = Where(template_mask_bool, mask_red,  zero)
      template_blue = Where(template_mask_bool, mask_blue, zero)

    狙い:
      template_mask f16 1800 memory を bool 900 memory にする。
      params は scalar zero だけ増える。
    """
    remove_nodes_producing(model, ["template_mask", "template_red", "template_blue"])
    remove_nodes_by_name(model, ["template_mask", "template_red", "template_blue"])

    model.graph.initializer.extend([
        init_f16("gpt_zero_f16_scalar_v13", [0.0], [1, 1, 1, 1]),
    ])

    model.graph.node.extend([
        helper.make_node(
            "Greater",
            ["R_02", "gpt_zero_f16_scalar_v13"],
            ["template_mask_bool_v13"],
            name="gpt_template_mask_bool_v13",
        ),
        helper.make_node(
            "Where",
            ["template_mask_bool_v13", "mask_red", "gpt_zero_f16_scalar_v13"],
            ["template_red"],
            name="gpt_template_red_where_v13",
        ),
        helper.make_node(
            "Where",
            ["template_mask_bool_v13", "mask_blue", "gpt_zero_f16_scalar_v13"],
            ["template_blue"],
            name="gpt_template_blue_where_v13",
        ),
    ])

    return model


# ----------------------------
# Candidate B:
# shift kernel bank: params増で runtime mask生成を減らす
# ----------------------------

def replace_shift_kernel_with_361_bank(model, s):
    """
    params を増やして runtime y/x mask + Mul を消す版。
    修正版:
      - shift_kernel_shape に依存しない
      - Gather raw を明示 shape [1,1,19,19] に Reshape
    """
    remove_nodes_producing(
        model,
        [
            f"y_mask_bool_{s}",
            f"x_mask_bool_{s}",
            f"y_mask_f16_{s}",
            f"x_mask_f16_{s}",
            f"shift_kernel_{s}",
        ],
    )
    remove_nodes_by_name(
        model,
        [
            f"y_mask_bool_{s}",
            f"x_mask_bool_{s}",
            f"y_mask_f16_{s}",
            f"x_mask_f16_{s}",
            f"shift_kernel_{s}",
        ],
    )

    bank = np.zeros((361, 1, 19, 19), dtype=np.float16)
    for y in range(19):
        for x in range(19):
            bank[y * 19 + x, 0, y, x] = 1.0

    model.graph.initializer.extend([
        numpy_helper.from_array(bank, name=f"gpt_shift_kernel_bank_s{s}_v13"),
        init_i64(f"gpt_19_i64_s{s}_v13", [19], [1, 1, 1, 1]),
        init_i64(f"gpt_shift_kernel_shape_s{s}_v13", [1, 1, 19, 19], [4]),
    ])

    model.graph.node.extend([
        helper.make_node(
            "Mul",
            [f"thresh_y_{s}", f"gpt_19_i64_s{s}_v13"],
            [f"gpt_shift_flat_y_s{s}_v13"],
            name=f"gpt_shift_flat_y_s{s}_v13",
        ),
        helper.make_node(
            "Add",
            [f"gpt_shift_flat_y_s{s}_v13", f"thresh_x_{s}"],
            [f"gpt_shift_flat_idx_s{s}_v13"],
            name=f"gpt_shift_flat_idx_s{s}_v13",
        ),
        helper.make_node(
            "Gather",
            [f"gpt_shift_kernel_bank_s{s}_v13", f"gpt_shift_flat_idx_s{s}_v13"],
            [f"gpt_shift_kernel_gather_raw_s{s}_v13"],
            name=f"gpt_shift_kernel_gather_raw_s{s}_v13",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            [f"gpt_shift_kernel_gather_raw_s{s}_v13", f"gpt_shift_kernel_shape_s{s}_v13"],
            [f"shift_kernel_{s}"],
            name=f"gpt_shift_kernel_reshape_s{s}_v13",
        ),
    ])

    return model

# ----------------------------
# Candidate C:
# blue_offset_full 59x59 を直接小さくする pad sweep
# ----------------------------

def direct_blue_offset_conv_to_cropped(model, pad_top, pad_left, out_size):
    """
    Candidate 300系。

    既存:
      blue_offset_full = Conv(..., pads=[29,29,29,29])  # 59x59
      blue_offset_cropped = Slice(blue_offset_full, ...) # 15x15想定
      blue_offset_flipped = Slice(blue_offset_cropped, negative step)

    置換:
      blue_offset_full Conv の output を直接 blue_offset_cropped にし、
      pads を調整して小さい spatial output にする。

    out_size=15なら pad_top+pad_bottom=14。
    out_size=16なら pad_top+pad_bottom=15。
    out_size=17なら pad_top+pad_bottom=16。

    これが通ると blue_offset_full [59,59] が消えるので memory 改善が大きい。
    """
    pad_sum = out_size - 1
    pad_bottom = pad_sum - pad_top
    pad_right = pad_sum - pad_left

    if pad_bottom < 0 or pad_right < 0:
        return model

    # crop node を消す
    remove_nodes_producing(model, ["blue_offset_cropped"])
    remove_nodes_by_name(model, ["blue_offset_cropped"])

    found = False
    for n in model.graph.node:
        if n.name == "blue_offset_full" and n.op_type == "Conv":
            n.output[0] = "blue_offset_cropped"
            set_attr_ints(n, "pads", [pad_top, pad_left, pad_bottom, pad_right])
            found = True

    if not found:
        print("WARNING: blue_offset_full Conv not found")

    return model


def main():
    base = onnx.load(BASE)

    # 199: baseline refinalize
    m = copy.deepcopy(base)
    finalize(m)
    save(m, "199_107_refinalized_v13.onnx")

    # 200: template_mask bool + Where
    m = copy.deepcopy(base)
    replace_template_mask_f16_with_bool_where(m)
    finalize(m)
    save(m, "200_107_template_bool_where.onnx")

    # 201: 200 + remove kernel attrs
    m = copy.deepcopy(base)
    replace_template_mask_f16_with_bool_where(m)
    finalize(m, remove_kernel_shape=True)
    save(m, "201_107_template_bool_where_remove_kernel_attrs.onnx")

    # 210-213: shift bank, params heavy / memory probe
    m = copy.deepcopy(base)
    replace_shift_kernel_with_361_bank(m, 1)
    finalize(m)
    save(m, "210_107_shift_bank_s1_params_heavy.onnx")

    m = copy.deepcopy(base)
    replace_shift_kernel_with_361_bank(m, 2)
    finalize(m)
    save(m, "211_107_shift_bank_s2_params_heavy.onnx")

    m = copy.deepcopy(base)
    replace_shift_kernel_with_361_bank(m, 3)
    finalize(m)
    save(m, "212_107_shift_bank_s3_params_heavy.onnx")

    m = copy.deepcopy(base)
    for s in [1, 2, 3]:
        replace_shift_kernel_with_361_bank(m, s)
    finalize(m)
    save(m, "213_107_shift_bank_all_params_heavy.onnx")

    # 220: template bool + shift bank all
    m = copy.deepcopy(base)
    replace_template_mask_f16_with_bool_where(m)
    for s in [1, 2, 3]:
        replace_shift_kernel_with_361_bank(m, s)
    finalize(m)
    save(m, "220_107_template_bool_shift_bank_all_params_heavy.onnx")

    # ----------------------------
    # blue_offset direct Conv focused sweep
    # ----------------------------
    # まずは out_size=15 の代表点だけ。
    # 通りそうなら FULL_SWEEP=True にして全探索。
    FULL_SWEEP = False

    if FULL_SWEEP:
        candidates = []
        for out_size in [15, 16, 17]:
            pad_sum = out_size - 1
            for pt in range(pad_sum + 1):
                for pl in range(pad_sum + 1):
                    candidates.append((out_size, pt, pl))
    else:
        # 代表点。中央寄り・端寄りだけをまず確認。
        candidates = []
        for out_size in [15, 16, 17]:
            pad_sum = out_size - 1
            vals = sorted(set([
                0,
                1,
                pad_sum // 4,
                pad_sum // 2,
                (pad_sum * 3) // 4,
                pad_sum - 1,
                pad_sum,
            ]))
            for pt in vals:
                for pl in vals:
                    candidates.append((out_size, pt, pl))

    idx = 300
    for out_size, pt, pl in candidates:
        m = copy.deepcopy(base)
        direct_blue_offset_conv_to_cropped(m, pt, pl, out_size)
        finalize(m)
        save(m, f"{idx}_107_blue_offset_direct_o{out_size}_pt{pt}_pl{pl}.onnx")
        idx += 1

    # 400系: template bool + blue_offset direct
    idx = 400
    for out_size, pt, pl in candidates:
        m = copy.deepcopy(base)
        replace_template_mask_f16_with_bool_where(m)
        direct_blue_offset_conv_to_cropped(m, pt, pl, out_size)
        finalize(m)
        save(m, f"{idx}_107_template_bool_blue_offset_direct_o{out_size}_pt{pt}_pl{pl}.onnx")
        idx += 1


if __name__ == "__main__":
    main()