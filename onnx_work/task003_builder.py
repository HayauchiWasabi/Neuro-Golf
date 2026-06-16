from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


Pair = Tuple[str, str]
CellCheck = Tuple[str, str, int]


def init_np(name: str, arr: np.ndarray):
    return numpy_helper.from_array(arr, name=name)


def scalar_f16(name: str, value: float):
    return init_np(name, np.array(value, dtype=np.float16))


def vec_i64(name: str, values: Iterable[int]):
    return init_np(name, np.array(list(values), dtype=np.int64))


def make_slice(
    nodes: List[onnx.NodeProto],
    x: str,
    y: str,
    starts: str,
    ends: str,
    axes: str,
    name: str,
):
    nodes.append(
        helper.make_node(
            "Slice",
            [x, starts, ends, axes],
            [y],
            name=name,
        )
    )


def add_squared_reduce_less(
    nodes: List[onnx.NodeProto],
    x: str,
    half: str,
    out_bool: str,
    prefix: str,
):
    """
    Check whether all compared values are equal.

    Logic:
      diff -> diff^2 -> ReduceSum -> Less(sum, 0.5)

    Since ARC values are 0/1-ish, exact match gives sum 0.
    Any mismatch gives sum >= 1, so threshold 0.5 is enough.
    """
    nodes.append(
        helper.make_node(
            "Mul",
            [x, x],
            [f"{prefix}_sq"],
            name=f"{prefix}_mul_sq",
        )
    )
    nodes.append(
        helper.make_node(
            "ReduceSum",
            [f"{prefix}_sq"],
            [f"{prefix}_sum"],
            name=f"{prefix}_reduce_sum",
            axes=[0, 1, 2, 3],
            keepdims=1,
        )
    )
    nodes.append(
        helper.make_node(
            "Less",
            [f"{prefix}_sum", half],
            [out_bool],
            name=f"{prefix}_less_half",
        )
    )


def add_pair_check(
    nodes: List[onnx.NodeProto],
    p: str,
    pairs: Sequence[Pair],
    half: str,
    out_bool: str,
    prefix: str,
):
    """
    Row-level pair check.

    Each row tensor is [1, 1, 1, 3].

    Example:
      pairs = [("r0", "r2"), ("r1", "r3")]

    means:
      r0 == r2 and r1 == r3
    """
    diffs = []

    for a, b in pairs:
        d = f"{prefix}_d_{a}_{b}"
        nodes.append(
            helper.make_node(
                "Sub",
                [f"{p}_{a}", f"{p}_{b}"],
                [d],
                name=f"{prefix}_sub_{a}_{b}",
            )
        )
        diffs.append(d)

    if len(diffs) == 1:
        add_squared_reduce_less(nodes, diffs[0], half, out_bool, prefix)
    else:
        nodes.append(
            helper.make_node(
                "Concat",
                diffs,
                [f"{prefix}_dcat"],
                name=f"{prefix}_concat_diffs",
                axis=2,
            )
        )
        add_squared_reduce_less(nodes, f"{prefix}_dcat", half, out_bool, prefix)


def add_cell_check(
    nodes: List[onnx.NodeProto],
    p: str,
    checks: Sequence[CellCheck],
    half: str,
    out_bool: str,
    prefix: str,
):
    """
    Cell-level check.

    Each check is:
      (row_a, row_b, col_index)

    Row tensors are [1, 1, 1, 3].
    We slice width axis=3 to one cell, then compare using:
      Sub -> Mul -> ReduceSum -> Less
    """
    diffs = []

    for idx, (a, b, col) in enumerate(checks):
        a_cell = f"{prefix}_{a}_c{col}_{idx}"
        b_cell = f"{prefix}_{b}_c{col}_{idx}"
        d = f"{prefix}_d_{a}_{b}_c{col}_{idx}"

        make_slice(
            nodes,
            f"{p}_{a}",
            a_cell,
            f"{p}_w{col}",
            f"{p}_w{col + 1}",
            f"{p}_axis3",
            f"{prefix}_slice_{a}_c{col}_{idx}",
        )
        make_slice(
            nodes,
            f"{p}_{b}",
            b_cell,
            f"{p}_w{col}",
            f"{p}_w{col + 1}",
            f"{p}_axis3",
            f"{prefix}_slice_{b}_c{col}_{idx}",
        )

        nodes.append(
            helper.make_node(
                "Sub",
                [a_cell, b_cell],
                [d],
                name=f"{prefix}_sub_{a}_{b}_c{col}_{idx}",
            )
        )
        diffs.append(d)

    if len(diffs) == 1:
        add_squared_reduce_less(nodes, diffs[0], half, out_bool, prefix)
    else:
        nodes.append(
            helper.make_node(
                "Concat",
                diffs,
                [f"{prefix}_dcat"],
                name=f"{prefix}_concat_diffs",
                axis=3,
            )
        )
        add_squared_reduce_less(nodes, f"{prefix}_dcat", half, out_bool, prefix)


def build_candidate(
    *,
    out_path: Path,
    candidate: str,
    p2_pairs: Sequence[Pair] | None = None,
    p3_pairs: Sequence[Pair] | None = None,
    p2_cells: Sequence[CellCheck] | None = None,
    p3_cells: Sequence[CellCheck] | None = None,
    use_split_rows: bool = True,
    compact_names: bool = True,
    ir_version: int = 8,
    opset: int = 10,
):
    """
    Build one task003 candidate ONNX.

    Model shape:
      input  float32 [1, 10, 30, 30]
      output float16 [1, 10, 30, 30]

    Core logic:
      ch1 = input[:, 1:2, 0:6, 0:3]
      rows r0..r5 from ch1

      P2 and P3 are decided by p2_pairs/p2_cells and p3_pairs/p3_cells.

      is_P3 = P3_raw
      is_P4 = not P2 and not P3_raw

      out6 = P4 ? r2 : r0
      out7 = P4 ? r3 : r1
      out8 = P3 ? r2 : r0

      ch1_9 = concat(ch1, out6, out7, out8)
      channel0 = 1 - ch1_9
      channel1 = 0
      channel2 = ch1_9
      channels3..9 = 0
      pad to 30x30
    """
    p = candidate if not compact_names else candidate.replace("rebuild_", "").replace("_", "")

    nodes: List[onnx.NodeProto] = []
    inits: List[onnx.TensorProto] = []

    one = f"{p}_one"
    zero = f"{p}_zero"
    half = f"{p}_half"
    axes123 = f"{p}_axes123"
    axis2 = f"{p}_axis2"
    axis3 = f"{p}_axis3"

    inits += [
        scalar_f16(one, 1.0),
        scalar_f16(zero, 0.0),
        scalar_f16(half, 0.5),
        vec_i64(axes123, [1, 2, 3]),
        vec_i64(axis2, [2]),
        vec_i64(axis3, [3]),
        vec_i64(f"{p}_ch1_start", [1, 0, 0]),
        vec_i64(f"{p}_ch1_end", [2, 6, 3]),
    ]

    # row slice constants
    for i in range(7):
        inits.append(vec_i64(f"{p}_c{i}", [i]))

    # width slice constants
    for i in range(4):
        inits.append(vec_i64(f"{p}_w{i}", [i]))

    # input[:, 1:2, 0:6, 0:3] -> float16
    make_slice(
        nodes,
        "input",
        f"{p}_ch1_f32",
        f"{p}_ch1_start",
        f"{p}_ch1_end",
        axes123,
        f"{p}_slice_ch1",
    )
    nodes.append(
        helper.make_node(
            "Cast",
            [f"{p}_ch1_f32"],
            [f"{p}_ch1"],
            name=f"{p}_cast_ch1_f16",
            to=TensorProto.FLOAT16,
        )
    )

    # r0..r5
    if use_split_rows:
        nodes.append(
            helper.make_node(
                "Split",
                [f"{p}_ch1"],
                [f"{p}_r{i}" for i in range(6)],
                name=f"{p}_split_rows",
                axis=2,
                split=[1, 1, 1, 1, 1, 1],
            )
        )
    else:
        for i in range(6):
            make_slice(
                nodes,
                f"{p}_ch1",
                f"{p}_r{i}",
                f"{p}_c{i}",
                f"{p}_c{i + 1}",
                axis2,
                f"{p}_slice_r{i}",
            )

    # P2 check
    if p2_cells is not None:
        add_cell_check(
            nodes,
            p,
            p2_cells,
            half,
            f"{p}_eq2_bool",
            f"{p}_p2cell",
        )
    elif p2_pairs is not None:
        add_pair_check(
            nodes,
            p,
            p2_pairs,
            half,
            f"{p}_eq2_bool",
            f"{p}_p2",
        )
    else:
        raise ValueError("p2_pairs or p2_cells is required")

    # P3 check
    if p3_cells is not None:
        add_cell_check(
            nodes,
            p,
            p3_cells,
            half,
            f"{p}_p3_raw_bool",
            f"{p}_p3cell",
        )
    elif p3_pairs is not None:
        add_pair_check(
            nodes,
            p,
            p3_pairs,
            half,
            f"{p}_p3_raw_bool",
            f"{p}_p3",
        )
    else:
        raise ValueError("p3_pairs or p3_cells is required")

    # v13/v23-style:
    # is_P3 = P3_raw directly
    # is_P4 = not P2 and not P3_raw
    nodes.append(
        helper.make_node(
            "Not",
            [f"{p}_eq2_bool"],
            [f"{p}_not_p2_bool"],
            name=f"{p}_not_p2_bool",
        )
    )
    nodes.append(
        helper.make_node(
            "Not",
            [f"{p}_p3_raw_bool"],
            [f"{p}_not_p3_bool"],
            name=f"{p}_not_p3_bool",
        )
    )
    nodes.append(
        helper.make_node(
            "And",
            [f"{p}_not_p2_bool", f"{p}_not_p3_bool"],
            [f"{p}_is_p4_bool"],
            name=f"{p}_and_is_p4_bool",
        )
    )

    # out6 = P4 ? r2 : r0
    nodes.append(
        helper.make_node(
            "Where",
            [f"{p}_is_p4_bool", f"{p}_r2", f"{p}_r0"],
            [f"{p}_out6"],
            name=f"{p}_where_out6",
        )
    )

    # out7 = P4 ? r3 : r1
    nodes.append(
        helper.make_node(
            "Where",
            [f"{p}_is_p4_bool", f"{p}_r3", f"{p}_r1"],
            [f"{p}_out7"],
            name=f"{p}_where_out7",
        )
    )

    # out8 = P3 ? r2 : r0
    nodes.append(
        helper.make_node(
            "Where",
            [f"{p}_p3_raw_bool", f"{p}_r2", f"{p}_r0"],
            [f"{p}_out8"],
            name=f"{p}_where_out8",
        )
    )

    # ch1_9 = original 6 rows + generated 3 rows
    nodes.append(
        helper.make_node(
            "Concat",
            [f"{p}_ch1", f"{p}_out6", f"{p}_out7", f"{p}_out8"],
            [f"{p}_ch1_9"],
            name=f"{p}_concat_ch1_9",
            axis=2,
        )
    )

    # output channels:
    # channel 0 = 1 - ch1_9
    # channel 1 = 0
    # channel 2 = ch1_9
    # channels 3..9 = 0
    nodes.append(
        helper.make_node(
            "Sub",
            [one, f"{p}_ch1_9"],
            [f"{p}_ch0_9"],
            name=f"{p}_sub_ch0",
        )
    )
    nodes.append(
        helper.make_node(
            "Mul",
            [f"{p}_ch1_9", zero],
            [f"{p}_zeros_9"],
            name=f"{p}_mul_zeros",
        )
    )

    nodes.append(
        helper.make_node(
            "Concat",
            [
                f"{p}_ch0_9",
                f"{p}_zeros_9",
                f"{p}_ch1_9",
                f"{p}_zeros_9",
                f"{p}_zeros_9",
                f"{p}_zeros_9",
                f"{p}_zeros_9",
                f"{p}_zeros_9",
                f"{p}_zeros_9",
                f"{p}_zeros_9",
            ],
            [f"{p}_out9x3"],
            name=f"{p}_concat_channels",
            axis=1,
        )
    )

    nodes.append(
        helper.make_node(
            "Pad",
            [f"{p}_out9x3"],
            ["output"],
            name=f"{p}_pad_output",
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 21, 27],
        )
    )

    graph = helper.make_graph(
        nodes,
        f"task003_{candidate}",
        inputs=[
            helper.make_tensor_value_info(
                "input",
                TensorProto.FLOAT,
                [1, 10, 30, 30],
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT16,
                [1, 10, 30, 30],
            )
        ],
        initializer=inits,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="gpt_task003_builder",
    )
    model.ir_version = ir_version

    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path)


def summarize_model(path: Path) -> str:
    m = onnx.load(path)

    op_counts = {}
    for n in m.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1

    init_elems = sum(
        int(np.prod(t.dims)) if t.dims else 1
        for t in m.graph.initializer
    )

    return (
        f"{path.name}: "
        f"nodes={len(m.graph.node)}, "
        f"initializers={len(m.graph.initializer)}, "
        f"init_elems={init_elems}, "
        f"ops={dict(sorted(op_counts.items()))}"
    )