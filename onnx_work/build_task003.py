#!/usr/bin/env python3
"""
Task003 output-assembly candidates after v79.

Requires:
  onnx_work/task003_builder.py

Run:
  cd "/Users/kaiikeda/Programming/Kaggle/Neuro Golf"
  python onnx_work/build_task003.py

Current best:
  task003_rebuild_v79_r0r3only_p2_c00_noconcat_p4_or_not.onnx
  score 18.185457
  cost_delta -279

New candidates:
  v99-v102:
    zero = ch0_9 * 0 instead of ch1_9 * 0

  v103-v106:
    zero = out6 * 0, then tile-ish concat by rows is NOT used;
    instead keep channel concat but zero has [1,1,1,3], expected may fail shape.
    So not included.

  v103-v106:
    build ch0/ch1/output using Add/Sub variants:
      ch0 = 1 + (-ch1_9) via Neg + Add

  v107-v110:
    p2/p3 logic same as v79, but use c00/c02/c20/c22 and ultra-short names.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import List

import onnx
from onnx import TensorProto, helper


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from task003_builder import (
    scalar_f16,
    vec_i64,
    make_slice,
    add_pair_check,
    summarize_model,
)


OUT_DIR = Path("outputs/gpt_workbench/task003")
BASELINE = OUT_DIR / "task003_rebuild_v79_r0r3only_p2_c00_noconcat_p4_or_not.onnx"


def add_p2_cell_sum(nodes, p, p2_cells, half, out_bool, prefix):
    sqs = []

    for idx, (a, b, col) in enumerate(p2_cells):
        a_cell = f"{prefix}a{idx}"
        b_cell = f"{prefix}b{idx}"
        d = f"{prefix}d{idx}"
        sq = f"{prefix}s{idx}"

        make_slice(nodes, f"{p}_{a}", a_cell, f"{p}_w{col}", f"{p}_w{col + 1}", f"{p}_axis3", f"{prefix}sa{idx}")
        make_slice(nodes, f"{p}_{b}", b_cell, f"{p}_w{col}", f"{p}_w{col + 1}", f"{p}_axis3", f"{prefix}sb{idx}")

        nodes.append(helper.make_node("Sub", [a_cell, b_cell], [d], name=f"{prefix}sub{idx}"))
        nodes.append(helper.make_node("Mul", [d, d], [sq], name=f"{prefix}mul{idx}"))
        sqs.append(sq)

    if len(sqs) == 1:
        total = sqs[0]
    else:
        total = f"{prefix}sum"
        nodes.append(helper.make_node("Sum", sqs, [total], name=f"{prefix}sumop"))

    nodes.append(helper.make_node("Less", [total, half], [out_bool], name=f"{prefix}less"))


def build_candidate_output(
    *,
    out_path: Path,
    candidate: str,
    p2_cells,
    zero_source: str = "ch1",
    ch0_mode: str = "sub",
    ir_version: int = 8,
    opset: int = 10,
):
    """
    zero_source:
      - "ch1": zero = ch1_9 * 0
      - "ch0": zero = ch0_9 * 0

    ch0_mode:
      - "sub": ch0 = 1 - ch1_9
      - "neg_add": ch0 = 1 + Neg(ch1_9)
    """
    # intentionally very short prefix
    p = candidate.replace("rebuild_", "v").replace("_", "")

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

    for i in range(5):
        inits.append(vec_i64(f"{p}_c{i}", [i]))

    for i in range(4):
        inits.append(vec_i64(f"{p}_w{i}", [i]))

    # ch1
    make_slice(nodes, "input", f"{p}_ch1_f32", f"{p}_ch1_start", f"{p}_ch1_end", axes123, f"{p}_slice_ch1")
    nodes.append(helper.make_node("Cast", [f"{p}_ch1_f32"], [f"{p}_ch1"], name=f"{p}_cast", to=TensorProto.FLOAT16))

    # r0..r3 only
    for i in range(4):
        make_slice(nodes, f"{p}_ch1", f"{p}_r{i}", f"{p}_c{i}", f"{p}_c{i + 1}", axis2, f"{p}_sr{i}")

    # P2 no-Concat
    add_p2_cell_sum(
        nodes,
        p,
        p2_cells,
        half,
        f"{p}_p2",
        f"{p}p2",
    )

    # P3 = r0 == r3
    add_pair_check(
        nodes,
        p,
        [("r0", "r3")],
        half,
        f"{p}_p3",
        f"{p}p3",
    )

    # P4 = Not(Or(P2, P3))
    nodes.append(helper.make_node("Or", [f"{p}_p2", f"{p}_p3"], [f"{p}_or"], name=f"{p}_or"))
    nodes.append(helper.make_node("Not", [f"{p}_or"], [f"{p}_p4"], name=f"{p}_not"))

    # output rows
    nodes.append(helper.make_node("Where", [f"{p}_p4", f"{p}_r2", f"{p}_r0"], [f"{p}_o6"], name=f"{p}_w6"))
    nodes.append(helper.make_node("Where", [f"{p}_p4", f"{p}_r3", f"{p}_r1"], [f"{p}_o7"], name=f"{p}_w7"))
    nodes.append(helper.make_node("Where", [f"{p}_p3", f"{p}_r2", f"{p}_r0"], [f"{p}_o8"], name=f"{p}_w8"))

    nodes.append(helper.make_node("Concat", [f"{p}_ch1", f"{p}_o6", f"{p}_o7", f"{p}_o8"], [f"{p}_ch1_9"], name=f"{p}_cat9", axis=2))

    # ch0
    if ch0_mode == "sub":
        nodes.append(helper.make_node("Sub", [one, f"{p}_ch1_9"], [f"{p}_ch0_9"], name=f"{p}_ch0sub"))
    elif ch0_mode == "neg_add":
        nodes.append(helper.make_node("Neg", [f"{p}_ch1_9"], [f"{p}_neg"], name=f"{p}_neg"))
        nodes.append(helper.make_node("Add", [one, f"{p}_neg"], [f"{p}_ch0_9"], name=f"{p}_ch0add"))
    else:
        raise ValueError(ch0_mode)

    # zeros
    if zero_source == "ch1":
        zero_input = f"{p}_ch1_9"
    elif zero_source == "ch0":
        zero_input = f"{p}_ch0_9"
    else:
        raise ValueError(zero_source)

    nodes.append(helper.make_node("Mul", [zero_input, zero], [f"{p}_z9"], name=f"{p}_zero9"))

    # channels
    nodes.append(
        helper.make_node(
            "Concat",
            [
                f"{p}_ch0_9",
                f"{p}_z9",
                f"{p}_ch1_9",
                f"{p}_z9",
                f"{p}_z9",
                f"{p}_z9",
                f"{p}_z9",
                f"{p}_z9",
                f"{p}_z9",
                f"{p}_z9",
            ],
            [f"{p}_out9x3"],
            name=f"{p}_catc",
            axis=1,
        )
    )

    nodes.append(helper.make_node("Pad", [f"{p}_out9x3"], ["output"], name=f"{p}_pad", mode="constant", pads=[0, 0, 0, 0, 0, 0, 21, 27]))

    graph = helper.make_graph(
        nodes,
        f"task003_{candidate}",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        initializer=inits,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="gpt_task003_output_candidates",
    )
    model.ir_version = ir_version

    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    made = []

    if BASELINE.exists():
        baseline_copy = OUT_DIR / "task003_best_v79_baseline.onnx"
        shutil.copy2(BASELINE, baseline_copy)
        made.append(baseline_copy)

    p2_variants = [
        ("c00", [("r0", "r2", 0), ("r1", "r3", 0)]),
        ("c02", [("r0", "r2", 0), ("r1", "r3", 2)]),
        ("c20", [("r0", "r2", 2), ("r1", "r3", 0)]),
        ("c22", [("r0", "r2", 2), ("r1", "r3", 2)]),
    ]

    candidates = []

    # v99-v102: zero from ch0 instead of ch1
    for i, (tag, p2_cells) in enumerate(p2_variants, start=99):
        candidates.append(
            (
                f"rebuild_v{i}_out_zero_from_ch0_p2_{tag}",
                p2_cells,
                "ch0",
                "sub",
            )
        )

    # v103-v106: ch0 via Neg+Add
    for i, (tag, p2_cells) in enumerate(p2_variants, start=103):
        candidates.append(
            (
                f"rebuild_v{i}_out_ch0_neg_add_p2_{tag}",
                p2_cells,
                "ch1",
                "neg_add",
            )
        )

    # v107-v110: ch0 via Neg+Add + zero from ch0
    for i, (tag, p2_cells) in enumerate(p2_variants, start=107):
        candidates.append(
            (
                f"rebuild_v{i}_out_ch0_neg_add_zero_ch0_p2_{tag}",
                p2_cells,
                "ch0",
                "neg_add",
            )
        )

    for name, p2_cells, zero_source, ch0_mode in candidates:
        path = OUT_DIR / f"task003_{name}.onnx"
        build_candidate_output(
            out_path=path,
            candidate=name,
            p2_cells=p2_cells,
            zero_source=zero_source,
            ch0_mode=ch0_mode,
        )
        made.append(path)

    print(f"Generated {len(made)} files under: {OUT_DIR}")
    for path in made:
        try:
            print("  " + summarize_model(path))
        except Exception as e:
            print(f"  {path.name}: summary failed: {e}")

    print("\nCurrent best to beat:")
    print("  task003_rebuild_v79_r0r3only_p2_c00_noconcat_p4_or_not.onnx")
    print("  score_delta +0.267166 / cost_delta -279")

    print("\nCandidate groups:")
    print("  v99-v102: zero = ch0_9 * 0")
    print("  v103-v106: ch0 = Neg(ch1_9) + 1")
    print("  v107-v110: Neg+Add ch0 and zero from ch0")

    print("\nAdopt only if validation_status == ok and pass == 265.")


if __name__ == "__main__":
    main()