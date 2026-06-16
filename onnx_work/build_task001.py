#!/usr/bin/env python3
"""
Build task001 output-redesign memory candidates.

Current best:
  task001_next_color_pad_short.onnx
  pass 268/268
  memory_delta -207
  params_delta -1
  cost_delta -208
  score_delta +0.080477

Current best graph:
  background:
    input -> Slice bg -> Cast BOOL
          -> Gather/Gather + Tile -> Or -> m

  foreground:
    input -> Slice fg -> Cast FLOAT16 -> ReduceMax -> cf
    Pad(cf, channel-before=1) -> co

  output:
    Where(m, bo, co) -> o9
    Pad(o9) -> output

New candidates:
1. task001_out_direct_mask_f16pad.onnx
   Replace:
     Where -> out9 -> Pad
   with:
     m BOOL -> Cast FLOAT16 -> Pad to 30x30 with value=1 -> Cast BOOL
     Where(mask30, bo, co) -> output

   Goal:
     Remove out9 [1,10,9,9] FLOAT16 and final Pad(out9).
   Risk:
     Adds mask30 [1,1,30,30], so memory may worsen.

2. task001_out_direct_mask_f32pad.onnx
   Same as above but mask Pad uses FLOAT instead of FLOAT16.
   Goal:
     Check runtime/type behavior. Likely worse memory, but may be valid.

3. task001_out_baseline_short.onnx
   Rebuild current best short as control.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT_DIR = Path("outputs/gpt_workbench/task001")


def init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def make_io():
    input_vi = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 10, 30, 30],
    )
    output_vi = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT16,
        [1, 10, 30, 30],
    )
    return input_vi, output_vi


def initializers():
    return [
        init("r", np.array([1, 1, 3, 3], dtype=np.int64)),
        init("i", np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)),
        init(
            "bo",
            np.array(
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.float16,
            ).reshape(1, 10, 1, 1),
        ),
    ]


def bg_nodes():
    """
    Known-good background mask.

    b0 = input[:, 0:1, 0:3, 0:3]
    b  = BOOL background mask [1,1,3,3]

    t = tiled repeat [1,1,9,9]
    g = block expand [1,1,9,9]
    m = t OR g
    """
    return [
        helper.make_node(
            "Slice",
            ["input"],
            ["b0"],
            name="a",
            axes=[1, 2, 3],
            starts=[0, 0, 0],
            ends=[1, 3, 3],
        ),
        helper.make_node(
            "Cast",
            ["b0"],
            ["b"],
            name="b",
            to=TensorProto.BOOL,
        ),
        helper.make_node(
            "Tile",
            ["b", "r"],
            ["t"],
            name="c",
        ),
        helper.make_node(
            "Gather",
            ["b", "i"],
            ["g0"],
            name="d",
            axis=2,
        ),
        helper.make_node(
            "Gather",
            ["g0", "i"],
            ["g"],
            name="e",
            axis=3,
        ),
        helper.make_node(
            "Or",
            ["t", "g"],
            ["m"],
            name="f",
        ),
    ]


def fg_nodes():
    """
    Known-best foreground.

    x  = foreground slice [1,9,3,3] FLOAT
    h  = foreground slice [1,9,3,3] FLOAT16
    cf = detected foreground color [1,9,1,1]
    co = color onehot [1,10,1,1], made by padding channel 0.
    """
    return [
        helper.make_node(
            "Slice",
            ["input"],
            ["x"],
            name="g",
            axes=[1, 2, 3],
            starts=[1, 0, 0],
            ends=[10, 3, 3],
        ),
        helper.make_node(
            "Cast",
            ["x"],
            ["h"],
            name="h",
            to=TensorProto.FLOAT16,
        ),
        helper.make_node(
            "ReduceMax",
            ["h"],
            ["cf"],
            name="j",
            axes=[2, 3],
            keepdims=1,
        ),
        helper.make_node(
            "Pad",
            ["cf"],
            ["co"],
            name="k",
            mode="constant",
            pads=[0, 1, 0, 0, 0, 0, 0, 0],
            value=0.0,
        ),
    ]


def output_baseline_short():
    """
    Current best output:
      Where -> out9
      Pad out9 -> output
    """
    return [
        helper.make_node(
            "Where",
            ["m", "bo", "co"],
            ["o9"],
            name="l",
        ),
        helper.make_node(
            "Pad",
            ["o9"],
            ["output"],
            name="m",
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 21, 21],
            value=0.0,
        ),
    ]


def output_direct_mask_f16pad():
    """
    Candidate 1:
      m BOOL [1,1,9,9]
        -> Cast FLOAT16
        -> Pad to [1,1,30,30] with value=1
        -> Cast BOOL
      Where(mask30, bo, co) -> output

    Outside the 9x9 area must be background.
    Since Where(condition, bo, co), condition=True selects background.
    Therefore mask padding value must be True.
    """
    return [
        helper.make_node(
            "Cast",
            ["m"],
            ["mf"],
            name="l",
            to=TensorProto.FLOAT16,
        ),
        helper.make_node(
            "Pad",
            ["mf"],
            ["mp"],
            name="m",
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 21, 21],
            value=1.0,
        ),
        helper.make_node(
            "Cast",
            ["mp"],
            ["mb"],
            name="n",
            to=TensorProto.BOOL,
        ),
        helper.make_node(
            "Where",
            ["mb", "bo", "co"],
            ["output"],
            name="o",
        ),
    ]


def output_direct_mask_f32pad():
    """
    Candidate 2:
    Same as f16pad, but use FLOAT32 mask pad.

    This may be worse, but helps check whether FLOAT16 Pad/Cast behavior
    is treated differently by runtime/evaluator.
    """
    return [
        helper.make_node(
            "Cast",
            ["m"],
            ["mf"],
            name="l",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Pad",
            ["mf"],
            ["mp"],
            name="m",
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 21, 21],
            value=1.0,
        ),
        helper.make_node(
            "Cast",
            ["mp"],
            ["mb"],
            name="n",
            to=TensorProto.BOOL,
        ),
        helper.make_node(
            "Where",
            ["mb", "bo", "co"],
            ["output"],
            name="o",
        ),
    ]


def build_model(name: str, output_builder):
    input_vi, output_vi = make_io()

    nodes = []
    nodes += bg_nodes()
    nodes += fg_nodes()
    nodes += output_builder()

    graph = helper.make_graph(
        nodes=nodes,
        name=name,
        inputs=[input_vi],
        outputs=[output_vi],
        initializer=initializers(),
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 9)],
        producer_name=name,
    )
    model.ir_version = 8
    return model


def save_model(model: onnx.ModelProto, filename: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename

    onnx.checker.check_model(model)
    onnx.save(model, path)

    print(f"saved: {path}")
    print(f"  nodes: {len(model.graph.node)}")
    print(f"  initializers: {len(model.graph.initializer)}")
    print()


def main():
    candidates = [
        (
            "task001_out_baseline_short",
            output_baseline_short,
            "task001_out_baseline_short.onnx",
        ),
        (
            "task001_out_direct_mask_f16pad",
            output_direct_mask_f16pad,
            "task001_out_direct_mask_f16pad.onnx",
        ),
        (
            "task001_out_direct_mask_f32pad",
            output_direct_mask_f32pad,
            "task001_out_direct_mask_f32pad.onnx",
        ),
    ]

    for name, builder, filename in candidates:
        try:
            model = build_model(name, builder)
            save_model(model, filename)
        except Exception as e:
            print(f"failed: {filename}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()