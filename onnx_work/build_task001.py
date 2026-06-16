#!/usr/bin/env python3
"""
Build task001 ONNX candidate: no shared p3 tensor.

Goal:
- Reduce memory by avoiding:
    p3 [1,10,3,3] float32
- Slice background and foreground directly from input.
- Keep the original background mask logic:
    out_bg_mask = Tile(bg_mask, [1,1,3,3]) OR BlockExpand(bg_mask)
- Do not use banned ops.

Output:
  outputs/gpt_workbench/task001/task001_no_p3_direct_slice.onnx
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT_PATH = Path("outputs/gpt_workbench/task001/task001_no_p3_direct_slice.onnx")


def make_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def build_model() -> onnx.ModelProto:
    input_vi = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 10, 30, 30],
    )

    # Base task001 passed with FLOAT16 output, so keep it unchanged.
    output_vi = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT16,
        [1, 10, 30, 30],
    )

    initializers = [
        make_initializer(
            "tile_repeats_mask",
            np.array([1, 1, 3, 3], dtype=np.int64),
        ),
        make_initializer(
            "blk_idx",
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64),
        ),
        make_initializer(
            "bg_onehot",
            np.array(
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.float16,
            ).reshape(1, 10, 1, 1),
        ),
        make_initializer(
            "fg_bg_zero",
            np.array([0], dtype=np.float16).reshape(1, 1, 1, 1),
        ),
    ]

    nodes = [
        # Background direct slice:
        # input[:, 0:1, 0:3, 0:3]
        helper.make_node(
            "Slice",
            inputs=["input"],
            outputs=["p3_bg"],
            name="slice_bg_direct",
            axes=[1, 2, 3],
            starts=[0, 0, 0],
            ends=[1, 3, 3],
        ),
        helper.make_node(
            "Cast",
            inputs=["p3_bg"],
            outputs=["bg3b"],
            name="cast_bg_to_bool",
            to=TensorProto.BOOL,
        ),

        # Tile repeat path:
        # 3x3 pattern repeated to 9x9.
        helper.make_node(
            "Tile",
            inputs=["bg3b", "tile_repeats_mask"],
            outputs=["pat_bg9"],
            name="tile_bg_pattern",
        ),

        # Block expand path:
        # Each 3x3 cell expands into 3x3 block.
        helper.make_node(
            "Gather",
            inputs=["bg3b", "blk_idx"],
            outputs=["sel_bg_rows"],
            name="gather_bg_rows_block",
            axis=2,
        ),
        helper.make_node(
            "Gather",
            inputs=["sel_bg_rows", "blk_idx"],
            outputs=["sel_bg9"],
            name="gather_bg_cols_block",
            axis=3,
        ),

        # Final background mask.
        helper.make_node(
            "Or",
            inputs=["pat_bg9", "sel_bg9"],
            outputs=["out_bg_mask"],
            name="or_tile_and_block_bg",
        ),

        # Foreground direct slice:
        # input[:, 1:10, 0:3, 0:3]
        #
        # Keep the original spirit:
        # - reduce float32 lifetime by slicing only foreground region
        # - cast to UINT8 before FLOAT16 as base did
        helper.make_node(
            "Slice",
            inputs=["input"],
            outputs=["p3_fg_float"],
            name="slice_fg_direct",
            axes=[1, 2, 3],
            starts=[1, 0, 0],
            ends=[10, 3, 3],
        ),
        helper.make_node(
            "Cast",
            inputs=["p3_fg_float"],
            outputs=["p3_fg_u8"],
            name="cast_fg_to_u8",
            to=TensorProto.UINT8,
        ),
        helper.make_node(
            "Cast",
            inputs=["p3_fg_u8"],
            outputs=["p3_fg_f16"],
            name="cast_fg_to_f16",
            to=TensorProto.FLOAT16,
        ),
        helper.make_node(
            "ReduceMax",
            inputs=["p3_fg_f16"],
            outputs=["color_fg"],
            name="reduce_fg_color",
            axes=[2, 3],
            keepdims=1,
        ),
        helper.make_node(
            "Concat",
            inputs=["fg_bg_zero", "color_fg"],
            outputs=["color_onehot"],
            name="concat_color_onehot",
            axis=1,
        ),
        helper.make_node(
            "Where",
            inputs=["out_bg_mask", "bg_onehot", "color_onehot"],
            outputs=["out9"],
            name="where_bg_or_fg",
        ),
        helper.make_node(
            "Pad",
            inputs=["out9"],
            outputs=["output"],
            name="pad_to_30x30",
            mode="constant",
            pads=[0, 0, 0, 0, 0, 0, 21, 21],
            value=0.0,
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="task001_no_p3_direct_slice",
        inputs=[input_vi],
        outputs=[output_vi],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 9)],
        producer_name="gpt_task001_no_p3_direct_slice",
    )
    model.ir_version = 8
    return model


def main() -> None:
    model = build_model()
    onnx.checker.check_model(model)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT_PATH)

    print(f"saved: {OUT_PATH}")
    print(f"nodes: {len(model.graph.node)}")
    print(f"initializers: {len(model.graph.initializer)}")


if __name__ == "__main__":
    main()