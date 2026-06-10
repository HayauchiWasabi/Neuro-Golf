import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from neurogolf_common import run_task


TASK_NUM = 1


def tensor(name, array, dtype=None):
    arr = np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)
    return numpy_helper.from_array(arr, name=name)


def build_model():
    nodes = [
        helper.make_node("Slice", ["input"], ["p3"], starts=[0, 0], ends=[3, 3], axes=[2, 3]),
        helper.make_node("Slice", ["p3"], ["p3_bg"], starts=[0], ends=[1], axes=[1]),
        helper.make_node("Slice", ["p3"], ["p3_fg"], starts=[1], ends=[10], axes=[1]),
        helper.make_node("Cast", ["p3_fg"], ["p3_fg_h16"], to=TensorProto.FLOAT16),
        helper.make_node("ReduceMax", ["p3_fg_h16"], ["color_fg"], axes=[2, 3], keepdims=1),
        helper.make_node("Concat", ["fg_bg_zero", "color_fg"], ["color_onehot"], axis=1),
        helper.make_node("Cast", ["p3_bg"], ["bg3b"], to=TensorProto.BOOL),
        helper.make_node("Tile", ["bg3b", "tile_repeats_mask"], ["pat_bg9"]),
        helper.make_node("Gather", ["bg3b", "blk_idx"], ["sel_bg_rows"], axis=2),
        helper.make_node("Gather", ["sel_bg_rows", "blk_idx"], ["sel_bg9"], axis=3),
        helper.make_node("Or", ["pat_bg9", "sel_bg9"], ["out_bg_mask"]),
        helper.make_node("Where", ["out_bg_mask", "bg_onehot", "color_onehot"], ["out9"]),
        helper.make_node(
            "Pad",
            ["out9"],
            ["output"],
            pads=[0, 0, 0, 0, 0, 0, 21, 21],
            mode="constant",
            value=0.0,
        ),
    ]
    initializers = [
        tensor("tile_repeats_mask", [1, 1, 3, 3], np.int64),
        tensor("blk_idx", [0, 0, 0, 1, 1, 1, 2, 2, 2], np.int64),
        tensor(
            "bg_onehot",
            np.array([1.0] + [0.0] * 9, dtype=np.float16).reshape(1, 10, 1, 1),
        ),
        tensor("fg_bg_zero", np.zeros((1, 1, 1, 1), dtype=np.float16)),
    ]
    graph = helper.make_graph(
        nodes,
        "task001_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 9)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main():
    run_task(TASK_NUM, build_model)


if __name__ == "__main__":
    main()
