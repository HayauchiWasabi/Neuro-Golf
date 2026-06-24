# onnx_work/task001.py
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT_DIR = "outputs/gpt_workbench/task001"


def t_i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def t_f16(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.float16), name=name)


def t_f32(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def make_common_initializers():
    inits = []

    # Slice input[:, 0:1, 0:3, 0:3]
    inits += [
        t_i64("s0_starts", [0, 0, 0]),
        t_i64("s0_ends", [1, 3, 3]),
        t_i64("s0_axes", [1, 2, 3]),
    ]

    # Slice input[:, 1:10, 0:3, 0:3]
    inits += [
        t_i64("s1_starts", [1, 0, 0]),
        t_i64("s1_ends", [10, 3, 3]),
        t_i64("s1_axes", [1, 2, 3]),
    ]

    # Tile repeats
    inits.append(t_i64("repeats_3x", [1, 1, 3, 3]))

    # ConvTranspose weight: [C_in, C_out/group, kH, kW]
    inits.append(t_f16("w_deconv_f16", np.ones((1, 1, 3, 3), dtype=np.float16)))
    inits.append(t_f32("w_deconv_f32", np.ones((1, 1, 3, 3), dtype=np.float32)))

    # Where true branch: color0 = 1, others = 0
    bo = np.zeros((1, 10, 1, 1), dtype=np.float16)
    bo[:, 0, :, :] = 1.0
    inits.append(t_f16("bo", bo))

    # Pad for [1,9,1,1] -> [1,10,1,1]
    # opset 11 Pad input format
    inits.append(t_i64("pads_color0", [0, 1, 0, 0, 0, 0, 0, 0]))
    inits.append(t_f16("pad_value_f16", np.array(0.0, dtype=np.float16)))

    # Pad for [1,10,9,9] -> [1,10,30,30]
    inits.append(t_i64("pads_to_30", [0, 0, 0, 0, 0, 0, 21, 21]))

    return inits


def common_prefix_nodes():
    return [
        helper.make_node(
            "Slice",
            ["input", "s0_starts", "s0_ends", "s0_axes"],
            ["b0"],
            name="slice_color0_3x3",
        ),
        helper.make_node(
            "Cast",
            ["b0"],
            ["b_bool"],
            name="cast_color0_bool",
            to=TensorProto.BOOL,
        ),
        helper.make_node(
            "Cast",
            ["b0"],
            ["b_f16"],
            name="cast_color0_f16",
            to=TensorProto.FLOAT16,
        ),
    ]


def common_suffix_nodes(mask_name):
    return [
        helper.make_node(
            "Slice",
            ["input", "s1_starts", "s1_ends", "s1_axes"],
            ["x"],
            name="slice_colors_1_9",
        ),
        helper.make_node(
            "Cast",
            ["x"],
            ["h"],
            name="cast_x_f16",
            to=TensorProto.FLOAT16,
        ),
        helper.make_node(
            "ReduceMax",
            ["h"],
            ["cf"],
            name="reduce_color_presence",
            axes=[2, 3],
            keepdims=1,
        ),
        helper.make_node(
            "Pad",
            ["cf", "pads_color0", "pad_value_f16"],
            ["co"],
            name="pad_color0_channel",
            mode="constant",
        ),
        helper.make_node(
            "Where",
            [mask_name, "bo", "co"],
            ["o9"],
            name="where_mask_to_colors",
        ),
        helper.make_node(
            "Pad",
            ["o9", "pads_to_30", "pad_value_f16"],
            ["output"],
            name="pad_to_30x30",
            mode="constant",
        ),
    ]


def save_model(filename, nodes, initializers):
    input_vi = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 10, 30, 30],
    )

    # 既存 task001.onnx analysis に合わせて FLOAT16 出力
    output_vi = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT16,
        [1, 10, 30, 30],
    )

    graph = helper.make_graph(
        nodes,
        filename.replace(".onnx", ""),
        [input_vi],
        [output_vi],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=8,
    )

    onnx.checker.check_model(model)

    out_path = os.path.join(OUT_DIR, filename)
    onnx.save(model, out_path)
    print(out_path)


def make_candidate_ct_block_only():
    """
    候補1:
    Gather/Gather/Tile/Or 全体を ConvTranspose block expand だけに置換。
    一番軽い可能性があるが、元の Tile OR Gather の Tile 成分を消すので不一致リスクあり。
    """
    inits = make_common_initializers()

    nodes = []
    nodes += common_prefix_nodes()
    nodes += [
        helper.make_node(
            "ConvTranspose",
            ["b_f16", "w_deconv_f16"],
            ["m_f16"],
            name="convtranspose_block_expand_3x",
            strides=[3, 3],
            kernel_shape=[3, 3],
        ),
        helper.make_node(
            "Cast",
            ["m_f16"],
            ["m"],
            name="cast_mask_bool",
            to=TensorProto.BOOL,
        ),
    ]
    nodes += common_suffix_nodes("m")

    save_model("task001_ct_block_only.onnx", nodes, inits)


def make_candidate_ct_block_plus_tile():
    """
    候補2:
    元の構造に近い。
    Gather/Gather を ConvTranspose に置換し、Tile と Or は残す。
    正解性は比較的高いが、削減幅は小さめ。
    """
    inits = make_common_initializers()

    nodes = []
    nodes += common_prefix_nodes()
    nodes += [
        helper.make_node(
            "Tile",
            ["b_bool", "repeats_3x"],
            ["t"],
            name="tile_periodic_3x",
        ),
        helper.make_node(
            "ConvTranspose",
            ["b_f16", "w_deconv_f16"],
            ["g_f16"],
            name="convtranspose_block_expand_3x",
            strides=[3, 3],
            kernel_shape=[3, 3],
        ),
        helper.make_node(
            "Cast",
            ["g_f16"],
            ["g"],
            name="cast_block_mask_bool",
            to=TensorProto.BOOL,
        ),
        helper.make_node(
            "Or",
            ["t", "g"],
            ["m"],
            name="or_tile_block",
        ),
    ]
    nodes += common_suffix_nodes("m")

    save_model("task001_ct_block_plus_tile.onnx", nodes, inits)


def make_candidate_tile_only():
    """
    候補3:
    Tile 成分だけ使う比較用。
    ConvTransposeなし。
    Gather/Gather側が冗長だった場合だけ通る。
    """
    inits = make_common_initializers()

    nodes = []
    nodes += common_prefix_nodes()
    nodes += [
        helper.make_node(
            "Tile",
            ["b_bool", "repeats_3x"],
            ["m"],
            name="tile_only_mask",
        ),
    ]
    nodes += common_suffix_nodes("m")

    save_model("task001_tile_only.onnx", nodes, inits)


def make_candidate_ct_block_only_f32_weight():
    """
    候補4:
    ConvTransposeを input float32 から直接実行。
    Cast b0->f16 を消せるが、weight/outputがfloat32なのでmemory/paramsは悪化する可能性あり。
    比較用。
    """
    inits = make_common_initializers()

    nodes = [
        helper.make_node(
            "Slice",
            ["input", "s0_starts", "s0_ends", "s0_axes"],
            ["b0"],
            name="slice_color0_3x3",
        ),
        helper.make_node(
            "ConvTranspose",
            ["b0", "w_deconv_f32"],
            ["m_f32"],
            name="convtranspose_block_expand_3x_f32",
            strides=[3, 3],
            kernel_shape=[3, 3],
        ),
        helper.make_node(
            "Cast",
            ["m_f32"],
            ["m"],
            name="cast_mask_bool",
            to=TensorProto.BOOL,
        ),
    ]
    nodes += common_suffix_nodes("m")

    save_model("task001_ct_block_only_f32_weight.onnx", nodes, inits)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    make_candidate_ct_block_only()
    make_candidate_ct_block_plus_tile()
    make_candidate_tile_only()
    make_candidate_ct_block_only_f32_weight()


if __name__ == "__main__":
    main()