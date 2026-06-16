# scripts/create_task002_strategy_b_candidates.py

from pathlib import Path
import re
import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


SRC = Path("outputs/gpt_workbench/task002/task002_holes_less_ch0_exterior.onnx")
OUT_DIR = Path("outputs/gpt_workbench/task002")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_node_by_output(model, output_name):
    for node in model.graph.node:
        if output_name in node.output:
            return node
    return None


def cleanup_dead_nodes(model):
    required = {out.name for out in model.graph.output}

    producer = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node

    live = set()
    changed = True

    while changed:
        changed = False
        for name in list(required):
            node = producer.get(name)
            if node is None:
                continue

            node_id = id(node)
            if node_id in live:
                continue

            live.add(node_id)
            changed = True

            for inp in node.input:
                if inp:
                    required.add(inp)

    old_nodes = list(model.graph.node)
    new_nodes = [node for node in old_nodes if id(node) in live]

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    return len(old_nodes) - len(new_nodes)


def used_tensor_names(model):
    used = set()

    for node in model.graph.node:
        for name in node.input:
            if name:
                used.add(name)

    for out in model.graph.output:
        used.add(out.name)

    return used


def cleanup_unused_initializers(model):
    used = used_tensor_names(model)

    old = list(model.graph.initializer)
    new = [init for init in old if init.name in used]

    del model.graph.initializer[:]
    model.graph.initializer.extend(new)

    return len(old) - len(new)


def remove_initializers_by_prefix(model, prefixes):
    kept = []
    removed = 0

    for init in model.graph.initializer:
        if any(init.name.startswith(p) for p in prefixes):
            removed += 1
        else:
            kept.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    return removed


def add_initializer(model, name, arr):
    # 同名があれば消す
    kept = [init for init in model.graph.initializer if init.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(arr, name=name))


def save_model(model, out_path):
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[warn] shape inference failed for {out_path.name}: {e}")

    onnx.checker.check_model(model)
    onnx.save(model, out_path)

    return {
        "path": str(out_path),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "filesize": out_path.stat().st_size,
    }


def get_node_index_by_output(model, output_name):
    for i, node in enumerate(model.graph.node):
        if output_name in node.output:
            return i
    raise RuntimeError(f"Cannot find node producing {output_name}")


def replace_input_name_everywhere(model, old_name, new_name):
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == old_name:
                node.input[i] = new_name


def make_kernel(kind, radius=None, k=None, orientation=None):
    """
    kind:
      - cross: Manhattan distance <= radius
      - square: full square kernel
      - hline: 1 x k
      - vline: k x 1
      - diamond_sparse: only center row/col up to radius, same as cross but explicit
    """
    if kind == "cross":
        assert radius is not None
        size = 2 * radius + 1
        arr = np.zeros((1, 1, size, size), dtype=np.float16)
        center = radius
        for y in range(size):
            for x in range(size):
                if abs(y - center) + abs(x - center) <= radius:
                    arr[0, 0, y, x] = 1.0
        return arr

    if kind == "square":
        assert radius is not None
        size = 2 * radius + 1
        return np.ones((1, 1, size, size), dtype=np.float16)

    if kind == "hline":
        assert k is not None and k % 2 == 1
        return np.ones((1, 1, 1, k), dtype=np.float16)

    if kind == "vline":
        assert k is not None and k % 2 == 1
        return np.ones((1, 1, k, 1), dtype=np.float16)

    raise ValueError(f"unknown kernel kind: {kind}")


def conv_attrs_for_kernel(arr):
    kh, kw = arr.shape[2], arr.shape[3]
    pad_top = kh // 2
    pad_bottom = kh // 2
    pad_left = kw // 2
    pad_right = kw // 2

    return {
        "kernel_shape": [kh, kw],
        "pads": [pad_top, pad_left, pad_bottom, pad_right],
        "strides": [1, 1],
    }


def build_fast_fill_nodes(model, candidate_name, steps):
    """
    steps:
      list of dicts:
        {
          "kind": "cross" / "square" / "hline" / "vline",
          "radius": int,  # cross/square
          "k": int,       # hline/vline
        }

    各 step:
      grown_sum = Conv(prev_exterior, kernel)
      grown_gt = Greater(grown_sum, zero)
      next_exterior = Where(grown_gt, bg_in, 0)

    最終 exterior を exterior_20 の代わりに使う。
    """

    new_nodes = []
    prev = "exterior_0"

    for i, step in enumerate(steps, start=1):
        kind = step["kind"]

        if kind in {"cross", "square"}:
            radius = int(step["radius"])
            kernel = make_kernel(kind=kind, radius=radius)
            suffix = f"{kind}_r{radius}"
        elif kind in {"hline", "vline"}:
            k = int(step["k"])
            kernel = make_kernel(kind=kind, k=k)
            suffix = f"{kind}_k{k}"
        else:
            raise ValueError(step)

        w_name = f"{candidate_name}_step{i}_{suffix}_w"
        add_initializer(model, w_name, kernel)

        grown_sum = f"{candidate_name}_step{i}_grown_sum"
        grown_gt = f"{candidate_name}_step{i}_grown_gt"
        out = f"{candidate_name}_exterior_{i}"

        attrs = conv_attrs_for_kernel(kernel)

        new_nodes.append(
            helper.make_node(
                "Conv",
                inputs=[prev, w_name],
                outputs=[grown_sum],
                name=f"{candidate_name}_step{i}_{suffix}_conv",
                **attrs,
            )
        )

        new_nodes.append(
            helper.make_node(
                "Greater",
                inputs=[grown_sum, "zero"],
                outputs=[grown_gt],
                name=f"{candidate_name}_step{i}_{suffix}_gt",
            )
        )

        new_nodes.append(
            helper.make_node(
                "Where",
                inputs=[grown_gt, "bg_in", "_boolmask_where_zero"],
                outputs=[out],
                name=f"{candidate_name}_step{i}_{suffix}_where",
            )
        )

        prev = out

    return new_nodes, prev


def make_candidate(candidate_name, steps):
    model = onnx.load(SRC)

    # 念のため、以前のstrategy_b initializerが混ざらないように掃除
    remove_initializers_by_prefix(model, ["sb_"])

    # exterior_0 を作るノードの直後に、新しい高速fill chainを挿入
    exterior0_idx = get_node_index_by_output(model, "exterior_0")

    fast_nodes, final_exterior = build_fast_fill_nodes(
        model=model,
        candidate_name=candidate_name,
        steps=steps,
    )

    nodes = list(model.graph.node)
    nodes[exterior0_idx + 1:exterior0_idx + 1] = fast_nodes

    del model.graph.node[:]
    model.graph.node.extend(nodes)

    # 既存の後段は exterior_20 を参照しているので、final_exterior に差し替える。
    # holes_less:
    #   holes_bool = Less(exterior_20, bg_in)
    # ch0_exterior:
    #   ch0_bool = Greater(exterior_20, zero)
    replace_input_name_everywhere(model, "exterior_20", final_exterior)

    removed_nodes = cleanup_dead_nodes(model)
    removed_inits = cleanup_unused_initializers(model)

    out_path = OUT_DIR / f"task002_{candidate_name}.onnx"
    stat = save_model(model, out_path)

    return {
        "candidate": out_path.name,
        "steps": steps,
        "final_exterior": final_exterior,
        "removed_nodes": removed_nodes,
        "removed_initializers": removed_inits,
        **stat,
    }


def repeated(kind, radius=None, k=None, n=1):
    if kind in {"cross", "square"}:
        return [{"kind": kind, "radius": radius} for _ in range(n)]
    if kind in {"hline", "vline"}:
        return [{"kind": kind, "k": k} for _ in range(n)]
    raise ValueError(kind)


def alternating_line(k, rounds):
    steps = []
    for _ in range(rounds):
        steps.append({"kind": "hline", "k": k})
        steps.append({"kind": "vline", "k": k})
    return steps


def alternating_line_reverse(k, rounds):
    steps = []
    for _ in range(rounds):
        steps.append({"kind": "vline", "k": k})
        steps.append({"kind": "hline", "k": k})
    return steps


def main():
    print(f"source: {SRC}")
    print(f"output dir: {OUT_DIR}")

    candidates = {
        # 比較的保守: 2マスずつ広げる。10段。
        # 壁を飛び越えるリスクはあるが、squareよりはcrossの方が少し安全。
        "sb_cross_r2_x10": repeated("cross", radius=2, n=10),

        # 3マスずつ。7段で21相当。
        "sb_cross_r3_x7": repeated("cross", radius=3, n=7),

        # 4マスずつ。5段で20相当。
        "sb_cross_r4_x5": repeated("cross", radius=4, n=5),

        # doubling風。かなり攻め。
        # 1,2,4,8,16相当で5段。
        "sb_cross_doubling_1_2_4_8_16": [
            {"kind": "cross", "radius": 1},
            {"kind": "cross", "radius": 2},
            {"kind": "cross", "radius": 4},
            {"kind": "cross", "radius": 8},
            {"kind": "cross", "radius": 16},
        ],

        # squareは壁を飛び越えやすいが、memory削減幅を見るための攻め候補。
        "sb_square_r2_x10": repeated("square", radius=2, n=10),
        "sb_square_r3_x7": repeated("square", radius=3, n=7),
        "sb_square_r4_x5": repeated("square", radius=4, n=5),

        # scanline風: 横→縦を繰り返す。
        # これも壁飛び越えリスクあり。ただしConv数は少なくなる。
        "sb_line_k5_hv_x5": alternating_line(k=5, rounds=5),
        "sb_line_k5_vh_x5": alternating_line_reverse(k=5, rounds=5),

        # より攻めたscanline風。
        "sb_line_k9_hv_x3": alternating_line(k=9, rounds=3),
        "sb_line_k9_vh_x3": alternating_line_reverse(k=9, rounds=3),

        # 混合: 小さめcrossで少し広げてからline。
        "sb_cross2_then_line5_x4": (
            [{"kind": "cross", "radius": 2}]
            + alternating_line(k=5, rounds=4)
        ),

        # 混合: lineで広げてからcrossで補正。
        "sb_line5_x4_then_cross2": (
            alternating_line(k=5, rounds=4)
            + [{"kind": "cross", "radius": 2}]
        ),
    }

    results = []

    for name, steps in candidates.items():
        try:
            r = make_candidate(name, steps)
            results.append(r)
            print(
                f"[ok] {r['candidate']} "
                f"nodes={r['nodes']} "
                f"inits={r['initializers']} "
                f"bytes={r['filesize']} "
                f"removed_nodes={r['removed_nodes']} "
                f"removed_inits={r['removed_initializers']}"
            )
        except Exception as e:
            print(f"[ng] {name}: {e}")

    print("\nGenerated:")
    for r in results:
        print(
            f"{r['candidate']:50s} "
            f"nodes={r['nodes']:3d} "
            f"inits={r['initializers']:2d} "
            f"bytes={r['filesize']:6d} "
            f"final={r['final_exterior']}"
        )


if __name__ == "__main__":
    main()