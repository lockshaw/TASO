import onnx
import taso as ts
from pathlib import Path

def conv(graph, input, weight_dims, strides, padding="SAME", activation="NONE"):
    weight = graph.new_weight(dims=tuple(weight_dims))
    return graph.conv2d(input=input, weight=weight, strides=tuple(strides), padding=padding, activation=activation)

DEFAULT_MODEL_PATH = Path("/home/groups/aaiken/unger/models/nasnet_test")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--export", action="store_true", default=False)
    p.add_argument("--print-subst", action="store_true", default=False)
    p.add_argument("--budget", "-b", default=100, type=int)
    p.add_argument("--alpha", default=1.05, type=float)
    p.add_argument("--debug-dir", "-d", type=Path, default=None)
    args = p.parse_args()
    batch_size = 8

    graph = ts.new_graph()
    op688 = graph.new_input(dims=(8, 512, 14, 14))
    op708 = conv(graph, op688, (512, 1, 3, 3), (1, 1))
    op712 = conv(graph, op688, (512, 1, 3, 3), (1, 1))
    op710 = conv(graph, op708, (512, 512, 1, 1), (1, 1))
    op714 = conv(graph, op712, (512, 512, 1, 1), (1, 1))
    op719 = graph.add(op710, op714)
    final = graph.reshape(op719, (8, 512 * 14 * 14))

    unoptimized_model = ts.export_onnx(graph)
    debug_dir = None
    if args.debug_dir is not None:
        debug_dir = args.debug_dir.resolve()

    if debug_dir is not None:
        graph.export_to_file(str(debug_dir / "unoptimized.txt").encode())
    if args.export:
        #onnx.checker.check_model(unoptimized_model)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        onnx.save(unoptimized_model, str(args.output_dir / f"nasnetmini_{batch_size}_unoptimized.onnx"))
    _optimized_model = ts.optimize(graph, alpha=args.alpha, budget=args.budget, print_subst=args.print_subst)
    if debug_dir is not None:
        _optimized_model.export_to_file(str(debug_dir / "optimized.txt").encode())
    if args.export:
        optimized_model = ts.export_onnx(_optimized_model)
        onnx.save(optimized_model, str(args.output_dir / f"nasnetmini_{batch_size}_optimized.onnx"))
