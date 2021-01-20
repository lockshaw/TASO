import taso as ts
import onnx
from pathlib import Path

def resnext_block(graph, input, strides, out_channels, groups):
    w1 = graph.new_weight(dims=(out_channels,input.dim(1),1,1))
    t = graph.conv2d(input=input, weight=w1,
                     strides=(1,1), padding="SAME",
                     activation="RELU")
    w2 = graph.new_weight(dims=(out_channels,t.dim(1)//groups,3,3))
    t = graph.conv2d(input=t, weight=w2,
                     strides=strides, padding="SAME",
                     activation="RELU")
    w3 = graph.new_weight(dims=(2*out_channels,t.dim(1),1,1))
    t = graph.conv2d(input=t, weight=w3,
                     strides=(1,1), padding="SAME")
    if (strides[0]>1) or (input.dim(1) != out_channels*2):
        w4 = graph.new_weight(dims=(out_channels*2,input.dim(1),1,1))
        input=graph.conv2d(input=input, weight=w4,
                           strides=strides, padding="SAME",
                           activation="RELU")
    return graph.relu(graph.add(input, t))

DEFAULT_MODEL_PATH = Path("/home/groups/aaiken/unger/models/resnext50")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("batch_size", type=int)
    p.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--export", action="store_true", default=False)
    p.add_argument("--print-subst", action="store_true", default=False)
    p.add_argument("--budget", "-b", default=100, type=int)
    p.add_argument("--alpha", default=1.05, type=float)
    p.add_argument("--debug-dir", "-d", type=Path, default=None)
    args = p.parse_args()
    batch_size = args.batch_size

    graph = ts.new_graph()
    input = graph.new_input(dims=(batch_size,3,224,224))
    weight = graph.new_weight(dims=(64,3,7,7))
    t = graph.conv2d(input=input, weight=weight, strides=(2,2),
                     padding="SAME", activation="RELU")
    t = graph.maxpool2d(input=t, kernels=(3,3), strides=(2,2), padding="SAME")
    for i in range(3):
        t = resnext_block(graph, t, (1,1), 128, 32)
    strides = (2,2)
    for i in range(4):
        t = resnext_block(graph, t, strides, 256, 32)
        strides = (1,1)
    strides = (2,2)
    for i in range(6):
        t = resnext_block(graph, t, strides, 512, 32)
        strides = (1,1)
    strides = (2,2)
    for i in range(3):
        t = resnext_block(graph, t, strides, 1024, 32)
        strides = (1,1)

    unoptimized_model = ts.export_onnx(graph)
    debug_dir = None
    if args.debug_dir is not None:
        debug_dir = args.debug_dir.resolve()
        debug_dir.mkdir(parents=True)
    if debug_dir is not None:
        graph.export_to_file(str(debug_dir / "unoptimized.txt").encode())
    if args.export:
        onnx.checker.check_model(unoptimized_model)
        onnx.save(unoptimized_model, str(args.output_dir / f"resnext50_{batch_size}_unoptimized.onnx"))
    _optimized_model = ts.optimize(graph, alpha=args.alpha, budget=args.budget, print_subst=args.print_subst)
    if debug_dir is not None:
        _optimized_model.export_to_file(str(debug_dir / "optimized.txt").encode())
    if args.export:
        optimized_model = ts.export_onnx(_optimized_model)
        onnx.save(optimized_model, str(args.output_dir / f"resnext50_{batch_size}_optimized.onnx"))
    # new_graph = ts.optimize(graph, alpha=args.alpha, budget=args.budget)
    # onnx_model = ts.export_onnx(new_graph)
    # onnx.checker.check_model(onnx_model)
    # onnx.save(onnx_model, "resnext50_xflow.onnx")
