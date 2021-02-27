import taso as ts
import onnx
from pathlib import Path


def conv(graph, input, out_channels, kernel_h, kernel_w, stride_h, stride_w):
    weight = graph.new_weight(dims=(out_channels, input.dim(1), kernel_h, kernel_w))
    return graph.conv2d(input=input, weight=weight, strides=(stride_h, stride_w), padding="SAME")

def relu(graph, input):
    return graph.relu(input)

def pool(graph, input, kernel_h, kernel_w, stride_h, stride_w, type_='MAX'):
    if type_ == 'MAX':
        return graph.maxpool2d(input=input, kernels=(kernel_h, kernel_w), strides=(stride_h, stride_w), padding="SAME")
    elif type_ == 'AVG':
        return graph.avgpool2d(input=input, kernels=(kernel_h, kernel_w), strides=(stride_h, stride_w), padding="SAME")
    else:
        raise Exception

def concat(graph, inputs, axis):
    return graph.concat(axis, inputs)

def inceptionA(graph, input, pool_features):
    print('inceptionA')
    t1 = conv(graph, input, 64, 1, 1, 1, 1)
    t2 = conv(graph, input, 48, 1, 1, 1, 1)
    t2 = conv(graph,    t2, 64, 5, 5, 1, 1)
    t3 = conv(graph, input, 64, 1, 1, 1, 1)
    t3 = conv(graph,    t3, 96, 3, 3, 1, 1)
    t4 = pool(graph, input,  3, 3, 1, 1, 'AVG')
    t4 = conv(graph,    t4,  pool_features, 1, 1, 1, 1)
    output = concat(graph, [t1, t2, t3, t4], 1)
    return output

def inceptionB(graph, input):
    print('inceptionB')
    t1 = conv(graph, input, 384, 3, 3, 2, 2)
    t2 = conv(graph, input, 64, 1, 1, 1, 1)
    t2 = conv(graph, t2, 96, 3, 3, 1, 1)
    t2 = conv(graph, t2, 96, 3, 3, 2, 2)
    t3 = pool(graph, input, 3, 3, 2, 2)
    output = concat(graph, [t1, t2, t3], 1)
    return output

def inceptionC(graph, input, channels):
    print('inceptionC')
    t1 = conv(graph, input, 192, 1, 1, 1, 1)
    t2 = conv(graph, input, channels, 1, 1, 1, 1)
    t2 = conv(graph, t2, channels, 1, 7, 1, 1)
    t2 = conv(graph, t2, 192, 7, 1, 1, 1)
    t3 = conv(graph, input, channels, 1, 1, 1, 1)
    t3 = conv(graph, t3, channels, 7, 1, 1, 1)
    t3 = conv(graph, t3, channels, 1, 7, 1, 1)
    t3 = conv(graph, t3, channels, 7, 1, 1, 1)
    t3 = conv(graph, t3, 192, 1, 7, 1, 1)
    t4 = pool(graph, input, 3, 3, 1, 1, 'AVG')
    t4 = conv(graph, t4, 192, 1, 1, 1, 1)
    output = concat(graph, [t1, t2, t3, t4], 1)
    return output

def inceptionD(graph, input):
    print('inceptionD')
    t1 = conv(graph, input, 192, 1, 1, 1, 1)
    t1 = conv(graph, t1, 320, 3, 3, 2, 2)
    t2 = conv(graph, input, 192, 1, 1, 1, 1)
    t2 = conv(graph, t2, 192, 1, 7, 1, 1)
    t2 = conv(graph, t2, 192, 7, 1, 1, 1)
    t2 = conv(graph, t2, 192, 3, 3, 2, 2)
    t3 = pool(graph, input, 3, 3, 2, 2)
    output = concat(graph, [t1, t2, t3], 1)
    return output

def inceptionE(graph, input):
    print('inceptionE')
    t1 = conv(graph, input, 320, 1, 1, 1, 1)
    t2i = conv(graph, input, 384, 1, 1, 1, 1)
    t2 = conv(graph, t2i, 384, 1, 3, 1, 1)
    t3 = conv(graph, t2i, 384, 3, 1, 1, 1)
    t3i = conv(graph, input, 448, 1, 1, 1, 1)
    t3i = conv(graph, t3i, 384, 3, 3, 1, 1)
    t4 = conv(graph, t3i, 384, 1, 3, 1, 1)
    t5 = conv(graph, t3i, 384, 3, 1, 1, 1)
    t6 = pool(graph, input, 3, 3, 1, 1, 'AVG')
    t6 = conv(graph, t6, 192, 1, 1, 1, 1)
    output = concat(graph, [t1, t2, t3, t4, t5, t6], 1)
    return output

def inception(graph, input):
    t = conv(graph, input, 32, 3, 3, 2, 2)
    t = conv(graph, t, 32, 3, 3, 1, 1)
    t = conv(graph, t, 64, 3, 3, 1, 1)
    t = pool(graph, t, 3, 3, 2, 2)
    t = conv(graph, t, 80, 1, 1, 1, 1)
    t = conv(graph, t, 192, 3, 3, 1, 1)
    t = pool(graph, t, 3, 3, 2, 2)
    t = inceptionA(graph, t, 32)
    t = inceptionA(graph, t, 64)
    t = inceptionA(graph, t, 64)
    t = inceptionB(graph, t)
    t = inceptionC(graph, t, 128)
    t = inceptionC(graph, t, 160)
    t = inceptionC(graph, t, 160)
    t = inceptionC(graph, t, 192)
    t = inceptionD(graph, t)
    t = inceptionE(graph, t)
    t = inceptionE(graph, t)
    #print('Final pool')
    #print(t.dim(0), t.dim(1), t.dim(2), t.dim(3))
    #t = pool(graph, t, 8, 8, 1, 1, 'AVG')
    #print('Post final pool')

DEFAULT_MODEL_PATH = Path("/home/groups/aaiken/unger/models/inception")

if __name__ == "__main__":
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
    input = graph.new_input(dims=(batch_size, 3, 299, 299))
    inception(graph, input)

    unoptimized_model = ts.export_onnx(graph)
    debug_dir = None
    args.output_dir.mkdir(exist_ok=True, parents=False)
    if args.debug_dir is not None:
        debug_dir = args.debug_dir.resolve()
        debug_dir.mkdir(parents=True)
    if debug_dir is not None:
        graph.export_to_file(str(debug_dir / "unoptimized.txt").encode())
    if args.export:
        onnx.checker.check_model(unoptimized_model)
        onnx.save(unoptimized_model, str(args.output_dir / f"inception_{batch_size}_unoptimized.onnx"))
    _optimized_model = ts.optimize(graph, alpha=args.alpha, budget=args.budget, print_subst=args.print_subst)
    if debug_dir is not None:
        _optimized_model.export_to_file(str(debug_dir / "optimized.txt").encode())
    if args.export:
        optimized_model = ts.export_onnx(_optimized_model)
        onnx.save(optimized_model, str(args.output_dir / f"inception_{batch_size}_optimized.onnx"))


