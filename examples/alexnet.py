import taso as ts
import onnx
from pathlib import Path

def conv(graph, input, out_channels, kernel_size, stride):
    weight = graph.new_weight(dims=(out_channels, input.dim(1), kernel_size, kernel_size))
    return graph.conv2d(input=input, weight=weight, strides=(stride, stride), padding="SAME")

def relu(graph, input):
    return graph.relu(input)

def maxpool2d(graph, input, kernel_size, stride):
    return graph.maxpool2d(input=input, kernels=(kernel_size, kernel_size), strides=(stride, stride), padding="SAME")

def linear(graph, input_tensor, output_dim, use_bias=False):
    assert use_bias == False
    weights = graph.new_weight(dims=(input_tensor.dim(1), output_dim))
    product = graph.matmul(input_tensor, weights)
    return product

def features(graph, input):
    input = conv(graph, input, out_channels=64, kernel_size=11, stride=4)
    input = relu(graph, input)
    input = maxpool2d(graph, input, kernel_size=3, stride=2)

    input = conv(graph, input, out_channels=192, kernel_size=5, stride=1)
    input = relu(graph, input)
    input = maxpool2d(graph, input, kernel_size=3, stride=2)

    input = conv(graph, input, out_channels=384, kernel_size=3, stride=1)
    input = relu(graph, input)

    input = conv(graph, input, out_channels=256, kernel_size=3, stride=1)
    input = relu(graph, input)

    input = conv(graph, input, out_channels=256, kernel_size=3, stride=1)
    input = relu(graph, input)

    input = maxpool2d(graph, input, kernel_size=3, stride=2)

    return input

def classifier(graph, input, num_classes):
    input = linear(graph, input, 4096)
    # input = relu(graph, input)
    # input = linear(graph, input, 4096)
    # input = relu(graph, input)
    # input = linear(graph, input, num_classes)
    # return input

def alexnet(graph, input, num_classes):
    print(input.dim(0), input.dim(1), input.dim(2), input.dim(3))
    input = features(graph, input)
    print(input.dim(0), input.dim(1), input.dim(2), input.dim(3))
    input = graph.reshape(input, (input.dim(0), 12544))
    print(input.dim(0), input.dim(1))
    #input = classifier(graph, input, num_classes)
    return input

DEFAULT_MODEL_PATH = Path("/home/groups/aaiken/unger/models/alexnet")

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
    p.add_argument("--num-classes", "-n", type=int, default=10)
    args = p.parse_args()
    batch_size = args.batch_size

    graph = ts.new_graph()
    input = graph.new_input(dims=(batch_size, 3, 224, 224))
    alexnet(graph, input, args.num_classes)

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
        onnx.save(unoptimized_model, str(args.output_dir / f"alexnet_{batch_size}_unoptimized_n{args.num_classes}.onnx"))
    _optimized_model = ts.optimize(graph, alpha=args.alpha, budget=args.budget, print_subst=args.print_subst)
    if debug_dir is not None:
        _optimized_model.export_to_file(str(debug_dir / "optimized.txt").encode())
    if args.export:
        optimized_model = ts.export_onnx(_optimized_model)
        onnx.save(optimized_model, str(args.output_dir / f"alexnet_{batch_size}_optimized_n{args.num_classes}.onnx"))

