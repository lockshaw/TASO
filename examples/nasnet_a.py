import taso as ts
import onnx
from pathlib import Path

DIR = Path(__file__).parent
OUTPUT_DIR = DIR / 'optimized_nasnet'

def squeeze(graph, out_channels, input):
    weight = graph.new_weight(dims=(out_channels, input.dim(1), 1, 1))
    return graph.conv2d(input=input, weight=weight,
                        strides=(1, 1), padding="SAME",
                        activation="RELU")

def fit(graph, current, input):
    if input.dim(2) == current.dim(2):
        return squeeze(graph, current.dim(1), input)
    else:
        weight = graph.new_weight(dims=(current.dim(1), input.dim(1), 3, 3))
        return graph.conv2d(input=input, weight=weight, strides=(2, 2), padding="SAME", activation="RELU")

def seperable_conv(graph, input, out_channels, kernels, strides, padding, activation = "NONE"):
    assert input.dim(1) % out_channels == 0, "input.dim(1)={}, out_channels={}".format(input.dim(1), out_channels)
    weight1 = graph.new_weight(dims=(out_channels, input.dim(1) // out_channels, kernels[0], kernels[1]))
    t = graph.conv2d(input=input, weight=weight1, strides=strides, padding=padding)
    weight2 = graph.new_weight(dims=(out_channels, t.dim(1), 1, 1))
    return graph.conv2d(input=t, weight=weight2, strides=(1, 1), padding="SAME", activation=activation)

def normal_cell(graph, prev, cur, out_channels):
    cur = squeeze(graph, out_channels, cur)
    prev = fit(graph, cur, prev)
    ts = list()
    ts.append(seperable_conv(graph, input=cur, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(cur)
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(seperable_conv(graph, input=cur, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(graph.avgpool2d(input=cur, kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(prev)
    ts.append(graph.avgpool2d(input=prev, kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(graph.avgpool2d(input=prev, kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    assert len(ts) == 10, "Expected 10 tensors, got {}".format(len(ts))
    outputs = list()
    for i in range(5):
        outputs.append(graph.add(ts[2*i], ts[2*i+1]))
    return graph.concat(1, outputs)

def reduction_cell(graph, prev, cur, out_channels):
    cur = squeeze(graph, out_channels, cur)
    prev = fit(graph, cur, prev)
    ts = list()
    outputs = list()
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(7,7), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=cur, out_channels=out_channels,
              kernels=(5,5), strides=(2,2), padding="SAME"))
    outputs.append(graph.add(ts[0], ts[1]))
    ts.append(graph.maxpool2d(input=cur, kernels=(3,3), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(7,7), strides=(2,2), padding="SAME"))
    outputs.append(graph.add(ts[2], ts[3]))
    ts.append(graph.avgpool2d(input=cur, kernels=(3,3), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=prev, out_channels=out_channels,
              kernels=(5,5), strides=(2,2), padding="SAME"))
    outputs.append(graph.add(ts[4], ts[5]))
    ts.append(graph.maxpool2d(input=cur, kernels=(3,3), strides=(2,2), padding="SAME"))
    ts.append(seperable_conv(graph, input=outputs[0], out_channels=out_channels,
              kernels=(3,3), strides=(1,1), padding="SAME"))
    outputs.append(graph.add(ts[6], ts[7]))
    ts.append(graph.avgpool2d(input=outputs[0], kernels=(3,3), strides=(1,1), padding="SAME"))
    ts.append(outputs[1])
    outputs.append(graph.add(ts[8], ts[9]))
    return graph.concat(1, outputs)

def dense(graph, input_tensor, output_dim, activation):
    weights = graph.new_weight(dims=(input_tensor.dim(1), output_dim))
    bias = graph.new_weight(dims=(1, output_dim))
    linear_tensor = graph.add(graph.matmul(input_tensor, weights), bias)
    if activation == "RELU":
        output_tensor = graph.relu(linear_tensor)
    else:
        raise ValueError(f"Unsupported activation {activation}")
    return output_tensor

DEFAULT_MODEL_PATH = Path("/home/groups/aaiken/unger/models/nasnet_a")

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
    p.add_argument("--normal-cells", "-n", type=int, default=5)
    args = p.parse_args()
    batch_size = args.batch_size
    graph = ts.new_graph()
    input = graph.new_input(dims=(batch_size,3,224,224))
    weight = graph.new_weight(dims=(64,3,7,7))
    input = graph.conv2d(input=input, weight=weight, strides=(2,2),
                     padding="SAME", activation="RELU")
    input = graph.maxpool2d(input=input, kernels=(3,3), strides=(2,2), padding="SAME")

    out_channels = 128
    for i in range(3):
        prev = input
        cur = input
        for j in range(args.normal_cells):
            t = normal_cell(graph, prev, cur, out_channels)
            prev = cur
            cur = t
        out_channels *= 2
        input = reduction_cell(graph, prev, cur, out_channels)
    #t = graph.relu(input)
    #t = graph.maxpool2d(t, [t.dims[2], t.dims[3]], [1, 1], [0, 0])
    #t = dense(graph, t, 1000, "RELU")
    #t = softmax(t)
    unoptimized_model = ts.export_onnx(graph)
    debug_dir = None
    if args.debug_dir is not None:
        debug_dir = args.debug_dir.resolve()
        debug_dir.mkdir(parents=True)
    if debug_dir is not None:
        graph.export_to_file(str(debug_dir / "unoptimized.txt").encode())
    if args.export:
        onnx.checker.check_model(unoptimized_model)
        onnx.save(unoptimized_model, str(args.output_dir / f"nasneta_{batch_size}_unoptimized_n{args.normal_cells}.onnx"))
    _optimized_model = ts.optimize(graph, alpha=args.alpha, budget=args.budget, print_subst=args.print_subst)
    if debug_dir is not None:
        _optimized_model.export_to_file(str(debug_dir / "optimized.txt").encode())
    if args.export:
        optimized_model = ts.export_onnx(_optimized_model)
        onnx.save(optimized_model, str(args.output_dir / f"nasneta_{batch_size}_optimized_n{args.normal_cells}.onnx"))
