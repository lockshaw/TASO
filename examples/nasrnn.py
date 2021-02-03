import taso as ts
import onnx
from pathlib import Path

hidden_size = 512
length = 5

PRINT = False
if not PRINT:
    print = lambda *args, **kwargs: None

def combine(graph, x, h):
    print('x = ', ' '.join(map(str, [x.dim(d) for d in range(x.nDim)])))
    print('h = ', ' '.join(map(str, [h.dim(d) for d in range(h.nDim)])))
    w1 = graph.new_weight(dims=(hidden_size, x.dim(1)))
    print('w1 = ', ' '.join(map(str, [w1.dim(d) for d in range(w1.nDim)])))
    w2 = graph.new_weight(dims=(hidden_size, h.dim(1)))
    print('w2 = ', ' '.join(map(str, [w2.dim(d) for d in range(w2.nDim)])))
    v1 = graph.matmul(x, w1)
    print('v1 = ', ' '.join(map(str, [v1.dim(d) for d in range(v1.nDim)])))
    v2 = graph.matmul(h, w2)
    print('v2 = ', ' '.join(map(str, [v2.dim(d) for d in range(v2.nDim)])))
    v = graph.add(v1, v2)
    print('v = ', ' '.join(map(str, [v.dim(d) for d in range(v.nDim)])))
    return v

def nas_node(graph, input, x):
    print('0')
    t = list()
    for i in range(8):
        t.append(combine(graph, x, input))
    midt = list()
    print('1')
    midt.append(graph.add(graph.relu(t[0]), graph.sigmoid(t[3])))
    print('2')
    midt.append(graph.add(graph.sigmoid(t[1]), graph.tanh(t[2])))
    print('3')
    midt.append(graph.mul(graph.sigmoid(t[4]), graph.tanh(t[5])))
    print('4')
    midt.append(graph.mul(graph.sigmoid(t[6]), graph.relu(t[7])))
    print('5')
    midt.append(graph.add(graph.sigmoid(midt[1]), graph.tanh(midt[2])))
    print('6')
    midt.append(graph.mul(graph.tanh(midt[0]), graph.tanh(midt[3])))
    print('7')
    midt.append(graph.mul(graph.tanh(midt[4]), graph.tanh(midt[5])))
    for m in midt:
        print(' '.join(map(str, [m.dim(d) for d in range(m.nDim)])))
    return graph.tanh(midt[6])

DEFAULT_MODEL_PATH = Path("/home/groups/aaiken/unger/models/nasrnn")

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
    print('a')
    xs = list()
    print('b')
    for i in range(length):
        xs.append(graph.new_input(dims=(batch_size, hidden_size), name=b"data"))
    print('c')
    state = graph.new_input(dims=(batch_size, hidden_size), name=b"state")
    print('d')
    for i in range(length):
        state = nas_node(graph, state, xs[i])
    print('e')
    #new_graph = taso.optimize(graph, alpha=1.0, budget=100)
    #onnx_model = taso.export_onnx(new_graph)

    #graph = ts.new_graph()
    #input = graph.new_input(dims=(batch_size, 3, 224, 224))
    #alexnet(graph, input, args.num_classes)

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
        onnx.save(unoptimized_model, str(args.output_dir / f"nasrnn_{batch_size}_unoptimized.onnx"))
    _optimized_model = ts.optimize(graph, alpha=args.alpha, budget=args.budget, print_subst=args.print_subst)
    if debug_dir is not None:
        _optimized_model.export_to_file(str(debug_dir / "optimized.txt").encode())
    if args.export:
        optimized_model = ts.export_onnx(_optimized_model)
        onnx.save(optimized_model, str(args.output_dir / f"nasrnn_{batch_size}_optimized.onnx"))

