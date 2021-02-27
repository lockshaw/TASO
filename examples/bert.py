import taso as ts
import onnx
from pathlib import Path

DIR = Path(__file__).parent
OUTPUT_DIR = DIR / 'optimized'

seq_length = 64
hidden_dims = 1024

def print_shape(tag, x):
    print(tag, ' '.join(str(x.dim(i)) for i in range(x.nDim)))

def attention(graph, input, heads):
    d_model = input.dim(1)
    d_k = d_model // heads
    assert input.dim(1) % heads == 0
    weights = list()
    for i in range(3):
        weights.append(graph.new_weight(dims=(d_model, d_model)))
    # compute query, key, value tensors
    q = graph.matmul(input, weights[0])
    k = graph.matmul(input, weights[1])
    v = graph.matmul(input, weights[2])
    print_shape('q', q)
    print_shape('k', k)
    print_shape('v', v)
    print_shape('input', input)
    # reshape query, key, value to multiple heads
    q = graph.reshape(q, shape=(input.dim(0),heads,64))
    assert input.dim(0) * heads * 64 == input.dim(0) * input.dim(1)
    k = graph.reshape(k, shape=(input.dim(0),heads,64))
    v = graph.reshape(v, shape=(input.dim(0),heads,64))
    # transpose query, key, value for batched matmul
    q = graph.transpose(q, perm=(1,0,2), shuffle=True)
    k = graph.transpose(k, perm=(1,2,0), shuffle=True)
    v = graph.transpose(v, perm=(1,0,2), shuffle=True)
    # perform matrix multiplications
    print_shape('k', k)
    print_shape('q', q)
    logits = graph.matmul(q, k)
    print_shape('logits', logits)
    output = graph.matmul(logits, v)
    # transpose the output back
    output = graph.transpose(output,perm=(1,0,2), shuffle=True)
    print_shape('output', output)
    output = graph.reshape(output, shape=(input.dim(0), 1024))

    # a final linear layer
    linear = graph.new_weight(dims=(d_model, d_model))
    output = graph.matmul(input, linear)
    return output

DEFAULT_MODEL_PATH = Path("/home/groups/aaiken/unger/models/bert")

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
    p.add_argument("--ff-budget", default=0, type=int)
    p.add_argument("--gpus", type=int, required=True)
    args = p.parse_args()
    batch_size = args.batch_size
    ff_budget = args.ff_budget

    graph = ts.new_graph()
    input = graph.new_input(dims=(batch_size, hidden_dims))
    input = graph.relu(input)
    t = input
    for i in range(8):
        t = attention(graph, t, 16)

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
        onnx.save(unoptimized_model, str(args.output_dir / f"bert-{args.budget}x{ff_budget}_{batch_size}_unoptimized_alpha{str(args.alpha).replace('.', 'p')}_g{args.gpus}.onnx"))
    _old_optimized_model = ts.optimize(graph, alpha=args.alpha, budget=args.budget, ff_budget=0, num_gpus=1, print_subst=args.print_subst)
    _optimized_model = ts.optimize(graph, alpha=args.alpha, budget=args.budget, ff_budget=ff_budget, num_gpus=args.gpus, print_subst=args.print_subst)
    if debug_dir is not None:
        _optimized_model.export_to_file(str(debug_dir / "optimized.txt").encode())
    if args.export:
        optimized_model = ts.export_onnx(_optimized_model)
        onnx.save(optimized_model, str(args.output_dir / f"bert-{args.budget}x{ff_budget}_{batch_size}_optimized_alpha{str(args.alpha).replace('.', 'p')}_g{args.gpus}.onnx"))
        old_optimized_model = ts.export_onnx(_old_optimized_model)
        onnx.save(optimized_model, str(args.output_dir / f"bert-{args.budget}x{ff_budget}_{batch_size}_old-optimized_alpha{str(args.alpha).replace('.', 'p')}_g{args.gpus}.onnx"))


#    for i, new_graph in enumerate(ts.optimize_multi(graph, alpha=1.1, budget=600, numResults=10)):
#        print(new_graph.cost(), new_graph.hash())
#        onnx_model = ts.export_onnx(new_graph)
#        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
#        onnx.save(onnx_model, OUTPUT_DIR / f'variant_{i}.onnx')
#    #new_graph = ts.optimize(graph, alpha=1.0, budget=100)
