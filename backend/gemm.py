import tvm
from tvm import te

def gemm(input: te.Tensor, weights: te.Tensor, bias: te.Tensor) -> te.Tensor:
    assert input.shape[1] == weights.shape[0]
    n = input.shape[0]
    m = weights.shape[1]
    k = te.reduce_axis((0, input.shape[1]), name="reduce")
    return te.compute((n, m), lambda i, j: te.sum(input[i, k] * weights[k, j], axis = k) + bias[j], name = "gemm")