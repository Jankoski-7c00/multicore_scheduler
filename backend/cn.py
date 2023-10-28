import tvm
from tvm import te
import numpy as np

def matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis = k), name = "matmul")

def batch_norm(data: te.Tensor, mean: te.Tensor, var: te.Tensor,gamma: te.Tensor, beta: te.Tensor, epsilon = 1e-5):
    assert len(data.shape) == 2, "The input data should be 2-D"

    bn = te.compute(data.shape, lambda i, j: ((data[i, j] - mean[j]) / te.sqrt(var[j] + epsilon)) * gamma[j] + beta[j], name = 'batch normalization')

    return bn

def relu(data: te.Tensor) -> te.Tensor:
    assert len(data.shape) == 2, "The input data should be 2-D"
    relu = te.compute(data.shape, lambda i, j: te.max(data[i, j], 0), name = "relu")
    return relu

a = te.placeholder(shape = (64, 64), name = 'matrix A')
b = te.placeholder(shape = (64, 64), name = 'matrix B')

c = matmul(a, b)
s = te.create_schedule(c.op)
block_x = te.thread_axis("blockIdx.x")
thread_x = te.thread_axis("threadIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_y = te.thread_axis("threadIdx.y")

bx, tx = s[c].split(c.op.axis[0], factor=32)
by, ty = s[c].split(c.op.axis[1], factor=32)
s[c].bind(bx, block_x)
s[c].bind(tx, thread_x)
s[c].bind(by, block_y)
s[c].bind(ty, thread_y)

mod = tvm.build(s, [a, b, c], 'cuda' ,name = 'matmul')