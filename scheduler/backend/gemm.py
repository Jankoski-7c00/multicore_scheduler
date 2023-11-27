import tvm
from classes.computationNode import ComputationNode
from tvm import topi, te, auto_scheduler
import numpy as np

def generate_gemm(CN: ComputationNode):
    if CN.op_type != 'Gemm':
        raise ValueError(f'Wrong op type: {CN.op_type}, expect: Gemm.')
    input_tensor = CN.tensor_fm[0]
    input_shape = input_tensor.get_shape()
    w_tensor = CN.tensor_w[0]
    w_shape = w_tensor.get_shape()
    if CN.has_bias == True:
        bias_tensor  = CN.tensor_w[1]
        bias_shape = bias_tensor.get_shape()

    input = te.placeholder(input_shape, name='A')
    w = te.placeholder(w_shape, name='B')
    args = [input, w]

    if CN.has_bias == True:
        bias = te.placeholder(bias_shape, name='bias')
        args.append(bias)
    else :
        bias = None
    
    output = topi.nn.matmul(input, w, bias=bias, transpose_a=CN.transA, transpose_b=CN.transB)
    args.append(output)

    s = te.create_schedule(output.op)
    func = tvm.build(s, args, 'llvm')
    return func



    