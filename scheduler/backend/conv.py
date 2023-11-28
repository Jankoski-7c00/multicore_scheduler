import tvm
from classes.computationNode import ComputationNode
from tvm import topi, te, auto_scheduler
import numpy as np

def generate_conv(CN: ComputationNode) :
    if CN.op_type != 'Conv':
        raise ValueError(f'Wrong op type: {CN.op_type}, expect: Conv.')
    input_tensor = CN.tensor_fm[0]
    input_shape = input_tensor.get_shape()
    kernel_tensor = CN.tensor_w[0]
    kernel_shape = kernel_tensor.get_shape()
    if CN.has_bias == True:
        bias_tensor  = CN.tensor_w[1]
        bias_shape = bias_tensor.get_shape()
    pads = CN.pads
    strides = CN.strides

    input = te.placeholder(input_shape, name='input')
    kernel = te.placeholder(kernel_shape, name='kernel')
    args = [input, kernel]

    if CN.has_bias == True:
        bias = te.placeholder(bias_shape, name='bias')
        args.append(bias)
    else:
        bias = None

    output = topi.nn.conv2d(input, kernel, strides, pads, 1)
    args.append(output)

    s = te.create_schedule(output.op)
    func = tvm.build(s, args, 'llvm')
    return func


    
