import tvm
from classes.computationNode import ComputationNode
from tvm import topi, te, auto_scheduler
import numpy as np

def generate_maxpool(CN: ComputationNode) :
    if CN.op_type != 'MaxPool':
        raise ValueError(f'Wrong op type: {CN.op_type}, expect: MaxPool.')
    
    input_tensor = CN.tensor_fm[0]
    input_shape = input_tensor.get_shape()
    pads = CN.pads
    strides = CN.strides
    kernel_shape = CN.kernel_shape

    input = te.placeholder(input_shape, name='input')

    output = topi.nn.pool2d(input, kernel_shape, strides, (1, 1), pads, 'max')
    s = te.create_schedule(output.op)
    func = tvm.build(s, [input, output], 'llvm')
    return func