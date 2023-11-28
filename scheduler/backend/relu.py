import tvm
from classes.computationNode import ComputationNode
from tvm import topi, te, auto_scheduler
import numpy as np

def generate_relu(CN: ComputationNode) :
    if CN.op_type != 'Relu':
        raise ValueError(f'Wrong op type: {CN.op_type}, expect: Relu.')
    
    input_tenor = CN.tensor_fm[0]
    input_shape = input_tenor.get_shape()

    input = te.placeholder(input_shape, name='input')
    output = topi.nn.relu(input)

    s = te.create_schedule(output.op)
    func = tvm.build(s, [input, output], 'llvm')
    return func
