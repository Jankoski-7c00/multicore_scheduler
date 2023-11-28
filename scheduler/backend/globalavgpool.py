import tvm
from classes.computationNode import ComputationNode
from tvm import topi, te, auto_scheduler
import numpy as np

def generate_global_avgpool(CN: ComputationNode):
    if CN.op_type != 'GlobalAveragePool':
        raise ValueError(f'Wrong op type: {CN.op_type}, expect: GlobalAveragePool.')
    
    input_tensor = CN.tensor_fm[0]
    input_shape = input_tensor.get_shape()
    input = te.placeholder(input_shape, name='input')

    output = topi.nn.global_pool(input, 'avg')

    s = te.create_schedule(output.op)
    func = tvm.build(s, [input, output], 'llvm')
    return func
    
