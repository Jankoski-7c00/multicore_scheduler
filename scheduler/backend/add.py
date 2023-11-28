import tvm
from classes.computationNode import ComputationNode
from tvm import topi, te, auto_scheduler
import numpy as np

def generate_add(CN: ComputationNode) :
    if CN.op_type != 'Add':
        raise ValueError(f'Wrong op type: {CN.op_type}, expect: Add.')
    
    A_tensor = CN.tensor_fm[0]
    B_tensor = CN.tensor_fm[1]
    A_shape = A_tensor.get_shape()
    B_shape = B_tensor.get_shape()

    A = te.placeholder(A_shape, name='A')
    B = te.placeholder(B_shape, name='B')
    output = topi.add(A, B)

    s = te.create_schedule(output.op)
    func = tvm.build(s, [A, B, output], 'llvm')
    return func