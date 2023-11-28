import tvm
from classes.computationNode import ComputationNode
from tvm import topi, te, auto_scheduler
import numpy as np

def generate_batchnorm(CN: ComputationNode, epsilon = 1e-5, momentum = 0.1) :
    if CN.op_type != 'BatchNormalization':
        raise ValueError(f'Wrong op type: {CN.op_type}, expect: BatchNormalization.')
    
    input_tensor = CN.tensor_fm[0]
    w_tensor = CN.tensor_w[0]
    input_shape = input_tensor.get_shape()
    gamma_size, beta_size, mean_size, var_size = w_tensor.get_shape()

    input = te.placeholder(input_shape, name='input')
    gamma = te.placeholder((gamma_size,), name='gamma')
    beta = te.placeholder((beta_size,), name='beta')
    mean = te.placeholder((mean_size,), name='mean')
    var = te.placeholder((var_size,), name='var')
    args = [input, gamma, beta, mean, var]

    output, _, _ = topi.nn.batch_norm(input, gamma, beta, mean, var, epsilon=epsilon, momentum=momentum)
    args.append(output)
    s = te.create_schedule(output.op)

    func = tvm.build(s, args, 'llvm')
    return func