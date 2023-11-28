import sys
sys.path.append('/Users/xiangyy/Projects/multicore_schedule/scheduler')
from frontend.onnx_parser import OnnxParser
from frontend.computationgraph import ComputationGraph
import onnx
import numpy as np
from backend.gemm import generate_gemm
from classes.computationNode import ComputationNode
import torch
import tvm
import backend

#model = onnx.load('/Users/xiangyy/Downloads/resnet50-v1-12.onnx')
model = onnx.load('/Users/xiangyy/Downloads/resnet18-v1-7.onnx')
parser = OnnxParser(model)
#print(parser.layer_graph.number_of_nodes())
cg = ComputationGraph(parser)

'''***************    test gemm    ***************'''
gemm_CN: ComputationNode = cg.split_layers_dict['resnetv15_dense0_fwd'][0]
gemm = generate_gemm(gemm_CN)
input_shape = gemm_CN.tensor_fm[0].get_shape()
w_shape = gemm_CN.tensor_w[0].get_shape()
bias_shape = gemm_CN.tensor_w[1].get_shape()

#expect results
A_data = np.random.uniform(size=input_shape).astype(np.float32)
W_data = np.random.uniform(size=w_shape).astype(np.float32)
bias_data = np.random.uniform(size=bias_shape).astype(np.float32)
A_torch = torch.tensor(A_data, dtype=torch.float32)
W_torch = torch.tensor(W_data, dtype=torch.float32)
bias_torch = torch.tensor(bias_data, dtype=torch.float32)
expect_result = torch.matmul(A_torch, W_torch.T) + bias_torch

A_tvm = tvm.nd.array(A_data)
W_tvm = tvm.nd.array(W_data)
bias_tvm = tvm.nd.array(bias_data)
out_tvm = tvm.nd.empty(gemm_CN.tensor_out[0].get_shape())
gemm(A_tvm, W_tvm, bias_tvm, out_tvm)

np.testing.assert_allclose(out_tvm.numpy(), expect_result.numpy(), rtol=1e-5)


'''***************    test add    ***************'''
add_CN: ComputationNode = cg.split_layers_dict['resnetv15_stage1__plus0'][0]
add_func = backend.generate_add(add_CN)
A_shape = add_CN.tensor_fm[0].get_shape()
B_shape = add_CN.tensor_fm[1].get_shape()

#expect results
A_np = np.random.uniform(size=A_shape).astype(np.float32)
B_np = np.random.uniform(size=B_shape).astype(np.float32)
expect_result = A_np + B_np

A_tvm = tvm.nd.array(A_np)
B_tvm = tvm.nd.array(B_np)
out_tvm = tvm.nd.empty(add_CN.tensor_out[0].get_shape())
add_func(A_tvm, B_tvm, out_tvm)

np.testing.assert_allclose(out_tvm.numpy(), expect_result, rtol=1e-5)


'''***************    test batchnorm    ***************'''
bn_CN: ComputationNode = cg.split_layers_dict['resnetv15_stage2_batchnorm3_fwd'][1]
epsilon = parser.epsilon
momentum = parser.momentum
bn_func = backend.generate_batchnorm(bn_CN, epsilon=epsilon, momentum=momentum)
input_shape = bn_CN.tensor_fm[0].get_shape()
gamma_size, beta_size, mean_size, var_size = bn_CN.tensor_w[0].get_shape()

input_np = np.random.uniform(size=input_shape).astype(np.float32)
gamma_np = np.random.uniform(size=(gamma_size,)).astype(np.float32)
beta_np = np.random.uniform(size=(beta_size,)).astype(np.float32)
mean_np = np.random.uniform(size=(mean_size,)).astype(np.float32)
var_np = np.random.uniform(size=(var_size,)).astype(np.float32)

#expect results
input_torch = torch.tensor(input_np, dtype=torch.float32)
gamma_torch = torch.tensor(gamma_np, dtype=torch.float32)
beta_torch = torch.tensor(beta_np, dtype=torch.float32)
mean_torch = torch.tensor(mean_np, dtype=torch.float32)
var_torch = torch.tensor(var_np, dtype=torch.float32)
expect_result = torch.batch_norm(input_torch, gamma_torch, beta_torch, mean_torch, var_torch, False, momentum, epsilon, False)

input_tvm = tvm.nd.array(input_np)
gamma_tvm = tvm.nd.array(gamma_np)
beta_tvm = tvm.nd.array(beta_np)
mean_tvm = tvm.nd.array(mean_np)
var_tvm = tvm.nd.array(var_np)
out_tvm = tvm.nd.empty(bn_CN.tensor_out[0].get_shape())
bn_func(input_tvm, gamma_tvm, beta_tvm, mean_tvm, var_tvm, out_tvm)

np.testing.assert_allclose(out_tvm.numpy(), expect_result.numpy(), rtol=1e-2)


'''***************    test conv    ***************'''
conv_CN: ComputationNode = cg.split_layers_dict['resnetv15_stage3_conv1_fwd'][2]
conv_func = backend.generate_conv(conv_CN)
input_shape = conv_CN.tensor_fm[0].get_shape()
kernel_shape = conv_CN.tensor_w[0].get_shape()
strides = conv_CN.strides
pad_top, pad_left, pad_bottom, pad_right = conv_CN.pads

input_np = np.random.uniform(size=input_shape).astype(np.float32)
kernel_np = np.random.uniform(size=kernel_shape).astype(np.float32)

#expect results
input_torch = torch.tensor(input_np, dtype=torch.float32)
kernel_torch = torch.tensor(kernel_np, dtype=torch.float32)
pad_input = torch.nn.functional.pad(input_torch, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
expect_result = torch.conv2d(pad_input, kernel_torch, stride=strides, padding=0)

input_tvm = tvm.nd.array(input_np)
kernel_tvm = tvm.nd.array(kernel_np)
out_tvm = tvm.nd.empty(conv_CN.tensor_out[0].get_shape())
conv_func(input_tvm, kernel_tvm, out_tvm)

np.testing.assert_allclose(out_tvm.numpy(), expect_result.numpy(), rtol=1e-5)


'''***************    test flatten    ***************'''
flatten_CN: ComputationNode = cg.split_layers_dict['flatten_170'][0]
flatten_func = backend.generate_flatten(flatten_CN)
input_shape = flatten_CN.tensor_fm[0].get_shape()

input_np = np.random.uniform(size=input_shape).astype(np.float32)

#expect results
input_torch = torch.tensor(input_np, dtype=torch.float32)
expect_result = torch.flatten(input_torch)
expect_result = expect_result.numpy()
expect_result = expect_result.reshape(1, 512)

input_tvm = tvm.nd.array(input_np)
out_tvm = tvm.nd.empty(flatten_CN.tensor_out[0].get_shape())
flatten_func(input_tvm, out_tvm)

np.testing.assert_allclose(out_tvm.numpy(), expect_result, rtol=1e-5)


'''***************    test global average pool    ***************'''
glavgpl_CN: ComputationNode = cg.split_layers_dict['resnetv15_pool1_fwd'][0]
glavgpl_func = backend.generate_global_avgpool(glavgpl_CN)
input_shape = glavgpl_CN.tensor_fm[0].get_shape()

input_np = np.random.uniform(size=input_shape).astype(np.float32)

#expect results
expect_result = np.mean(input_np, axis=(2, 3))
expect_result = expect_result.reshape(glavgpl_CN.tensor_out[0].get_shape())

input_tvm = tvm.nd.array(input_np)
out_tvm = tvm.nd.empty(glavgpl_CN.tensor_out[0].get_shape())
glavgpl_func(input_tvm, out_tvm)

np.testing.assert_allclose(out_tvm.numpy(), expect_result, rtol=1e-5)


'''***************    test maxpool    ***************'''
maxpool_CN: ComputationNode = cg.split_layers_dict['resnetv15_pool0_fwd'][0]
maxpool_func = backend.generate_maxpool(maxpool_CN)
input_shape = maxpool_CN.tensor_fm[0].get_shape()
strides = maxpool_CN.strides
pad_top, pad_left, pad_bottom, pad_right = maxpool_CN.pads
kernel_shape = maxpool_CN.kernel_shape

input_np = np.random.uniform(size=input_shape).astype(np.float32)

#expect results
input_torch = torch.tensor(input_np, dtype=torch.float32)
pad_input = torch.nn.functional.pad(input_torch, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
expect_result = torch.max_pool2d(pad_input, kernel_shape, stride=strides)

input_tvm = tvm.nd.array(input_np)
out_tvm = tvm.nd.empty(maxpool_CN.tensor_out[0].get_shape())
maxpool_func(input_tvm, out_tvm)

np.testing.assert_allclose(out_tvm.numpy(), expect_result.numpy(), rtol=1e-5)


'''***************    test relu    ***************'''
relu_CN: ComputationNode = cg.split_layers_dict['resnetv15_stage1_relu0_fwd'][2]
relu_func = backend.generate_relu(relu_CN)
input_shape = relu_CN.tensor_fm[0].get_shape()

input_np = np.random.uniform(size=input_shape).astype(np.float32)

#expect results
expect_result = np.maximum(0, input_np)

input_tvm = tvm.nd.array(input_np)
out_tvm = tvm.nd.empty(relu_CN.tensor_out[0].get_shape())
relu_func(input_tvm, out_tvm)

np.testing.assert_allclose(out_tvm.numpy(), expect_result, rtol=1e-5)
