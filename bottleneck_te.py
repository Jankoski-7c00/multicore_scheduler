from tvm import te
import tvm

#假设img2col以及对应的weights降维已经做好，处理的数据为2D
def matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis = k), name = "matmul")

def batch_norm(data: te.Tensor, mean: te.Tensor, var: te.Tensor,gamma: te.Tensor, beta: te.Tensor, epsilon = 1e-5):
    assert len(data.shape) == 2, "The input data should be 2-D"

    #height, channel = data.shape

    #h_axis_1 = te.reduce_axis((0, height), name='h_axis_1')
    #h_axis_2 = te.reduce_axis((0, height), name='h_axis_2')

    #sum_data = te.compute((channel,), lambda c: te.sum(data[h_axis_1, c], axis = h_axis_1), name='sum_data')
    #mean = te.compute((channel,), lambda c: sum_data[c] / height, name='mean')

    #sum_squared_diff = te.compute((channel,), lambda c: te.sum(tvm.te.power(data[h_axis_2, c] - mean[c], 2), axis = h_axis_2), name='sum_squared_diff')
    #sum_squared_diff = te.compute((channel,), lambda c: te.sum(tvm.te.multiply(data[h_axis_2, c] - mean[c], data[h_axis_2, c] - mean[c]), axis = h_axis_2), name='sum_squared_diff')
    #var = te.compute((channel,), lambda c: sum_squared_diff[c] / height, name='var')

    bn = te.compute(data.shape, lambda i, j: ((data[i, j] - mean[j]) / te.sqrt(var[j] + epsilon)) * gamma[j] + beta[j], name = 'batch normalization')

    #bn_out = te.compute(data.shape, lambda i, j: normalized_data[i, j] * gamma[j] + beta[j], name = 'bn_out')

    return bn

def relu(data: te.Tensor) -> te.Tensor:
    assert len(data.shape) == 2, "The input data should be 2-D"
    relu = te.compute(data.shape, lambda i, j: te.max(data[i, j], 0), name = "relu")
    return relu

#reshape对应不同层间的数据整理，应交由加速器相应硬件去做，这里不代表实际数据排布情况，只用于说明数据排布会有转换
def reshape(data: te.Tensor, next_kernel_size: tuple):
    assert len(data.shape) == 2, "The input data should be 2-D"
    height, width = data.shape
    out_height = height
    out_width = width * next_kernel_size[0] * next_kernel_size[1]

    output = te.compute((out_height,out_width), lambda i, j: data[i, j % (next_kernel_size[0] * next_kernel_size[1])], name = "reshape")
    return output

#***********************    Bottleneck    ***********************#

#prepare data
input = te.placeholder(shape = (56*56, 64), name = "input")
weights_0 = te.placeholder(shape = (64, 64), name = "weights_0")
weights_1 = te.placeholder(shape = (3*3*64, 64), name = "weights_1")
weights_2 = te.placeholder(shape = (64, 256), name = "weights_2")
weights_shortcut = te.placeholder(shape = (64, 256), name = "weights_shortcut")

mean_0 = te.placeholder(shape = (64,), name = "mean_0")
mean_1 = te.placeholder(shape = (64,), name = "mean_1")
mean_2 = te.placeholder(shape = (64,), name = "mean_2")
mean_shortcut = te.placeholder(shape = (64,), name = "mean_shortcut")

var_0 = te.placeholder(shape = (64,), name = "var_0")
var_1 = te.placeholder(shape = (64,), name = "var_1")
var_2 = te.placeholder(shape = (64,), name = "var_2")
var_shortcut = te.placeholder(shape = (64,), name = "var_shortcut")

gamma_0 = te.placeholder(shape = (64,), name = "gamma_0")
gamma_1 = te.placeholder(shape = (64,), name = "gamma_1")
gamma_2 = te.placeholder(shape = (64,), name = "gamma_2")
gamma_shortcut = te.placeholder(shape = (64,), name = "gamma_shortcut")

beta_0 = te.placeholder(shape = (64,), name = "beta_0")
beta_1 = te.placeholder(shape = (64,), name = "beta_1")
beta_2 = te.placeholder(shape = (64,), name = "beta_2")
beta_shortcut = te.placeholder(shape = (64,), name = "beta_shortcut")

#layer0
layer0_matmul = matmul(input, weights_0)
layer0_bn = batch_norm(layer0_matmul, mean_0, var_0, gamma_0, beta_0)
layer0_relu = relu(layer0_bn)

#layer1
layer1 = reshape(layer0_relu, (3, 3))
layer1_matmul = matmul(layer1, weights_1)
layer1_bn = batch_norm(layer1_matmul, mean_1, var_1, gamma_1, beta_1)
layer1_relu = relu(layer1_bn)

#layer2
layer2_matmul = matmul(layer1_relu, weights_2)
layer2_bn = batch_norm(layer2_matmul, mean_2, var_2, gamma_2, beta_2)

#shortcut
shortcut_matmul = matmul(input, weights_shortcut)
shortcut_bn = batch_norm(shortcut_matmul, mean_shortcut, var_shortcut, gamma_shortcut, beta_shortcut)

#output
output_add = te.compute(layer2_bn.shape, lambda i, j: layer2_bn[i, j] + shortcut_bn[i, j], name = "shortcut_sum")
output = relu(output_add)


# Compile the function with TVM
sch = te.create_schedule(output.op)

'''
#layer0的matmul拆分
matmul_op = layer0_matmul.op
n, m = matmul_op.axis
k, = matmul_op.reduce_axis

no, ni = sch[layer0_matmul].split(n, factor=16)
mo, mi = sch[layer0_matmul].split(m, factor=16)
ko, ki = sch[layer0_matmul].split(k, factor=16)

sch[layer0_matmul].reorder(no, mo, ko, ni, mi, ki)

#layer1的matmul拆分
matmul_op = layer1_matmul.op
n, m = matmul_op.axis
k, = matmul_op.reduce_axis

no, ni = sch[layer1_matmul].split(n, factor=16)
mo, mi = sch[layer1_matmul].split(m, factor=16)
ko, ki = sch[layer1_matmul].split(k, factor=16)

sch[layer1_matmul].reorder(no, mo, ko, ni, mi, ki)

#layer2的matmul拆分
matmul_op = layer2_matmul.op
n, m = matmul_op.axis
k, = matmul_op.reduce_axis

no, ni = sch[layer2_matmul].split(n, factor=16)
mo, mi = sch[layer2_matmul].split(m, factor=16)
ko, ki = sch[layer2_matmul].split(k, factor=16)

sch[layer2_matmul].reorder(no, mo, ko, ni, mi, ki)

#shortcut的matmul拆分
matmul_op = shortcut_matmul.op
n, m = matmul_op.axis
k, = matmul_op.reduce_axis

no, ni = sch[shortcut_matmul].split(n, factor=16)
mo, mi = sch[shortcut_matmul].split(m, factor=16)
ko, ki = sch[shortcut_matmul].split(k, factor=16)

sch[shortcut_matmul].reorder(no, mo, ko, ni, mi, ki)
'''

#算子拆分
matmul = [layer0_matmul, layer1_matmul, layer2_matmul, shortcut_matmul]
bn = [layer0_bn, layer1_bn, layer2_bn, shortcut_bn]
relu = [layer0_relu, layer1_relu, output]

#保存拆分后的axis，以便于进行算子融合
matmul_axis = []
bn_axis = []
relu_axis = []

#拆分matmul
for x in range(4):
    matmul_op = matmul[x].op
    n, m = matmul_op.axis
    k, = matmul_op.reduce_axis
    
    no, ni = sch[matmul[x]].split(n, factor=16)
    mo, mi = sch[matmul[x]].split(m, factor=16)
    #sch[x].tile(n, m, 16, 16)
    ko, ki = sch[matmul[x]].split(k, factor=16)
    sch[matmul[x]].reorder(no, mo, ko, ni, mi, ki)
    matmul_axis.append(mo)

#拆分bn
for x in range(4):
    bn_op = bn[x].op
    n, m = bn_op.axis
    
    no, ni = sch[bn[x]].split(n, factor=16)
    mo, mi = sch[bn[x]].split(m, factor=16)
    sch[bn[x]].reorder(no, mo, ni, mi)
    #sch[x].tile(n, m, 16, 16)
    bn_axis.append(mo)

#拆分relu
for x in range(3):
    relu_op = relu[x].op
    n, m = relu_op.axis

    no, ni = sch[relu[x]].split(n, factor=16)
    mo, mi = sch[relu[x]].split(m, factor=16)
    sch[relu[x]].reorder(no, mo, ni, mi)
    relu_axis.append(mo)
    
#最后一层的加法拆分
n, m = output_add.op.axis
no_, ni_ = sch[output_add].split(n, factor=16)
mo_, mi_ = sch[output_add].split(m, factor=16)
sch[output_add].reorder(no_, mo_, ni_, mi_)

#reshape拆分
n, m = layer1.op.axis
no_1, ni_1 = sch[layer1].split(n, factor=16)
mo_1, mi_1 = sch[layer1].split(m, factor=16)
sch[layer1].reorder(no_1, mo_1, ni_1, mi_1)

#算子融合
#bn和matmul融合
for i in range(4):
    sch[matmul[i]].compute_at(sch[bn[i]], bn_axis[i])

#bn和relu算子融合
for i in range(2):
    sch[bn[i]].compute_at(sch[relu[i]], relu_axis[i])


#inter layer fusion
sch[layer0_relu].compute_at(sch[layer1], mo_1)
sch[layer1].compute_at(sch[layer1_matmul], matmul_axis[1])
sch[layer1_relu].compute_at(sch[layer2_matmul], matmul_axis[2])

#shortcut层后的单独处理
sch[shortcut_bn].compute_at(sch[output_add], mo_)
sch[layer2_bn].compute_at(sch[output_add], mo_)
sch[output_add].compute_at(sch[output], relu_axis[2])

print(tvm.lower(sch, [input, weights_0, weights_1, weights_2, weights_shortcut,
                      mean_0, mean_1, mean_2, mean_shortcut,
                      var_0, var_1, var_2, var_shortcut,
                      gamma_0, gamma_1, gamma_2, gamma_shortcut,
                      beta_0, beta_1, beta_2, beta_shortcut, output], simple_mode = True))