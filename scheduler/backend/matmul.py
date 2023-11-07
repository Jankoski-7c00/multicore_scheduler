import tvm
from tvm import te
import numpy as np
import threading
import time

def matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis = k), name = "matmul")

a = te.placeholder(shape = (32, 32), name = 'matrixA')
b = te.placeholder(shape = (32, 32), name = 'matrixB')

c = matmul(a, b)
s = te.create_schedule(c.op)

#memory hierarchy
BS = s.cache_read(b, 'shared', [c])
AS = s.cache_read(a, 'shared', [c])
AL = s.cache_read(AS, 'local', [c])
BL = s.cache_read(BS, 'local', [c])
CL= s.cache_write(c, 'local')

num_thread = 16
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")

#split and bind workload
x, y = s[c].op.axis
bx, tx = s[c].split(x, factor = num_thread)
by, ty = s[c].split(y, factor = num_thread)
s[c].reorder(bx, by, tx, ty)
s[c].bind(bx, block_x)
s[c].bind(by, block_y)
s[c].bind(tx, thread_x)
s[c].bind(ty, thread_y)

s[AS].compute_at(s[c], by)
s[BS].compute_at(s[c], by)
s[AL].compute_at(s[c], ty)
s[BL].compute_at(s[c], ty)
s[CL].compute_at(s[c], ty)

func = tvm.build(s, [a, b, c], target = 'cuda', name = 'matmul')






# 1. 创建输入数据
data_a = np.random.uniform(0, 1, (64, 64)).astype(np.float32)
data_b = np.random.uniform(0, 1, (64, 64)).astype(np.float32)


# 2. 使用TVM运行时执行生成的函数
ctx = tvm.cuda(0)  # 使用第一个GPU设备
ctx_2 = tvm.cuda(1)
a_tvm = tvm.nd.array(data_a, ctx)
b_tvm = tvm.nd.array(data_b, ctx)
c_tvm = tvm.nd.array(np.zeros((64, 64), dtype=np.float32), ctx)

d_tvm = tvm.nd.array(data_a, ctx_2)
e_tvm = tvm.nd.array(data_b, ctx_2)
f_tvm = tvm.nd.array(np.zeros((64, 64), dtype=np.float32), ctx_2)

def matmul(data_a, data_b, c_tvm, d_tvm, e_tvm, f_tvm):
    func(data_a, data_b, c_tvm)
    func(d_tvm, e_tvm, f_tvm)

thread = threading.Thread(target=matmul, args=(a_tvm, b_tvm, c_tvm, d_tvm, e_tvm, f_tvm))
start_time = time.time()
thread.start()
thread.join()
end_time = time.time()
t = end_time - start_time


#parallel
a = tvm.nd.array(data_a, ctx)
b = tvm.nd.array(data_b, ctx)
c = tvm.nd.array(np.zeros((64, 64), dtype=np.float32), ctx)

d = tvm.nd.array(data_a, ctx_2)
e = tvm.nd.array(data_b, ctx_2)
f = tvm.nd.array(np.zeros((64, 64), dtype=np.float32), ctx_2)

def matmul_on_gpu(data_a, data_b, c_tvm):
    func(data_a, data_b, c_tvm)
    
thread1 = threading.Thread(target=matmul_on_gpu, args=(a, b, c))  # GPU 0
thread2 = threading.Thread(target=matmul_on_gpu, args=(d, e, f))  # GPU 1

start_time = time.time()
thread1.start()
thread2.start()

thread1.join()
thread2.join()
ctx.sync()
ctx_2.sync()
end_time = time.time()
q = np.allclose(c.asnumpy(), c_tvm.asnumpy())
print(f'parallel :{t}')


#serial
a_tvm = tvm.nd.array(data_a, ctx)
b_tvm = tvm.nd.array(data_b, ctx)
c_tvm = tvm.nd.array(np.zeros((64, 64), dtype=np.float32), ctx)

d_tvm = tvm.nd.array(data_a, ctx_2)
e_tvm = tvm.nd.array(data_b, ctx_2)
f_tvm = tvm.nd.array(np.zeros((64, 64), dtype=np.float32), ctx_2)
thread = threading.Thread(target=matmul, args=(a_tvm, b_tvm, c_tvm, d_tvm, e_tvm, f_tvm))
start_time = time.time()
thread.start()
thread.join()
ctx.sync()
ctx_2.sync()
end_time = time.time()
t = end_time - start_time
print(f'serial :{t}')
