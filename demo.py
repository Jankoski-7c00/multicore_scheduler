import tvm
from tvm import te

# 定义矩阵大小
N = 1024
M = 1024

# 定义TVM计算
A = te.placeholder((N, M), name='A')
B = te.compute((N, M), lambda i, j: A[i, j] * 2, name='B')
C = te.compute((N, M), lambda i, j: B[i, j] * 3, name='C')

s = te.create_schedule(C.op)

# 使用compute_at进行优化
xo, xi = s[C].split(C.op.axis[0], factor=32)
yo, yi = s[C].split(C.op.axis[1], factor=32)
s[B].compute_at(s[C], yo)

print(tvm.lower(s, [A, C], simple_mode=True))