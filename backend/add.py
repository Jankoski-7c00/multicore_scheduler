from tvm import te
import tvm

def add(a: te.Tensor, b: te.Tensor) -> te.Tensor:
    assert len(a.shape) == 2 and len(b.shape) == 2, "The input data should be 2-D"
    #assert a.shape == b.shape, 'The shape of two matrix does not match.'
    add = te.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name = 'add')
    return add

a = te.placeholder(shape = (32, 32), name = 'MatrixA')
b = te.placeholder(shape = (32, 32), name = 'MattixB')
sum = add(a, b)
s = te.create_schedule(sum.op)

num_thread = 16
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")

AS = s.cache_read(a, 'shared', [sum])
AL = s.cache_read(AS, 'local', [sum])
BS = s.cache_read(b, 'shared', [sum])
BL = s.cache_read(BS, 'local', [sum])
SL = s.cache_write(sum, 'local')

x, y = s[sum].op.axis
bx, tx = s[sum].split(x, factor = num_thread)
by, ty = s[sum].split(y, factor = num_thread)
s[sum].reorder(bx, by, tx, ty)
s[sum].bind(bx, block_x)
s[sum].bind(by, block_y)
s[sum].bind(tx, thread_x)
s[sum].bind(ty, thread_y)

s[AS].compute_at(s[sum], by)
s[BS].compute_at(s[sum], by)
s[AL].compute_at(s[sum], ty)
s[BL].compute_at(s[sum], ty)
s[SL].compute_at(s[sum], ty)

func = tvm.build(s, [a, b, sum], target = 'cuda', name = 'add')