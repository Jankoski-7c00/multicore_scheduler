from tvm import te
import tvm

def relu(data: te.Tensor) -> te.Tensor:
    assert len(data.shape) == 2, "The input data should be 2-D"
    relu = te.compute(data.shape, lambda i, j: te.max(data[i, j], 0), name = "relu")
    return relu

data = te.placeholder(shape = (32, 32), name = 'input')
r = relu(data)
s = te.create_schedule(r.op)

num_thread = 16
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")

DS = s.cache_read(data, 'shared', [r])
DL = s.cache_read(DS, 'local', [r])
RL = s.cache_write(r, 'local')

x, y = s[r].op.axis
bx, tx = s[r].split(x, factor = num_thread)
by, ty = s[r].split(y, factor = num_thread)
s[r].reorder(bx, by, tx, ty)
s[r].bind(bx, block_x)
s[r].bind(by, block_y)
s[r].bind(tx, thread_x)
s[r].bind(ty, thread_y)

s[DS].compute_at(s[r], by)
s[DL].compute_at(s[r], ty)
s[RL].compute_at(s[r], ty)

func = tvm.build(s, [data, r], target = 'cuda', name = 'relu')