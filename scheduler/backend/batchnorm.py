from tvm import te
import tvm

def batch_norm(data: te.Tensor, weights: te.Tensor, epsilon = 1e-5):
    '''
    the second dimension od data is channel.
    weights: [mean, var, gamma, beta]
    '''
    assert len(data.shape) == 2, "The input data should be 2-D"
    bn = te.compute(data.shape, lambda i, j: ((data[i, j] - weights[0, j]) / te.sqrt(weights[1, j] + epsilon)) * weights[2, j] + weights[3, j], name = 'batch_norm')
    return bn

data = te.placeholder(shape = (32, 32), name = 'data')
weights = te.placeholder(shape = (4, 32), name = 'weights')

bn = batch_norm(data, weights)
s = te.create_schedule(bn.op)

num_thread = 16
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")

DS = s.cache_read(data, 'shared', [bn])
WS = s.cache_read(weights, 'shared', [bn])
DL = s.cache_read(DS, 'local', [bn])
WL = s.cache_read(WS, 'local', [bn])
BL = s.cache_write(bn, 'local')

x, y = s[bn].op.axis
bx, tx = s[bn].split(x, factor = num_thread)
by, ty = s[bn].split(y, factor = num_thread)
s[bn].reorder(bx, by, tx, ty)
s[bn].bind(bx, block_x)
s[bn].bind(by, block_y)
s[bn].bind(tx, thread_x)
s[bn].bind(ty, thread_y)

s[DS].compute_at(s[bn], by)
s[WS].compute_at(s[bn], by)
s[DL].compute_at(s[bn], ty)
s[WL].compute_at(s[bn], ty)
s[BL].compute_at(s[bn], ty)

func = tvm.build(s, [data, weights, bn], target = 'cuda', name = 'batchnorm')