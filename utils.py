import sys
import itertools

import numpy as np
if len(sys.argv) > 1:
    import cupy as cp
    cp.cuda.Device(sys.argv[1]).use()
else:
    import numpy as cp


def to_gpu(x):
    return cp.asarray(x)

def from_gpu(x):
    return cp.asnumpy(x)

def relu(x):
    return cp.maximum(0, x)

def softmax(x, a=-1):
    c = cp.exp(x-cp.amax(x, axis=a, keepdims=True))
    return c/cp.sum(c, axis=a, keepdims=True)

def mutate(var, num, rate, scale):
    mutation_mask = cp.random.random(size=[num, *var.shape], dtype=np.float32)
    mutation = cp.random.normal(scale=scale, size=[num, *var.shape])

    return var[cp.newaxis,...] + mutation*mutation_mask
