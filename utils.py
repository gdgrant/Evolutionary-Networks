import sys
import itertools

import numpy as np
if len(sys.argv) > 1:
    import cupy as cp
    cp.cuda.Device(sys.argv[1]).use()
else:
    import numpy as cp


def to_gpu(x):
    """ Move numpy array to GPU """
    return cp.asarray(x)

def to_cpu(x):
    """ Move cupy array to CPU """
    return cp.asnumpy(x)

def relu(x):
    """ Performs relu on x """
    return cp.maximum(0., x)

def softmax(x, a=-1):
    """ Performs softmax on x, across the last axis by default """
    c = cp.exp(x-cp.amax(x, axis=a, keepdims=True))
    return c/cp.sum(c, axis=a, keepdims=True)

def mutate(var, num, rate, scale):
    """ Mutates a given variable by a given rate and scale,
        generating as many offspring as num """
    mutation_mask = cp.random.random(size=[num, *var.shape], dtype=np.float32)
    mutation = cp.random.normal(scale=scale, size=[num, *var.shape])
    return var[cp.newaxis,...] + mutation*mutation_mask

def accuracy(output, target, mask):
    """ Calculate accuracy from output, target, and mask for the networks """
    arg_output = cp.argmax(output, -1)
    arg_target = cp.argmax(target, -1)
    mask = mask * (arg_target != 0)
    return cp.sum(mask * (arg_output == arg_target), axis=(0,2))/cp.sum(mask, axis=(0,2))
