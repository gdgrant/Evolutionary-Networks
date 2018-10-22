import sys
import itertools
import pickle

import numpy as np
if len(sys.argv) > 1:
    import cupy as cp
    cp.cuda.Device(sys.argv[1]).use()
else:
    import numpy as cp


### GPU utilities

def to_gpu(x):
    """ Move numpy array(s) to GPU """
    if type(x) == dict:
        return {k:cp.asarray(a) for (k, a) in x.items()}
    else:
        return cp.asarray(x)

def to_cpu(x):
    """ Move cupy array to CPU """
    if len(sys.argv) > 1:
        return cp.asnumpy(x.astype(cp.float32))
    else:
        return x.astype(cp.float32)


### Network functions

def relu(x):
    """ Performs relu on x """
    return cp.maximum(0., x)

def softmax(x, a=-1):
    """ Performs stable softmax on x, across the last axis by default """
    c = cp.exp(x-cp.amax(x, axis=a, keepdims=True))
    return c/cp.sum(c, axis=a, keepdims=True)

def apply_EI(var, ei):
    """ Applies EI masking to a square variable, according to the given
        excitatory/inhibitory mask """
    return cp.matmul(relu(var), ei)

def synaptic_plasticity(h_out, syn_x, syn_u, constants, use_stp, hidden_size):
    """ If required, applies STP updates to the hidden state and STP
        variables.  If not required, just ensures correct hidden shape. """

    if use_stp:
        syn_x += constants['alpha_std']*(1-syn_x) - syn_u*syn_x*h_out
        syn_u += constants['alpha_stf']*(constants['U']-syn_x) - constants['U']*(1-syn_u)*h_out
        syn_x = cp.minimum(1., relu(syn_x))
        syn_u = cp.minimum(1., relu(syn_u))
        h_post = syn_u*syn_x*h_out
    else:
        h_post = h_out*cp.ones([1,1,hidden_size])

    return h_post, syn_x, syn_u


def run_adex(V, w, I, constants):

    I = I.astype(cp.float32)

    V_next      = adex_membrane(V, w, I, constants)
    w_next      = adex_adaptation(V, w, constants)
    V, w, h_out = adex_spike(V_next, w_next, constants)

    return V, w, h_out.astype(cp.float16)

def adex_membrane(V, w, I, c):

    term1 = I + c['g']*c['D']*cp.exp((V-c['V_T'])/c['D'])
    term2 = w + c['g']*(V-c['E'])
    return V + (c['dt']/c['C'])*(term1-term2)

def adex_adaptation(V, w, c):

    term1 = c['a']*(V-c['E'])
    term2 = w
    return w + (c['dt']/c['tau'])*(term1-term2)

def adex_spike(V, w, c):

    spike = V > c['Vth']
    V = cp.where(spike, c['V_r'], V)
    w = cp.where(spike, w + c['b'], w)

    return V, w, spike


### Judgement functions

def cross_entropy(mask, target, output, eps=1e-7):
    """ Calculate the cross entropy loss for a rate-based network """
    return -cp.mean(mask[...,cp.newaxis]*target*cp.log(softmax(output)+eps), axis=(0,2,3))

def cross_entropy_cpu(mask, target, output, eps=1e-7):
    """ Calculate the cross entropy loss for a rate-based network """
    return -np.mean(mask[...,np.newaxis]*target*np.log(softmax(output)+eps), axis=(0,2,3))

def spiking_cross_entropy(mask, target, output, eps=1e-7):
    """ Calculate the cross entropy loss for a spiking network """

    fixation_loss = mask[...,cp.newaxis]*(output[...,0]==1.)


### Optimization functions

def cross(var1, var2, rate):
    """ Transmit some of var2 over to var1, based on the give rate """
    return cp.where(cp.random.choice([True,False], size=var1.shape, p=[rate, 1-rate]), var1, var2)

def mutate(var, num, rate, scale):
    """ Mutates a given variable by a given rate and scale,
        generating as many offspring as num """
    mutation_mask = cp.random.random(size=[num, *var.shape], dtype=np.float32).astype(cp.float16)
    mutation = cp.random.normal(scale=scale, size=[num, *var.shape])
    return var[cp.newaxis,...] + mutation*mutation_mask


### Reporting functions

def accuracy(output, target, mask, inc_fix=False):
    """ Calculate accuracy from output, target, and mask for the networks """
    output = output.astype(cp.float32)
    target = target.astype(cp.float32)
    mask   = mask.astype(cp.float32)

    arg_output = cp.argmax(output, -1)
    arg_target = cp.argmax(target, -1)
    mask = mask if inc_fix else mask * (arg_target != 0)

    acc = cp.sum(mask * (arg_output == arg_target), axis=(0,2))/cp.sum(mask, axis=(0,2))

    return acc.astype(cp.float16)
