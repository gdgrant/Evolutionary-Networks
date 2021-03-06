import sys, time, pickle
import itertools

import numpy as np
if len(sys.argv) > 1:
    import cupy as cp
    cp.cuda.Device(sys.argv[1]).use()
else:
    import numpy as cp


### GPU utilities

def to_gpu(x):
    """ Move numpy arrays (or dicts of arrays) to GPU """
    if type(x) == dict:
        return {k:cp.asarray(a) for (k, a) in x.items()}
    else:
        return cp.asarray(x)

def to_cpu(x):
    """ Move cupy arrays (or dicts of arrays) to CPU """
    if len(sys.argv) > 1:
        if type(x) == dict:
            return {k:cp.asnumpy(a) for (k, a) in x.items()}
        else:
            return cp.asnumpy(x)
    else:
        if type(x) == dict:
            return {k:a for (k, a) in x.items()}
        else:
            return x


### Network functions

def matmul(a, b, l=False, exception=False):
    """ Does matrix multiplication as a @ b, accounting for
        whether either is of dtype=int8.  'l' indicates whether
        this matrix operation is being used for a latency weight matrix """

    # Exception hardcoded for W_rnn_latency because it has more dims than usual
    if exception == "h_in * W_rnn_latency":

        print("STARTING MATMUL()")
        print("a.shape")
        print(a.shape)

        a = a[cp.newaxis,...]

        print("new a.shape")
        print(a.shape)

        print("b.shape")
        print(b.shape)

        input()

        print("a.nbytes")
        print(a.nbytes)

        print("b.nbytes")
        print(b.nbytes)

        input()

        print("cp.sum(a[...,cp.newaxis]*b[:,cp.newaxis,...], axis=-2, dtype=cp.float16)")
        print(cp.sum(a*b, axis=-2, dtype=cp.float16))

        print("a*b.nbytes")
        print((a*b).nbytes)

        input()

        print("a*b.shape")
        print((a*b).shape)

        input()

        print("cp.matmul(a, b).shape")
        print(cp.matmul(a, b).shape)

        input()
        return cp.matmul(a, b)
    else:
        a = a[...,0]
        b = b[:,0,:,:]
        return cp.matmul(a, b)[...,np.newaxis]

# TODO:  fix matmuls in following function
# def matmul(a, b, l=False):
#     """ Does matrix multiplication as a @ b, accounting for
#         whether either is of dtype=int8.  'l' indicates whether
#         this matrix operation is being used for a latency weight matrix """
#     if cp.int8 in [a.dtype, b.dtype]:
#         # Set up for a=state, b=weights
#         if not l:
#             return cp.sum(a[...,cp.newaxis]*b[:,cp.newaxis,...], axis=-2, dtype=cp.float16)
#         else:
#             return cp.sum(a[...,cp.newaxis]*b[:,:,cp.newaxis,...], axis=-2, dtype=cp.float16)
#     else:
#         a = a[...,0]
#         b = b[:,0,:,:]
#         return cp.matmul(a, b)

def relu(x):
    """ Performs relu on x """
    return cp.maximum(0., x, dtype=x.dtype)

def softmax(x, a=-1):
    """ Performs stable softmax on x, across the last axis by default """
    c = cp.exp(x-cp.amax(x, axis=a, keepdims=True))
    return c/cp.sum(c, axis=a, keepdims=True).astype(cp.float32)

def apply_EI(var, ei):
    """ Applies EI masking to a square variable, according to the given
        excitatory/inhibitory mask """
    return cp.matmul(relu(var), ei)

which_step = 1

def synaptic_plasticity(h, syn_x, syn_u, constants, use_stp, hidden_size):
    """ If required, applies STP updates to the hidden state and STP
        variables.  If not required, just ensures correct hidden shape. """

    # h = h[0,...] if h.ndim == 5 else False

    if use_stp:

        global which_step

        print("STARTING SYNAPTIC_PLASTICITY")
        print("running step {}".format(which_step))
        print(which_step)

        which_step = which_step+1

        print("syn_x.shape")
        print(syn_x.shape)
        print("syn_u.shape")
        print(syn_u.shape)
        print("h.shape")
        print(h.shape)

        # input()

        syn_x += constants['alpha_std']*(1-syn_x) - constants['stp_mod']*syn_u*syn_x*h
        syn_u += constants['alpha_stf']*(constants['U']-syn_u) + constants['stp_mod']*constants['U']*(1-syn_u)*h
        syn_x = cp.minimum(1., relu(syn_x))
        syn_u = cp.minimum(1., relu(syn_u))
        h_post = syn_u*syn_x*h
    else:
        h_post = h*cp.ones([1,1,hidden_size], dtype=h.dtype)

    return h_post, syn_x, syn_u


### Adaptive-Exponential spiking model

def run_adex(V, w, I, constants):
    """ Run one step of the AdEx algorithm """

    V = V.astype(cp.float32)
    w = w.astype(cp.float32)
    I = I.astype(cp.float32)/constants['current_divider']

    V_next      = adex_membrane(V, w, I, constants)
    w_next      = adex_adaptation(V, w, constants)
    V, w, spike = adex_spike(V_next, w_next, constants)

    return V.astype(cp.float32), w.astype(cp.float32), spike.astype(cp.int8)

def adex_membrane(V, w, I, c):
    """ Calculate the new membrane potential """

    term1 = I + c['g']*c['D']*cp.exp((V-c['V_T'])/c['D'])
    term2 = w + c['g']*(V-c['E'])
    return V + (c['dt']/c['C'])*(term1-term2)

def adex_adaptation(V, w, c):
    """ Calculate the new adaptation current """

    term1 = c['a']*(V-c['E'])
    term2 = w
    return w + (c['dt']/c['tau'])*(term1-term2)

def adex_spike(V, w, c):
    """ Check potential thresholds for new spikes """

    # Spike projects 0mV or 1 mV (nothing or unit value) to the network
    # with the current parameters
    spike = V > c['Vth']
    V = cp.where(spike, c['V_r'], V)
    w = cp.where(spike, w + c['b'], w)

    return V, w, spike


### Judgement functions

def cross_entropy(mask, target, output, eps=1e-16):
    """ Calculate the cross entropy loss for a rate-based network """
    mask   = mask.astype(cp.float32)
    target = target.astype(cp.float32)
    output = output.astype(cp.float32)
    return -cp.mean(mask[...,cp.newaxis]*target*cp.log(softmax(output)+eps), axis=(0,2,3)).astype(cp.float32)


### Optimization functions

def cross(var1, var2, rate):
    """ Transmit some of var2 over to var1, based on the give rate """
    return cp.where(cp.random.choice([True,False], size=var1.shape, p=[rate, 1-rate]), var1, var2)

def mutate(var, num, rate, scale, epsilon=0.):
    """ Mutates a given variable by a given rate and scale,
        generating as many offspring as num """
    #mutation_mask = cp.random.random(size=[num, *var.shape], dtype=np.float32).astype(cp.float32)
    mutation_mask = cp.random.random(size=[num, *var.shape]).astype(cp.float32)
    mutation = cp.random.normal(loc=epsilon, scale=scale, size=[num, *var.shape])
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

    return acc.astype(cp.float32)
