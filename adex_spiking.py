import numpy as np
import matplotlib.pyplot as plt


"""
C   = 281e-12
g   = 30e-9
E   = -70.6e-3
V_T = -50.4e-3
D   = 2e-3
a   = 4e-9
tau = 144e-3
b   = 80.5e-12
V_r = -70.6e-3
dt  = 1e-3
#"""

"""
# cNA Inhibitory Neuron
C   = 59e-12
g   = 2.9e-9
E   = -62e-3
V_T = -42e-3
D   = 3e-3
a   = 1.8e-9
tau = 16e-3
b   = 61e-12
V_r = -54e-3
dt  = 1e-3
#"""

#"""
# RS Excitatory Neuron
C   = 104e-12
g   = 4.3e-9
E   = -65e-3
V_T = -52e-3
D   = 0.8e-3
a   = -0.8e-9
tau = 88e-3
b   = 65e-12
V_r = -53e-3
dt  = 1e-3
#"""



def membrane(V, w, I):

    term1 = I + g*D*np.exp((V-V_T)/D)
    term2 = w + g*(V-E)

    dV = (dt/C)*(term1-term2)

    return V + dV


def adaptation(V, w):

    term1 = a*(V-E)
    term2 = w

    dw = (dt/tau)*(term1-term2)

    return w + dw


def spike(V, w):

    if V > 20e-3:
        return V_r, w + b
    else:
        return V, w


V = 21e-3
w = 2e-12
I = 1e-9
I = 0.1e-9

V_record = [V]
w_record = [w]
I_record = [0]

max_t = 10000
for t in range(max_t):
    I_t = I*(t/max_t)**2
    V_next = membrane(V, w, I_t)
    w_next = adaptation(V, w)
    V, w = spike(V_next, w_next)

    V_record.append(V)
    w_record.append(w)
    I_record.append(I_t)





fig, ax = plt.subplots(1,3, sharex=True)

ax[0].plot(I_record, label='V')
ax[0].set_xlabel('ms')
ax[0].set_ylabel('Input Current')

ax[1].plot(V_record, label='V')
ax[1].set_xlabel('ms')
ax[1].set_ylabel('Membrane Voltage')

ax[2].plot(w_record, label='w')
ax[2].set_xlabel('ms')
ax[2].set_ylabel('Adaptation Current')

plt.show()
