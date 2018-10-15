import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth(curve):
    return savgol_filter(curve, 11, 3)

savedir = './savedir/'
ext     = '.pkl'
fig, ax = plt.subplots(1,2, sharex=True)
plt.suptitle('Mean Accuracy Curve of Top 10% Evolved Models')

titles = ['Same, but with MS disjoint', '250 Hidden Neurons, 24 Motion Neurons, $MS_0=1.0$']
for fn, id, title in zip(['further_ms_disjoint', 'larger_network_higher_strength'], [0,1], titles):
    data = pickle.load(open(savedir+fn+ext, 'rb'))

    iters    = data['iter']
    task_acc = data['task_acc']
    full_acc = data['full_acc']
    loss     = data['loss']
    mut_str  = data['mut_str']

    for curve, name, color in zip([full_acc, task_acc, mut_str], \
        ['Full Accuracy', 'Task Accuracy', 'Mutation Strength'], [[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]]):

            ax[id].plot(iters, curve, c=color+[0.2])
            ax[id].plot(iters, smooth(curve), label=name, c=color)


    ax[id].grid()
    ax[id].set_xlim(iters[0], iters[-1])
    ax[id].set_ylim(0,1)
    ax[id].set_yticks(np.linspace(0,1,11))
    ax[id].set_xlabel('Iteration')
    ax[id].set_ylabel('Accuracy, Mutation Strength')

    ax[id].set_title(title)
plt.legend()
plt.show()
