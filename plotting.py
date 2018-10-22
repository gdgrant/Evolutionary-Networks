import numpy as np
import os, pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth(curve, size=11):
    return savgol_filter(curve, size, 3)


def versus_plot():
    savedir = './savedir/'
    ext     = '.pkl'
    fig, ax = plt.subplots(1,2, sharex=True)
    plt.suptitle('Mean Accuracy Curve of Top 10% Evolved Models')

    titles = ['Spiking MS=0.8', 'Spiking MS=1.0']
    for fn, id, title in zip(['spiking_ms080_v0', 'spiking_ms100_v0'], [0,1], titles):
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
        if id == 1:
            ax[id].set_xlim(0, 2000)
        ax[id].set_ylim(0,1)
        ax[id].set_yticks(np.linspace(0,1,11))
        ax[id].set_xlabel('Iteration')
        ax[id].set_ylabel('Accuracy, Mutation Strength')

        ax[id].set_title(title)
    plt.legend()
    plt.show()


def weight_distribution():
    savedir = './savedir/'
    ext     = '.pkl'
    fig, ax = plt.subplots(2,2, sharex=True)
    plt.suptitle('Histogrammed Weights of All Networks')

    titles = ['Spiking MS=0.8']
    for fn, id, title in zip(['spiking_ms080_v0'], [0], titles):
        weights = pickle.load(open(savedir+fn+'_weights'+ext, 'rb'))

        EI_vector    = np.ones(100, dtype=np.float16)
        EI_vector[80:] *= -1
        EI_mask      = np.diag(EI_vector)[np.newaxis,:,:]


        W_rnn = weights['W_rnn']
        W_rnn = np.maximum(0., W_rnn)
        W_rnn = np.matmul(W_rnn, EI_mask)


        ax[0,0].set_title('E --> E')
        ax[0,0].hist(W_rnn[:,:80,:80].flatten(), bins=100)
        ax[0,0].set_ylim(0,210000)

        ax[0,1].set_title('I --> E')
        ax[0,1].hist(W_rnn[:,:80,80:].flatten(), bins=100)
        ax[0,1].set_ylim(0,60000)

        ax[1,0].set_title('E --> I')
        ax[1,0].hist(W_rnn[:,80:,:80].flatten(), bins=100)
        ax[1,0].set_ylim(0,50000)

        ax[1,1].set_title('I --> I')
        ax[1,1].hist(W_rnn[:,80:,80:].flatten(), bins=100)
        ax[1,1].set_ylim(0,20000)


    plt.show()


def get_curves(fn, size=11):

    data = pickle.load(open(fn, 'rb'))
    iters    = data['iter']
    task_acc = data['task_acc']
    full_acc = data['full_acc']
    loss     = data['loss']
    mut_str  = data['mut_str']

    return iters, smooth(task_acc, size)


def sweep_plot():
    savedir = './savedir/'
    ext     = '.pkl'
    fns     = [fn for fn in os.listdir(savedir) if 'standard' in fn]
    ms      = [fn for fn in fns if 'ms' in fn]
    mr      = [fn for fn in fns if 'mr' in fn]
    sr      = [fn for fn in fns if 'sr' in fn]
    cr      = [fn for fn in fns if 'cr' in fn]

    params = {
        'ms'      : [0.8, 0.9, 1.1],
        'mr'      : [0.1, 0.2, 0.4],
        'sr'      : [0.05, 0.125, 0.25],
        'cr'      : [0.1, 0.3, 0.5]
    }

    defaults = {
        'sr'         : 0.10,
        'mr'         : 0.25,
        'ms'         : 1.00,
        'cr'         : 0.25,
    }

    fig, ax = plt.subplots(2,2, figsize=[10,8])
    fig.suptitle('Sweeps into Evolution Parameters')
    fig.subplots_adjust(hspace=0.3)

    colors = [[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]]
    abbrv = ['ms', 'mr', 'sr', 'cr']
    names = ['mutation_strength', 'mutation_rate', 'survival_rate', 'cross_rate']
    for i, (group, name, ab) in enumerate(zip([ms, mr, sr, cr], names, abbrv)):

        a = ax[i%2, i//2]
        a.set_xlim(0,2000)
        a.set_xticks(np.linspace(0,2000,9))
        a.set_ylim(0.5,1)
        a.set_yticks(np.linspace(0.5,1,11))
        a.grid()
        a.set_title(name)
        a.set_xlabel('Iteration')
        a.set_ylabel('Accuracy')
        a.plot(*get_curves(savedir+'baseline_v0.pkl', size=21), c='k', label='baseline : {}'.format(defaults[ab]))
        a.plot(*get_curves(savedir+'baseline_v1.pkl', size=21), c='k')

        for c, j in zip(colors, range(3)):
            subgroup = [group for group in fns if ab+str(j) in group]
            for k, fn in enumerate(subgroup):
                if k == 0:
                    a.plot(*get_curves(savedir+fn, size=21), c=c, label='{}={}'.format(ab, params[ab][j]))
                else:
                    a.plot(*get_curves(savedir+fn, size=21), c=c)



        a.legend(ncol=2, loc='lower right')

    plt.show()


def explore_spiking():
    x = pickle.load(open('./savedir/h_out_array.pkl', 'rb'))[:,0,:,:]
    plt.imshow(x[:,0,:])
    plt.show()

    return x


#sweep_plot()
#x = explore_spiking()
weight_distribution()
