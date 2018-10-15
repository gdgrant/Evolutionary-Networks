import model
import time
from parameters import *

def try_model(updates):

    print('Updating parameters...')
    update_parameters(updates)

    t0 = time.time()
    try:
        model.main()
        print('Model run concluded.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt.  Run time: {:5.3f} s.\n\n'.format(time.time()-t0))


defaults = {
    'survival_rate'         : 0.10,
    'mutation_rate'         : 0.25,
    'mutation_strength'     : 1.00,
    'cross_rate'            : 0.25,
}


def MS_sweep(j):
    print('Resetting...')
    update_parameters(defaults)
    print('Reset complete.\n')

    mutation_strengths = [0.8, 0.9, 1.1]
    for i, a in enumerate(mutation_strengths):
        updates = {
            'mutation_strength' : a,
            'save_fn'           : 'standard_ms{}_v{}'.format(i,j)
        }

        try_model(updates)


def MR_sweep(j):
    print('Resetting...')
    update_parameters(defaults)
    print('Reset complete.\n')

    mutation_rates = [0.1, 0.2, 0.4]
    for i, a in enumerate(mutation_rates):
        updates = {
            'mutation_rate'     : a,
            'save_fn'           : 'standard_mr{}_v{}'.format(i,j)
        }

        try_model(updates)


def SR_sweep(j):
    print('Resetting...')
    update_parameters(defaults)
    print('Reset complete.\n')

    survival_rates = [0.05, 0.125, 0.25]
    for i, a in enumerate(survival_rates):
        updates = {
            'survival_rate'     : a,
            'save_fn'           : 'standard_sr{}_v{}'.format(i,j)
        }

        try_model(updates)


def CR_sweep(j):
    print('Resetting...')
    update_parameters(defaults)
    print('Reset complete.\n')

    cross_rates = [0.1, 0.3, 0.5]
    for i, a in enumerate(cross_rates):
        updates = {
            'cross_rate'        : a,
            'save_fn'           : 'standard_cr{}_v{}'.format(i,j)
        }

        try_model(updates)


for j in range(5):
    MS_sweep(j)
    MR_sweep(j)


    #SR_sweep(j)
    #CR_sweep(j)
