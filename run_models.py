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


def cross_rate_sweep():

    updates = {
        'iterations'        : 1001,
        'task'              : 'dms',
        'n_hidden'          : 150,
    }

    for i, rate in enumerate([0.0, 0.01, 0.05, 0.1, 0.2]):
        updates['cross_rate'] = rate
        updates['save_fn'] = 'crossrate{}_dms_v0'.format(i)
        try_model(updates)


def mutation_strength_sweep():

    updates = {
        'iterations'        : 1001,
        'task'              : 'dms',
        'n_hidden'          : 150,
    }

    for i, rate in enumerate([0.1, 0.25, 0.4, 0.6]):
        updates['mutation_strength'] = rate
        updates['save_fn'] = 'mutstrength{}_dms_v0'.format(i)
        try_model(updates)


#cross_rate_sweep()
#mutation_strength_sweep()







double_neurons = {
    'iterations'        : 10001,
    'task'              : 'dms',
    'save_fn'           : 'double_neurons_dms_v1',
    'n_hidden'          : 200,
    'batch_size'        : 256,
    'mutation_strength' : 0.40
}

quad_neurons = {
    'iterations'        : 10001,
    'task'              : 'dms',
    'save_fn'           : 'quad_neurons_dms_v1',
    'n_hidden'          : 400,
    'batch_size'        : 64,
    'mutation_strength' : 0.40
}

output_constant = {
    'iterations'      : 10001,
    'task'            : 'dms',
    'save_fn'         : 'output_constant80_dms_v0',
    'output_constant' : 80,
}

base_model = {
    'iterations'          : 10001,
    'task'                : 'dms',
    'save_fn'             : 'standard_model_dms_v2',
    'use_weight_momentum' : True,
    'n_hidden'            : 400,
    'batch_size'          : 64,
    'freq_cost'           : 1e-3,
}

try_model(base_model)
