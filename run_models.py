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

updates = {
    'iterations'        : 10001,
    'use_stp'           : False,
    'mutation_strength' : 0.008,
    'save_fn'           : 'dms_without_stp_v0'
}

try_model(updates)
