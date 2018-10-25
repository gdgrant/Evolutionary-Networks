import numpy as np
from itertools import product
print('\n--> Loading parameters...')

global par
par = {

    'save_dir'              : './savedir/',
    'save_fn'               : 'testing_spiking',
    'cell_type'             : 'adex',    # 'rate', 'LIF', 'adex'
    'use_stp'               : True,
    'iters_per_output'      : 5,
    'learning_method'       : 'GA', # Evo search = 'ES', Genetic algorithm = 'GA'

    'EI_prop'               : 0.8,
    'balance_EI'            : True,
    'exc_model'             : 'RS',
    'inh_model'             : 'cNA',
    'current_divider'       : 3e6,

    'use_latency'           : True,
    'latency_min'           : 5,
    'latency_max'           : 20,

    'freq_cost'             : 1e-4,
    'freq_target'           : 0.,

    'reciprocal_cost'       : 2.,
    'reciprocal_threshold'  : 3.,
    'reciprocal_max'        : 0.2,

    'use_weight_momentum'   : False,
    'momentum_scale'        : 1.,
    'ES_learning_rate'      : 0.01,
    'ES_sigma'              : 0.05,
    'adam_beta1'            : 0.9,
    'adam_beta2'            : 0.999,
    'adam_epsilon'          : 1e-8,


    'n_networks'            : 500,
    'n_hidden'              : 100,
    'n_output'              : 3,

    'num_motion_tuned'      : 24,
    'num_fix_tuned'         : 2,
    'num_rule_tuned'        : 2,
    'num_receptive_fields'  : 1,
    'num_motion_dirs'       : 8,

    'batch_size'            : 128,
    'iterations'            : 1001,

    'input_gamma'           : 0.2,
    'rnn_gamma'             : 0.025,
    'noise_rnn_sd'          : 0.05,
    'noise_in_sd'           : 0.05,

    'dt'                    : 1,
    'membrane_constant'     : 20,
    'output_constant'       : 40,

    'dead_time'             : 100,
    'fix_time'              : 200,
    'sample_time'           : 200,
    'delay_time'            : 300,
    'test_time'             : 200,
    'mask_time'             : 50,

    'tau_fast'              : 200,
    'tau_slow'              : 1500,

    'survival_rate'         : 0.10,
    'mutation_rate'         : 0.25,
    'mutation_strength'     : 0.80,
    'cross_rate'            : 0.01,

    'use_crossing'          : False,

    'task'                  : 'dms',
    'kappa'                 : 2.0,
    'tuning_height'         : 20.0,
    'response_multiplier'   : 10.0,
    'num_rules'             : 1,

    'loss_baseline'         : 10.,

}

def update_parameters(updates):
    for k in updates.keys():
        print(k.ljust(24), ': {}'.format(updates[k]))
        par[k] = updates[k]
    update_dependencies()


def update_dependencies():

    if par['learning_method'] == 'ES':
        # ensure that n_networks is odd
        par['n_networks'] = par['n_networks'] + 1 if par['n_networks']%2==0 else par['n_networks']

    par['trial_length'] = par['dead_time'] + par['fix_time'] + +par['sample_time'] + par['delay_time'] + par['test_time']
    par['num_time_steps'] = par['trial_length'] // par['dt']

    par['n_input'] = par['num_motion_tuned']*par['num_receptive_fields'] + par['num_fix_tuned'] + par['num_rule_tuned']
    par['n_EI'] = int(par['n_hidden']*par['EI_prop'])

    par['h_init_init']  = 0.1*np.ones([par['n_networks'],1,par['n_hidden']], dtype=np.float16)
    par['W_in_init']    = np.float16(np.random.gamma(shape=par['input_gamma'], scale=1., size=[par['n_networks'], par['n_input'], par['n_hidden']]))
    par['W_out_init']   = np.float16(np.random.gamma(shape=par['input_gamma'], scale=1., size=[par['n_networks'], par['n_hidden'], par['n_output']]))
    par['W_rnn_init']   = np.float16(np.random.gamma(shape=par['rnn_gamma'], scale=1., size=[par['n_networks'], par['n_hidden'], par['n_hidden']]))

    if par['balance_EI']:
        par['W_rnn_init'][:,par['n_EI']:,:par['n_EI']] = np.float16(np.random.gamma(shape=2*par['rnn_gamma'], scale=1., size=par['W_rnn_init'][:,par['n_EI']:,:par['n_EI']].shape))
        par['W_rnn_init'][:,:par['n_EI'],par['n_EI']:] = np.float16(np.random.gamma(shape=2*par['rnn_gamma'], scale=1., size=par['W_rnn_init'][:,:par['n_EI'],par['n_EI']:].shape))

    par['b_rnn_init']   = np.zeros([par['n_networks'], 1, par['n_hidden']], dtype=np.float16)
    par['b_out_init']   = np.zeros([par['n_networks'], 1, par['n_output']], dtype=np.float16)

    par['W_rnn_mask']   = 1 - np.eye(par['n_hidden'])[np.newaxis,:,:]
    par['W_rnn_init']  *= par['W_rnn_mask']

    par['EI_vector']    = np.ones(par['n_hidden'], dtype=np.float16)
    par['EI_vector'][par['n_EI']:] *= -1
    par['EI_mask']      = np.diag(par['EI_vector'])[np.newaxis,:,:]

    par['threshold_init'] = 0.5*np.ones([par['n_networks'],1,par['n_hidden']], dtype=np.float16)
    par['reset_init']     = np.zeros([par['n_networks'],1,par['n_hidden']], dtype=np.float16)

    par['dt_sec']       = par['dt']/1000
    par['alpha_neuron'] = np.float16(par['dt']/par['membrane_constant'])
    par['beta_neuron']  = np.float16(par['dt']/par['output_constant'])
    par['noise_rnn']    = np.float16(np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd'])
    par['noise_in']     = np.float16(np.sqrt(2/par['alpha_neuron'])*par['noise_rnn_sd'])

    par['num_survivors'] = int(par['n_networks'] * par['survival_rate'])


    ### Synaptic plasticity
    if par['use_stp']:
        par['alpha_stf']  = np.ones((1, 1, par['n_hidden']), dtype=np.float16)
        par['alpha_std']  = np.ones((1, 1, par['n_hidden']), dtype=np.float16)
        par['U']          = np.ones((1, 1, par['n_hidden']), dtype=np.float16)

        par['syn_x_init'] = np.zeros((1, 1, par['n_hidden']), dtype=np.float16)
        par['syn_u_init'] = np.zeros((1, 1, par['n_hidden']), dtype=np.float16)

        for i in range(0,par['n_hidden'],2):
            par['alpha_stf'][0,0,i] = par['dt']/par['tau_slow']
            par['alpha_std'][0,0,i] = par['dt']/par['tau_fast']
            par['U'][0,0,i] = 0.15
            par['syn_x_init'][0,0,i] = 1
            par['syn_u_init'][0,0,i] = par['U'][0,0,i+1]

            par['alpha_stf'][0,0,i+1] = par['dt']/par['tau_fast']
            par['alpha_std'][0,0,i+1] = par['dt']/par['tau_slow']
            par['U'][0,0,i+1] = 0.45
            par['syn_x_init'][0,0,i+1] = 1
            par['syn_u_init'][0,0,i+1] = par['U'][0,0,i+1]

        par['stp_mod'] = par['dt_sec'] if par['cell_type'] == 'rate' else 1.


    ### Adaptive-Expoential spiking
    if par['cell_type'] == 'adex':

        # Note that voltages are in units of mV and currents
        # are in units of mA.  When pulling from a table based in volts/amps,
        # multiply E, V_T, D, b, V_r, and Vth by 1000
        par['cNA'] = {
            'C'   : 59e-12,     'g'   : 2.9e-9,     'E'   : -62,
            'V_T' : -42,        'D'   : 3,          'a'   : 1.8e-9,
            'tau' : 16e-3,      'b'   : 61e-9,      'V_r' : -54,
            'Vth' : 20,         'dt'  : par['dt']/1000 }
        par['RS']  = {
            'C'   : 104e-12,    'g'   : 4.3e-9,     'E'   : -65,
            'V_T' : -52,        'D'   : 0.8,        'a'   : -0.8e-9,
            'tau' : 88e-3,      'b'   : 65e-9,      'V_r' : -53,
            'Vth' : 20,         'dt'  : par['dt']/1000 }

        par['adex'] = {}
        for (k0, v_exc), (k1, v_inh) in zip(par[par['exc_model']].items(), par[par['inh_model']].items()):
            assert(k0 == k1)
            par_matrix = np.ones([1,1,par['n_hidden']], dtype=np.float32)
            par_matrix[...,:int(par['n_hidden']*par['EI_prop'])] *= v_exc
            par_matrix[...,int(par['n_hidden']*par['EI_prop']):] *= v_inh
            par['adex'][k0] = par_matrix

        par['w_init'] = par['adex']['b']
        par['adex']['current_divider'] = par['current_divider']


    ### Latency-based weights
    if par['use_latency']:
        par['max_latency'] = par['latency_max']//par['dt']
        par['latency_matrix'] = np.random.uniform(par['latency_min']//par['dt'], par['latency_max']//par['dt'], \
            size=[par['n_hidden'], par['n_hidden']]).astype(np.int8)

        par['latency_mask'] = np.zeros([par['max_latency'], par['n_hidden'], par['n_hidden']]).astype(np.float32)
        for i, j in product(range(par['n_hidden']), range(par['n_hidden'])):
            par['latency_mask'][par['latency_matrix'][i,j],i,j] = 1.


update_dependencies()
print('--> Parameters loaded.\n')
