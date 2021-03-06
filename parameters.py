import numpy as np
from itertools import product
print('\n--> Loading parameters...')

global par
par = {

    'save_dir'              : './savedir/',
    'save_fn'               : 'adex_ga_large_slow_v3',
    'iters_per_output'      : 1,
    'datatype'              : 'int8',    # 'float32', 'float16', 'int8'

    'batch_size'            : 256,
    'iterations'            : 128,

    'learning_method'       : 'GA',     # Evo search = 'ES', genetic = 'GA', thermal = 'TA'
    'cell_type'             : 'rate',   # 'rate', 'adex'
    'use_stp'               : True,
    'use_adam'              : True,     # Only for 'ES'

    'temperature'           : 0.2,
    'temperature_decay'     : 0.999,

    'ES_learning_rate'      : 0.005,
    'ES_sigma'              : 0.02,

    'local_learning'        : False,
    'local_learning_vars'   : ['W_out', 'b_out'],
    'local_learning_rate'   : 0.005,

    'EI_prop'               : 0.8,
    'balance_EI'            : True,
    'exc_model'             : 'RS',
    'inh_model'             : 'cNA',
    'current_divider'       : 3e6,

    'use_latency'           : True,
    'latency_min'           : 8,
    'latency_max'           : 12,

    'freq_cost'             : 1e-4,
    'freq_target'           : 0.,

    'reciprocal_cost'       : 0.,
    'reciprocal_threshold'  : 3.,
    'reciprocal_max'        : 0.2,

    'use_weight_momentum'   : False,
    'momentum_scale'        : 1.,

    'adam_beta1'            : 0.9,
    'adam_beta2'            : 0.999,
    'adam_epsilon'          : 1e-8,

    'n_networks'            : 100,
    'n_hidden'              : 100,
    'n_output'              : 3,

    'num_motion_tuned'      : 24,
    'num_fix_tuned'         : 0,
    'num_rule_tuned'        : 0,
    'num_receptive_fields'  : 1,
    'num_motion_dirs'       : 8,

    'input_gamma'           : 0.08*8,# *15 works for n_hidden=100
    'rnn_gamma'             : 0.04*8,
    'output_gamma'          : 0.08,
    'noise_rnn_sd'          : 0.05,
    'noise_in_sd'           : 0.05,

    'dt'                    : 1,
    'membrane_constant'     : 20,
    'output_constant'       : 40,

    'tau_fast'              : 100,
    'tau_slow'              : 800,

    'dead_time'             : 50,
    'fix_time'              : 50,
    'sample_time'           : 100,
    'delay_time'            : 50,
    'test_time'             : 150,
    'mask_time'             : 20,
    'fixation_on'           : False,

    'survival_rate'         : 0.1,
    'mutation_rate'         : 0.1,
    'mutation_strength'     : 0.20,
    'cross_rate'            : 0.00,
    'use_crossing'          : False,
    'loss_baseline'         : 10.,

    'task'                  : 'dms',
    'kappa'                 : 2.0,
    'tuning_height'         : 4.0,
    'response_multiplier'   : 4.0,
    'num_rules'             : 1,

}


def update_parameters(updates):
    for k in updates.keys():
        print(k.ljust(24), ': {}'.format(updates[k]))
        par[k] = updates[k]
    update_dependencies()


def update_dependencies():

    if par['datatype'] == 'float32':
        par['w_dtype'] = np.float32     # Weight datatype
        par['c_dtype'] = np.float32     # Cell computation datatype
    elif par['datatype'] == 'float16':
        par['w_dtype'] = np.float16
        par['c_dtype'] = np.float16
    elif par['datatype'] == 'int8':
        par['w_dtype'] = np.int8
        par['c_dtype'] = np.float16

        par['input_gamma']  = 100*par['input_gamma']
        par['rnn_gamma']    = 100*par['rnn_gamma']
        par['output_gamma'] = 100*par['output_gamma']

    par['n_networks'] += 1 if par['learning_method'] == 'ES' \
        and par['n_networks']%2 == 0 else 0

    par['use_adam'] = par['use_adam'] and par['learning_method'] == 'ES'

    par['trial_length'] = par['dead_time'] + par['fix_time'] + +par['sample_time'] + par['delay_time'] + par['test_time']
    par['num_time_steps'] = par['trial_length'] // par['dt']

    par['n_input'] = par['num_motion_tuned']*par['num_receptive_fields'] + par['num_fix_tuned'] + par['num_rule_tuned']
    par['n_EI'] = int(par['n_hidden']*par['EI_prop'])

    par['h_init_init']  = 0.1*np.ones([par['n_networks'],1,par['n_hidden'],1], dtype=par['c_dtype'])
    par['W_in_init']    = np.random.gamma(par['input_gamma'], size=[par['n_networks'], 1, par['n_input'], par['n_hidden']]).astype(par['w_dtype'])
    par['W_rnn_init']   = np.random.gamma(par['rnn_gamma'], size=[par['n_networks'], 1, par['n_hidden'], par['n_hidden']]).astype(par['w_dtype'])
    par['W_out_init']   = np.random.gamma(par['output_gamma'], size=[par['n_networks'], 1, par['n_hidden'], par['n_output']]).astype(par['w_dtype'])

    if par['balance_EI']:
        par['W_rnn_init'][:,par['n_EI']:,:par['n_EI']] = np.random.gamma(2*par['rnn_gamma'], size=par['W_rnn_init'][:,par['n_EI']:,:par['n_EI']].shape).astype(par['w_dtype'])
        par['W_rnn_init'][:,:par['n_EI'],par['n_EI']:] = np.random.gamma(2*par['rnn_gamma'], size=par['W_rnn_init'][:,:par['n_EI'],par['n_EI']:].shape).astype(par['w_dtype'])

    par['b_rnn_init']   = np.zeros([par['n_networks'], 1, par['n_hidden'], 1], dtype=par['w_dtype'])
    par['b_out_init']   = np.zeros([par['n_networks'], 1, par['n_output'], 1], dtype=par['w_dtype'])

    par['W_rnn_mask']   = 1 - np.eye(par['n_hidden'])[np.newaxis,:,:].astype(par['w_dtype'])
    par['W_rnn_init']  *= par['W_rnn_mask']

    par['EI_vector']    = np.ones(par['n_hidden'])
    par['EI_vector'][par['n_EI']:] *= -1
    par['EI_mask']      = np.diag(par['EI_vector'])[np.newaxis,:,:].astype(par['w_dtype'])

    par['y_init_shape'] = [par['num_time_steps'], par['n_networks'], par['batch_size'], par['n_output']]

    par['dt_sec']       = par['dt']/1000
    par['alpha_neuron'] = par['c_dtype'](par['dt']/par['membrane_constant'])
    par['beta_neuron']  = par['c_dtype'](par['dt']/par['output_constant'])
    par['noise_rnn']    = par['c_dtype'](np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd'])
    par['noise_in']     = par['c_dtype'](np.sqrt(2/par['alpha_neuron'])*par['noise_rnn_sd'])

    par['num_survivors'] = int(par['n_networks'] * par['survival_rate']) if par['learning_method'] in ['GA', 'TA'] else 1

    ### Synaptic plasticity
    if par['use_stp']:
        par['alpha_stf']  = np.ones((1, 1, par['n_hidden'], 1), dtype=par['c_dtype'])
        par['alpha_std']  = np.ones((1, 1, par['n_hidden'], 1), dtype=par['c_dtype'])
        par['U']          = np.ones((1, 1, par['n_hidden'], 1), dtype=par['c_dtype'])

        par['syn_x_init'] = np.zeros((1, 1, par['n_hidden'], 1), dtype=par['c_dtype'])
        par['syn_u_init'] = np.zeros((1, 1, par['n_hidden'], 1), dtype=par['c_dtype'])

        for i in range(0,par['n_hidden'],2):
            par['alpha_stf'][0,0,i,0] = par['dt']/par['tau_slow']
            par['alpha_std'][0,0,i,0] = par['dt']/par['tau_fast']
            par['U'][0,0,i] = 0.15
            par['syn_x_init'][0,0,i,0] = 1
            par['syn_u_init'][0,0,i,0] = par['U'][0,0,i+1]

            par['alpha_stf'][0,0,i+1,0] = par['dt']/par['tau_fast']
            par['alpha_std'][0,0,i+1,0] = par['dt']/par['tau_slow']
            par['U'][0,0,i+1] = 0.45
            par['syn_x_init'][0,0,i+1,0] = 1
            par['syn_u_init'][0,0,i+1,0] = par['U'][0,0,i+1]

        par['stp_mod'] = par['dt_sec'] if par['cell_type'] == 'rate' else 1.


    ### Adaptive-Exponential spiking
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

        par['state_buffer_shape'] = [par['max_latency'], par['n_networks'], par['batch_size'], par['n_hidden']]


update_dependencies()
print('--> Parameters loaded.\n')
