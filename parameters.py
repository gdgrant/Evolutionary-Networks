import numpy as np
print('\n--> Loading parameters...')

global par
par = {

    'save_dir'              : './savedir/',
    'save_fn'               : 'testing_spiking',
    'network_type'          : 'spiking',
    'use_stp'               : True,
    'EI_prop'               : 0.8,
    'iters_per_output'      : 5,

    'n_networks'            : 2000,
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
    'mutation_strength'     : 1.00,
    'cross_rate'            : 0.25,

    'task'                  : 'oic',
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

    par['trial_length'] = par['dead_time'] + par['fix_time'] + +par['sample_time'] + par['delay_time'] + par['test_time']
    par['num_time_steps'] = par['trial_length'] // par['dt']

    par['n_input'] = par['num_motion_tuned']*par['num_receptive_fields'] + par['num_fix_tuned'] + par['num_rule_tuned']

    par['h_init_init']  = 0.1*np.ones([par['n_networks'],1,par['n_hidden']], dtype=np.float16)
    par['W_in_init']    = np.float16(np.random.gamma(shape=par['input_gamma'], scale=1., size=[par['n_networks'], par['n_input'], par['n_hidden']]))
    par['W_out_init']   = np.float16(np.random.gamma(shape=par['input_gamma'], scale=1., size=[par['n_networks'], par['n_hidden'], par['n_output']]))
    par['W_rnn_init']   = np.float16(np.random.gamma(shape=par['rnn_gamma'], scale=1., size=[par['n_networks'], par['n_hidden'], par['n_hidden']]))

    par['b_rnn_init']   = np.zeros([par['n_networks'], 1, par['n_hidden']], dtype=np.float16)
    par['b_out_init']   = np.zeros([par['n_networks'], 1, par['n_output']], dtype=np.float16)

    par['W_rnn_mask']   = 1 - np.eye(par['n_hidden'])[np.newaxis,:,:]
    par['W_rnn_init']  *= par['W_rnn_mask']

    par['EI_vector']    = np.ones(par['n_hidden'], dtype=np.float16)
    par['EI_vector'][int(par['n_hidden']*par['EI_prop']):] *= -1
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


update_dependencies()
print('--> Parameters loaded.\n')
