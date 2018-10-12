import numpy as np
print('\n--> Loading parameters...')

global par
par = {

    'save_dir'              : './savedir/',

    'n_networks'            : 200,
    'n_hidden'              : 250,
    'n_output'              : 3,

    'num_motion_tuned'      : 32,
    'num_fix_tuned'         : 2,
    'num_rule_tuned'        : 2,
    'num_receptive_fields'  : 1,
    'num_motion_dirs'       : 8,

    'batch_size'            : 128,
    'iterations'            : 501,

    'input_gamma'           : 0.2,
    'rnn_gamma'             : 0.025,
    'noise_rnn_sd'          : 0.05,
    'noise_in_sd'           : 0.05,

    'dt'                    : 20,
    'membrane_constant'     : 100,

    'dead_time'             : 100,
    'fix_time'              : 200,
    'sample_time'           : 200,
    'delay_time'            : 200,
    'test_time'             : 200,
    'mask_time'             : 50,

    'survival_rate'         : 0.10,
    'mutation_rate'         : 0.25,
    'mutation_strength'     : 0.50,

    'task'                  : 'dms',
    'kappa'                 : 2.0,
    'tuning_height'         : 4.0,
    'num_rules'             : 1,

}


def update_dependencies():

    par['trial_length'] = par['dead_time'] + par['fix_time'] + +par['sample_time'] + par['delay_time'] + par['test_time']
    par['num_time_steps'] = par['trial_length'] // par['dt']

    par['n_input'] = par['num_motion_tuned']*par['num_receptive_fields'] + par['num_fix_tuned'] + par['num_rule_tuned']

    par['h_init_init']  = 0.1*np.ones([par['n_networks'], 1,par['n_hidden']], dtype=np.float32)
    par['W_in_init']    = np.float32(np.random.gamma(shape=par['input_gamma'], scale=1., size=[par['n_networks'], par['n_input'], par['n_hidden']]))
    par['W_out_init']   = np.float32(np.random.gamma(shape=par['input_gamma'], scale=1., size=[par['n_networks'], par['n_hidden'], par['n_output']]))
    par['W_rnn_init']   = np.float32(np.random.gamma(shape=par['rnn_gamma'], scale=1., size=[par['n_networks'], par['n_hidden'], par['n_hidden']]))

    par['b_rnn_init']   = np.zeros([par['n_networks'], 1, par['n_hidden']], dtype=np.float32)
    par['b_out_init']   = np.zeros([par['n_networks'], 1, par['n_output']], dtype=np.float32)

    par['W_rnn_mask']   = 1 - np.eye(par['n_hidden'])[np.newaxis,:,:]
    par['W_rnn_init']  *= par['W_rnn_mask']

    par['alpha_neuron'] = np.float32(par['dt']/par['membrane_constant'])
    par['noise_rnn']    = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in']     = np.sqrt(2/par['alpha_neuron'])*par['noise_rnn_sd']

    par['num_survivors'] = int(par['n_networks'] * par['survival_rate'])



update_dependencies()
print('--> Parameters loaded.\n')
