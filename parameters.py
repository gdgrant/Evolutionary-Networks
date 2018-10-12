import numpy as np
print('\n--> Loading parameters...')

global par
par = {

    'save_dir'          : './savedir/',

    'n_networks'        : 100,
    'n_input'           : 40,
    'n_hidden'          : 250,
    'n_output'          : 12,

    'batch_size'        : 128,
    'n_batches'         : 501,

    'input_gamma'       : 0.2,
    'rnn_gamma'         : 0.025,
    'noise_rnn_sd'      : 0.05,

    'trial_length'      : 2000,
    'dt'                : 20,
    'membrane_constant' : 100,

    'survival_rate'     : 0.1,
    'mutation_rate'     : 0.5,
    'mutation_strength' : 1.0,

}


def update_dependencies():

    par['num_time_steps'] = par['trial_length'] // par['dt']


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

    par['num_survivors'] = int(par['n_networks'] * par['survival_rate'])



update_dependencies()
print('--> Parameters loaded.\n')
