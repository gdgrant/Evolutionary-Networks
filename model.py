from utils import *
from parameters import par, update_dependencies
from stimulus import Stimulus

class NetworkController:

    def __init__(self):
        """ Load initial network ensemble state """

        self.make_constants()
        self.make_variables()

        self.size_ref = cp.ones([par['n_networks'],par['batch_size'],par['n_hidden']], \
            dtype=cp.float16)


    def make_variables(self):
        """ Pull network variables into GPU """

        if par['cell_type'] == 'rate':
            var_names = ['W_in', 'W_out', 'W_rnn', 'b_rnn', 'b_out', 'h_init']
        elif par['cell_type'] == 'adex':
            var_names = ['W_in', 'W_out', 'W_rnn']

        self.var_dict = {}
        for v in var_names:
            self.var_dict[v] = to_gpu(par[v+'_init'])


    def make_constants(self):
        """ Pull constants for computation into GPU """

        gen_constants   = ['n_networks', 'n_hidden', 'W_rnn_mask', 'EI_mask', 'noise_rnn']
        time_constants  = ['alpha_neuron', 'beta_neuron', 'dt', 'num_time_steps']
        loss_constants  = ['freq_cost', 'freq_target', 'reciprocal_cost', 'reciprocal_max', 'reciprocal_threshold']

        stp_constants   = ['syn_x_init', 'syn_u_init', 'U', 'alpha_stf', 'alpha_std', 'stp_mod']
        adex_constants  = ['adex', 'w_init']
        lat_constants   = ['latency_mask', 'max_latency']

        GA_constants    = ['mutation_rate', 'mutation_strength', 'cross_rate', 'loss_baseline']
        ES_constants    = ['ES_learning_rate', 'ES_sigma']

        constant_names  = gen_constants + time_constants + loss_constants
        constant_names += stp_constants if par['use_stp'] else []
        constant_names += adex_constants if par['cell_type'] == 'adex' else []
        constant_names += lat_constants if par['use_latency'] else []
        constant_names += GA_constants if par['learning_method'] == 'GA' else []
        constant_names += ES_constants if par['learning_method'] == 'ES' else []

        self.con_dict = {}
        for c in constant_names:
            self.con_dict[c] = to_gpu(par[c])


    def update_constant(self, name, val):
        """ Update a given constant in the model """

        self.con_dict[name] = to_gpu(val)


    def run_models(self, input_data):
        """ Run network ensemble based on input data, collecting network outputs into y """

        # Establish inputs, outputs, and recording
        input_data = to_gpu(input_data)
        self.y = cp.zeros(par['y_init_shape'], dtype=cp.float16)
        self.spiking_means = cp.zeros([par['n_networks']])

        # Initialize cell states
        if par['cell_type'] == 'rate':
            spike = self.var_dict['h_init'] * self.size_ref
        else:
            spike = 0. * self.size_ref
            state = self.con_dict['adex']['V_r'] * self.size_ref
            adapt = self.con_dict['w_init'] * self.size_ref

        # Initialize STP if being used
        if par['use_stp']:
            syn_x = self.con_dict['syn_x_init'] * self.size_ref
            syn_u = self.con_dict['syn_u_init'] * self.size_ref
        else:
            syn_x = syn_u = 0.

        # Initialize latency buffer if being used
        if par['use_latency']:
            self.state_buffer = cp.zeros(par['state_buffer_shape'], dtype=cp.float16)

        # Apply the EI mask to the recurrent weights
        self.W_rnn_effective = apply_EI(self.var_dict['W_rnn'], self.con_dict['EI_mask'])


        # Loop across time and collect network output into y, using the
        # desired recurrent cell type
        for t in range(par['num_time_steps']):
            if par['cell_type'] == 'rate':
                spike, syn_x, syn_u = self.rate_recurrent_cell(spike, input_data[t], syn_x, syn_u, t)
                self.y[t,...] = cp.matmul(spike, self.var_dict['W_out']) + self.var_dict['b_out']
                self.spiking_means += cp.mean(spike, axis=(1,2))/self.con_dict['num_time_steps']

            elif par['cell_type'] == 'adex':
                spike, state, adapt, syn_x, syn_u = self.AdEx_recurrent_cell(spike, state, adapt, input_data[t], syn_x, syn_u, t)
                self.y[t,...] = (1-self.con_dict['beta_neuron'])*self.y[t-1,...] \
                    + self.con_dict['beta_neuron']*cp.matmul(spike, self.var_dict['W_out'])
                self.spiking_means += cp.mean(spike, axis=(1,2))*1000/self.con_dict['num_time_steps']


    def rnn_matmul(self, h_in, W_rnn, t):
        """ Perform the matmul operation required for the recurrent
            weight matrix, performing special operations such as latency
            where ncessary """

        if par['use_latency']:
            # Calculate this time step's latency-affected W_rnn and switch
            # to next time step
            W_rnn_latency = W_rnn[cp.newaxis,:,...] * self.con_dict['latency_mask'][:,cp.newaxis,...]
            self.con_dict['latency_mask'] = cp.roll(self.con_dict['latency_mask'], shift=1, axis=0)

            # Zero out the previous time step's buffer, and add to the
            # buffer for the upcoming time steps
            self.state_buffer[t-1%self.con_dict['max_latency'],...] = 0.
            self.state_buffer += cp.matmul(h_in, W_rnn_latency)

            # Return the hidden state buffer for this time step
            return self.state_buffer[t%self.con_dict['max_latency'],...]
        else:
            return cp.matmul(h_in, W_rnn)


    def rate_recurrent_cell(self, h, rnn_input, syn_x, syn_u, t):
        """ Process one time step of the hidden layer
            based on the previous state and the current input,
            using the rate-based model"""

        # Apply synaptic plasticity
        h_post, syn_x, syn_u = synaptic_plasticity(h, syn_x, syn_u, \
            self.con_dict, par['use_stp'], par['n_hidden'])

        # Calculate new hidden state
        h = relu((1-self.con_dict['alpha_neuron'])*h \
          + self.con_dict['alpha_neuron']*(cp.matmul(rnn_input, self.var_dict['W_in']) \
          + self.rnn_matmul(h_post, self.W_rnn_effective, t) + self.var_dict['b_rnn']) \
          + cp.random.normal(scale=self.con_dict['noise_rnn'], size=h.shape).astype(cp.float16))

        return h, syn_x, syn_u


    def AdEx_recurrent_cell(self, spike, V, w, rnn_input, syn_x, syn_u, t):
        """ Process one time step of the hidden layer
            based on the previous state and the current input,
            using the adaptive-exponential spiking model """

        # Apply synaptic plasticity
        spike_post, syn_x, syn_u = synaptic_plasticity(spike, syn_x, syn_u, \
            self.con_dict, par['use_stp'], par['n_hidden'])

        # Calculate the current incident on the hidden neurons
        I = cp.matmul(rnn_input, self.var_dict['W_in']) + self.rnn_matmul(spike_post, self.W_rnn_effective, t)
        V, w, spike = run_adex(V, w, I, self.con_dict['adex'])

        return spike, V, w, syn_x, syn_u


    def judge_models(self, output_data, output_mask):
        """ Determine the loss of each model, and rank them accordingly """

        # Load output data and mask to the GPU
        self.output_data = to_gpu(output_data)
        self.output_mask = to_gpu(output_mask)

        # Calculate the task loss of each network (returns an array of size [n_networks])
        self.task_loss = cross_entropy(self.output_mask, self.output_data, self.y)

        # Calculate the frequency loss of each network (returns an array of size [n_networks])
        self.freq_loss = self.con_dict['freq_cost']*cp.abs(self.spiking_means-self.con_dict['freq_target'])

        # Calculate the reciprocal weights loss of each network (returns an array of size [n_networks])
        weight_ref = self.var_dict['W_rnn'][:,:par['n_EI'],:par['n_EI']] > self.con_dict['reciprocal_threshold']
        self.reci_loss = cp.mean(weight_ref * weight_ref.transpose([0,2,1]), axis=(1,2))
        self.reci_loss = -self.con_dict['reciprocal_cost']*cp.minimum(self.con_dict['reciprocal_max'], self.reci_loss)

        # Aggregate the various loss terms
        self.loss = self.task_loss + self.freq_loss + self.reci_loss

        # If a network explodes due to a poorly-selected recurrent connection,
        # set that network's loss to the loss baseline (chosen prior to the 0th iteration)
        self.loss[cp.where(cp.isnan(self.loss))] = self.con_dict['loss_baseline']

        # Rank the networks (returns [n_networks] indices)
        self.rank = cp.argsort(self.loss.astype(cp.float32)).astype(cp.int16)

        # Sort the weights if required by the current learning method
        if par['learning_method'] == 'GA':
            for name in self.var_dict.keys():
                self.var_dict[name] = self.var_dict[name][self.rank,...]


    def get_spiking(self):
        """ Return the spiking means of each network (unranked) """
        return to_cpu(self.spiking_means)


    def get_weights(self):
        """ Return the mean of the surviving networks' weights
            (post-sort, if sorted by the current learning method) """
        return to_cpu({name:np.mean(self.var_dict[name][:par['num_survivors'],...], axis=0) \
            for name in self.var_dict.keys()})


    def get_losses(self, ranked=True):
        """ Return the losses of each network, ranked if desired """
        if ranked:
            return to_cpu(self.loss[self.rank])
        else:
            return to_cpu(self.loss)


    def get_losses_by_type(self, ranked=True):
        """ Return the losses of each network, separated by type,
            and ranked if desired """
        if ranked:
            return to_cpu({'task':self.task_loss[self.rank], \
                'freq':self.freq_loss[self.rank], 'reci':self.reci_loss[self.rank]})
        else:
            return to_cpu({'task':self.task_loss, 'freq':self.freq_loss, 'reci':self.reci_loss})


    def get_performance(self, ranked=True):
        """ Return the accuracies of each network, ranked if desired """
        self.task_accuracy = accuracy(self.y, self.output_data, self.output_mask)
        self.full_accuracy = accuracy(self.y, self.output_data, self.output_mask, inc_fix=True)

        if ranked:
            return to_cpu(self.task_accuracy[self.rank]), to_cpu(self.full_accuracy[self.rank])
        else:
            return to_cpu(self.task_accuracy), to_cpu(self.full_accuracy)


    def breed_models_genetic(self):
        """ Based on the first s networks in the ensemble, produce more networks
            slightly mutated from those s """

        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'], par['n_networks'], par['num_survivors'])

            if par['use_crossing']:
                raise Exception('Crossing not currently implemented.')

            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0], \
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])

        self.var_dict['W_rnn'] *= self.con_dict['W_rnn_mask']


def main():

    # Start the model run by loading the network controller and stimulus
    print('\nStarting model run: {}'.format(par['save_fn']))
    control = NetworkController()
    stim    = Stimulus()

    # Select whether to get losses ranked, according to learning method
    if par['learning_method'] == 'GA':
        is_ranked = True
    elif par['learning_method'] == 'ES':
        is_ranked = False
    else:
        raise Exception('Unknown learning method: {}'.format(par['learning_method']))

    # Get loss baseline and update the ensemble reference accordingly
    trial_info = stim.make_batch()
    control.run_models(trial_info['neural_input'])
    control.judge_models(trial_info['desired_output'], trial_info['train_mask'])
    loss_baseline = np.mean(control.get_losses(is_ranked))
    control.update_constant('loss_baseline', loss_baseline)

    # Establish records for training loop
    save_record = {'iter':[], 'mean_task_acc':[], 'mean_full_acc':[], 'top_task_acc':[], \
        'top_full_acc':[], 'loss':[], 'mut_str':[], 'spiking':[], 'loss_factors':[]}

    # Run the training loop
    for i in range(par['iterations']):

        # Process a batch of stimulus using the current models
        trial_info = stim.make_batch()
        control.run_models(trial_info['neural_input'])
        control.judge_models(trial_info['desired_output'], trial_info['train_mask'])

        # Get the current loss scores
        loss = control.get_losses(is_ranked)

        # Apply optimization based on the current learning method
        if par['learning_method'] == 'GA':
            mutation_strength = par['mutation_strength']*(np.mean(loss[:par['num_survivors']])/loss_baseline)
            control.update_constant('mutation_strength', mutation_strength)
            thresholds = [0.25, 0.1, 0.05, 0.025, 0]
            modifiers  = [1/2, 1/4, 1/8, 1/16]
            for t in range(len(thresholds))[:-1]:
                if thresholds[t] > mutation_strength > thresholds[t+1]:
                    mutation_strength = par['mutation_strength']*np.mean(loss)/loss_baseline * modifiers[t]
                    break

            control.breed_models_genetic()

        elif par['learning_method'] == 'ES':
            control.breed_models_evo_search(i)

        # Print and save network performance as desired
        if i%par['iters_per_output'] == 0:
            task_accuracy, full_accuracy = control.get_performance(is_ranked)
            loss_dict = control.get_losses_by_type(is_ranked)
            spikes    = control.get_spiking()

            task_loss = np.mean(loss_dict['task'][:par['num_survivors']])
            freq_loss = np.mean(loss_dict['freq'][:par['num_survivors']])
            reci_loss = np.mean(loss_dict['reci'][:par['num_survivors']])

            mean_loss = np.mean(loss[:par['num_survivors']])
            task_acc  = np.mean(task_accuracy[:par['num_survivors']])
            full_acc  = np.mean(full_accuracy[:par['num_survivors']])
            spiking   = np.mean(spikes[:par['num_survivors']])

            if par['learning_method'] == 'GA':
                top_task_acc = task_accuracy.max()
                top_full_acc = full_accuracy.max()
            elif par['learning_method'] == 'ES':
                top_task_acc = task_accuracy[0]
                top_full_acc = full_accuracy[0]

            save_record['iter'].append(i)
            save_record['top_task_acc'].append(top_task_acc)
            save_record['top_full_acc'].append(top_full_acc)
            save_record['mean_task_acc'].append(task_acc)
            save_record['mean_full_acc'].append(full_acc)
            save_record['loss'].append(mean_loss)
            save_record['loss_factors'].append(loss_dict)
            save_record['mut_str'].append(mutation_strength)
            save_record['spiking'].append(spiking)
            pickle.dump(save_record, open(par['save_dir']+par['save_fn']+'.pkl', 'wb'))
            if i%(10*par['iters_per_output']) == 0:
                print('Saving weights for iteration {}... ({})\n'.format(i, par['save_fn']))
                pickle.dump(to_cpu(control.var_dict), open(par['save_dir']+par['save_fn']+'_weights.pkl', 'wb'))

            status_stringA = 'Iter: {:4} | Task Loss: {:5.3f} | Freq Loss: {:5.3f} | Reci Loss: {:5.3f}'.format( \
                i, task_loss, freq_loss, reci_loss)
            status_stringB = ' '*11 + '| Full Loss: {:5.3f} | Mut Str: {:7.5f} | Spiking: {:5.2f} Hz'.format( \
                mean_loss, mutation_strength, spiking)
            status_stringC = ' '*11 + '| Top Acc (Task/Full): {:5.3f} / {:5.3f}  | Mean Acc (Task/Full): {:5.3f} / {:5.3f}'.format( \
                top_task_acc, top_full_acc, task_acc, full_acc)
            print(status_stringA + '\n' + status_stringB + '\n' + status_stringC)

if __name__ == '__main__':
    main()
