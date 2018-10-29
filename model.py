from utils import *
from parameters import par, update_dependencies
from stimulus import Stimulus
from sklearn.neighbors import NearestNeighbors

class NetworkController:

    def __init__(self):
        """ Load initial network ensemble state """

        self.make_constants()
        self.make_variables()
        if par['learning_method'] == 'ES':
            # use the ADAM optimizer
            self.make_adam_variables()

        self.size_ref = cp.ones([par['n_networks'],par['batch_size'],par['n_hidden']], dtype=cp.float16)

        self.NNbs = NearestNeighbors(n_neighbors=20, algorithm='kd_tree', \
            radius=1.0, leaf_size=100)


    def make_variables(self):
        """ Pull variables into GPU """

        var_names = ['W_in', 'W_out', 'W_rnn', 'b_rnn', 'b_out', 'h_init']
        var_names += ['threshold', 'reset'] if par['cell_type']=='LIF' else []
        self.var_dict = {}
        for v in var_names:
            self.var_dict[v] = to_gpu(par[v+'_init'])


    def make_adam_variables(self):

        self.adam_par = {}
        self.adam_par['beta1'] = to_gpu(par['adam_beta1'])
        self.adam_par['beta2'] = to_gpu(par['adam_beta2'])
        self.adam_par['epsilon'] = to_gpu(par['adam_epsilon'])
        self.adam_par['t'] = to_gpu(0)

        for v in self.var_dict.keys():
            self.adam_par['m_' + v] = cp.zeros_like(self.var_dict[v][0])
            self.adam_par['v_' + v] = cp.zeros_like(self.var_dict[v][0])



    def make_constants(self):
        """ Pull constants into GPU """

        constant_names    = ['alpha_neuron', 'beta_neuron', 'noise_rnn', 'W_rnn_mask', \
            'mutation_rate', 'mutation_strength', 'cross_rate', 'EI_mask', 'loss_baseline', \
            'dt', 'freq_cost', 'freq_target', 'num_time_steps', 'reciprocal_max', 'reciprocal_cost', 'reciprocal_threshold']

        stp_constants     = ['syn_x_init', 'syn_u_init', 'U', 'alpha_stf', 'alpha_std', 'stp_mod']
        adex_constants    = ['adex', 'w_init']
        latency_constants = ['max_latency', 'latency_mask']
        evo_constants     = ['ES_learning_rate', 'ES_sigma', 'n_networks']

        constant_names += stp_constants  if par['use_stp'] else []
        constant_names += adex_constants if par['cell_type'] == 'adex' else []
        constant_names += latency_constants if par['use_latency'] else []
        constant_names += evo_constants if par['learning_method'] == 'ES' else []

        self.con_dict = {}
        for c in constant_names:
            self.con_dict[c] = to_gpu(par[c])


    def update_constant(self, con_name, con):
        """ Update a given constant in the model """

        self.con_dict[con_name] = to_gpu(con)


    def run_models(self, input_data):
        """ Run model based on input data, collecting network outputs into y """

        input_data = to_gpu(input_data)

        self.y = cp.zeros([par['num_time_steps'], par['n_networks'], par['batch_size'], par['n_output']], dtype=cp.float16)
        syn_x  = self.con_dict['syn_x_init']  * self.size_ref if par['use_stp'] else 0.
        syn_u  = self.con_dict['syn_u_init']  * self.size_ref if par['use_stp'] else 0.
        h      = self.var_dict['h_init']      * self.size_ref
        h_out  = cp.zeros_like(h)
        if par['cell_type'] == 'adex':
            w  = self.con_dict['w_init']      * self.size_ref
            h  = self.con_dict['adex']['V_r'] * self.size_ref
        if par['use_latency']:
            self.h_out_buffer = cp.zeros([par['max_latency'], par['n_networks'], par['batch_size'], par['n_hidden']], dtype=cp.float16)

        self.W_rnn_effective = apply_EI(self.var_dict['W_rnn'], self.con_dict['EI_mask'])

        self.spiking_means = cp.zeros([par['n_networks']])
        for t in range(par['num_time_steps']):
            if par['cell_type'] == 'rate':
                _, h, syn_x, syn_u = self.rate_recurrent_cell(None, h, input_data[t], syn_x, syn_u, t)
                self.y[t,...] = cp.matmul(h, self.var_dict['W_out']) + self.var_dict['b_out']

                self.spiking_means += cp.mean(h, axis=(1,2))/self.con_dict['num_time_steps']

            elif par['cell_type'] == 'LIF':
                h_out, h, syn_x, syn_u = self.LIF_spiking_recurrent_cell(h_out, h, input_data[t], syn_x, syn_u, t)
                self.y[t,...] = (1-self.con_dict['beta_neuron'])*self.y[t-1,...] \
                              + self.con_dict['beta_neuron']*cp.matmul(h_out, self.var_dict['W_out'])# + self.var_dict['b_out']
                self.y[t,...] = cp.minimum(relu(self.y[t,...]), 5)

                self.spiking_means += cp.mean(h_out, axis=(1,2))*(1000/self.con_dict['num_time_steps'])

            elif par['cell_type'] == 'adex':
                h_out, h, w, syn_x, syn_u = self.AdEx_spiking_recurrent_cell(h_out, h, w, input_data[t], syn_x, syn_u, t)
                self.y[t,...] = (1-self.con_dict['beta_neuron'])*self.y[t-1,...] \
                              + self.con_dict['beta_neuron']*cp.matmul(h_out, self.var_dict['W_out'])# + self.var_dict['b_out']
                self.y[t,...] = cp.minimum(relu(self.y[t,...]), 5)

                self.spiking_means += cp.mean(h_out, axis=(1,2))*(1000/self.con_dict['num_time_steps'])

        return to_cpu(self.spiking_means)


    def rnn_matmul(self, h_in, W_rnn, t):
        """ Performs the matrix multiplication required for the recurrent
            weight matrix, performing special operations such as latency
            where necessary """

        if par['use_latency']:
            # Calculate this time step's latency-affected W_rnn and switch
            # to next time step
            W_rnn_latency = W_rnn[cp.newaxis,:,...] * self.con_dict['latency_mask'][:,cp.newaxis,...]
            self.con_dict['latency_mask'] = cp.roll(self.con_dict['latency_mask'], shift=1, axis=0)

            # Zero out the previous time step's buffer, and add to the
            # buffer for the upcoming time steps
            self.h_out_buffer[t-1%self.con_dict['max_latency'],...] = 0.
            self.h_out_buffer += cp.matmul(h_in, W_rnn_latency)

            # Return the hidden state buffer for this time step
            return self.h_out_buffer[t%self.con_dict['max_latency'],...]
        else:
            return cp.matmul(h_in, W_rnn)


    def rate_recurrent_cell(self, h_out, h, rnn_input, syn_x, syn_u, t):
        """ Process one time step of the hidden layer
            based on the previous state and the current input """

        h_post, syn_x, syn_u = synaptic_plasticity(h, syn_x, syn_u, \
            self.con_dict, par['use_stp'], par['n_hidden'])

        h = relu((1-self.con_dict['alpha_neuron'])*h \
            + self.con_dict['alpha_neuron']*(cp.matmul(rnn_input, self.var_dict['W_in']) \
            + self.rnn_matmul(h_post, self.W_rnn_effective, t) + self.var_dict['b_rnn']) + \
            + cp.random.normal(scale=self.con_dict['noise_rnn'], size=h.shape))

        return None, h, syn_x, syn_u


    def LIF_spiking_recurrent_cell(self, h_out, h, rnn_input, syn_x, syn_u, t):
        """ Process one time step of the hidden layer
            based on the previous state and the current input,
            using leaky integrate-and-fire spiking """

        h_post, syn_x, syn_u = synaptic_plasticity(h_out, syn_x, syn_u, \
            self.con_dict, par['use_stp'], par['n_hidden'])

        h = (1-self.con_dict['alpha_neuron'])*h \
            + self.con_dict['alpha_neuron']*(cp.matmul(rnn_input, self.var_dict['W_in']) \
            + self.rnn_matmul(h_post, self.W_rnn_effective, t) + self.var_dict['b_rnn']) + \
            + cp.random.normal(scale=self.con_dict['noise_rnn'], size=h.shape)

        h_out = cp.where(h > self.var_dict['threshold'], cp.ones_like(h_out), cp.zeros_like(h_out))
        h     = (1 - h_out)*h + h_out*self.var_dict['reset']

        return h_out, h, syn_x, syn_u


    def AdEx_spiking_recurrent_cell(self, h_out, V, w, rnn_input, syn_x, syn_u, t):
        """ Process one time step of the hidden layer
            based on the previous state and the current input,
            using adaptive-exponential spiking """

        h_post, syn_x, syn_u = synaptic_plasticity(h_out, syn_x, syn_u, \
            self.con_dict, par['use_stp'], par['n_hidden'])

        I = cp.matmul(rnn_input, self.var_dict['W_in']) + self.rnn_matmul(h_post, self.W_rnn_effective, t)# + self.var_dict['b_rnn']
        V, w, h_out = run_adex(V, w, I, self.con_dict['adex'])

        return h_out, V, w, syn_x, syn_u


    def judge_models(self, output_data, output_mask):
        """ Determine the loss and accuracy of each model,
            and rank them accordingly """

        self.output_data = to_gpu(output_data)
        self.output_mask = to_gpu(output_mask)

        self.task_loss = cross_entropy(self.output_mask, self.output_data, self.y)

        self.freq_loss = self.con_dict['freq_cost']*cp.abs(self.spiking_means-self.con_dict['freq_target'])

        weight_ref = self.var_dict['W_rnn'][:,:par['n_EI'],:par['n_EI']]
        self.reciprocal_loss = cp.mean((weight_ref > self.con_dict['reciprocal_threshold']) \
            * cp.transpose(weight_ref > self.con_dict['reciprocal_threshold'], [0,2,1]), axis=(1,2))
        self.reciprocal_loss = -self.con_dict['reciprocal_cost']*cp.minimum(self.con_dict['reciprocal_max'], self.reciprocal_loss)

        self.loss = self.task_loss + self.freq_loss + self.reciprocal_loss

        self.loss[cp.where(cp.isnan(self.loss))] = self.con_dict['loss_baseline']
        self.rank = cp.argsort(self.loss.astype(cp.float32))

        if par['learning_method'] == 'GA':
            for name in self.var_dict.keys():
                self.var_dict[name] = self.var_dict[name][self.rank,...]

        return to_cpu(self.loss[self.rank])


    def get_losses(self):
        """ Return the ranked loss for each loss type """
        return to_cpu({'task':self.task_loss[self.rank], \
            'freq':self.freq_loss[self.rank], 'reci':self.reciprocal_loss[self.rank]})


    def get_weights(self):
        """ Return the mean of the surviving networks' weights (post-sort) """

        return to_cpu({name:np.mean(self.var_dict[name][:par['num_survivors'],...], axis=0) for name in self.var_dict.keys()})


    def get_performance(self):
        """ Only output accuracy when requested """

        self.task_accuracy = accuracy(self.y, self.output_data, self.output_mask)
        self.full_accuracy = accuracy(self.y, self.output_data, self.output_mask, inc_fix=True)

        return to_cpu(self.task_accuracy[self.rank]), to_cpu(self.full_accuracy[self.rank])


    def breed_models(self, epsilons=None):
        """ Based on the first s networks in the ensemble,
            produce more networks slightly mutated from those s """

        epsilons = {k:0. for k in self.var_dict.keys()} if epsilons == None else to_gpu(epsilons)

        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'],par['n_networks'],par['num_survivors'])

            if par['use_crossing']:
                mate_id = to_gpu(np.random.choice(np.setdiff1d(np.arange(par['num_survivors']), s)))
                self.var_dict[name][indices,...] = cross(self.var_dict[name][s,...], self.var_dict[name][mate_id,...], \
                    par['cross_rate'])

            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0], \
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'], epsilons[name])

        self.var_dict['W_rnn'] *= self.con_dict['W_rnn_mask']


    def breed_models_evo_search(self, iteration):

        """
        Evo search without ADAM or k-NearestNeighbors weighting
        We calculate the gradient from the previous run, and then adjust the base network parameters

        self.var_dict[name][0] is considered our base network, whose parameters we will
        adjust with evolutionary search
        self.var_dict[name][1 thru N] are used to calculate the loss in a region nearby
        the base network, in order to calculate the "gradient"
        """

        learning_rate = self.con_dict['ES_learning_rate']

        for name in self.var_dict.keys():
            grad_epsilon = self.var_dict[name][1:,...] - self.var_dict[name][0:1,...]
            delta_var = grad_epsilon * self.loss[1:][:,cp.newaxis,cp.newaxis]
            self.var_dict[name][0] -= learning_rate * cp.mean(delta_var, axis=0)

            var_epsilon = cp.random.normal(0, self.con_dict['ES_sigma'], \
                size=self.var_dict[name][1::2,...].shape).astype(cp.float16)
            self.var_dict[name][1::2] = self.var_dict[name][0:1,...] + var_epsilon
            self.var_dict[name][2::2] = self.var_dict[name][0:1,...] - var_epsilon

        self.var_dict['W_rnn'] *= self.con_dict['W_rnn_mask']


    def breed_models_evo_search_with_adam(self, iteration):

        """
        Evo search with ADAM and k-NearestNeighbors weighting
        """

        self.adam_par['t'] += 1
        learning_rate = (self.con_dict['ES_learning_rate']/self.con_dict['ES_sigma'])* \
            cp.sqrt(1-self.adam_par['beta2']**self.adam_par['t'])/(1-self.adam_par['beta1']**self.adam_par['t'])

        epsilons = cp.empty([par['n_networks']-1,1])

        t0 = time.time()
        changing_flag = False
        for name in self.var_dict.keys():
            if iteration == 0:
                self.var_dict[name] = self.var_dict[name][self.rank,...]
                min = self.loss[0]
                changing_flag = True
            else:

                grad_epsilon = self.var_dict[name][1:,...] - self.var_dict[name][0:1,...]
                NN_loss = cp.mean(self.loss[1:][self.NNb_inds], axis=1)
                delta_var = cp.mean(grad_epsilon * NN_loss[:,cp.newaxis,cp.newaxis], axis=0)

                if True:
                    self.adam_par['m_' + name] = self.adam_par['beta1']*self.adam_par['m_' + name] + \
                        (1 - self.adam_par['beta1'])*delta_var
                    self.adam_par['v_' + name] = self.adam_par['beta2']*self.adam_par['v_' + name] + \
                        (1 - self.adam_par['beta2'])*delta_var*delta_var

                    self.var_dict[name][0] -= learning_rate * self.adam_par['m_' + name]/(self.adam_par['epsilon'] + \
                        cp.sqrt(self.adam_par['v_' + name]))

                else:
                    min = NN_loss.min()
                    if min < self.loss[0]:
                        ind = cp.argmin(NN_loss)
                        self.var_dict[name][0] = self.var_dict[name][1+ind,...]
                        changing_flag = True

            var_epsilon = cp.random.normal(0, self.con_dict['ES_sigma'], \
                size=self.var_dict[name][1::2,...].shape).astype(cp.float16)
            var_epsilon = cp.concatenate([var_epsilon, -var_epsilon], axis=0)
            self.var_dict[name][1:,...] = self.var_dict[name][0:1,...] + var_epsilon

            var_epsilon = cp.reshape(var_epsilon, [par['n_networks']-1,-1])
            epsilons = cp.concatenate((epsilons, var_epsilon), axis=1)

        epsilons = to_cpu(epsilons)
        self.NNbs.fit(epsilons)
        _, self.NNb_inds = self.NNbs.kneighbors(epsilons)
        self.NNb_inds = to_gpu(self.NNb_inds)

        self.var_dict['W_rnn'] *= self.con_dict['W_rnn_mask']
        #print('\nMin Loss: {:5.3f} from {:5.3f} | Will change: {}\n'.format(to_cpu(min), to_cpu(self.loss[0]), changing_flag))



def main():

    print('\nStarting model run: {}'.format(par['save_fn']))
    control = NetworkController()
    stim    = Stimulus()

    # Get loss baseline
    trial_info = stim.make_batch()
    control.run_models(trial_info['neural_input'])
    loss_baseline = np.mean(control.judge_models(trial_info['desired_output'], trial_info['train_mask']))
    control.update_constant('loss_baseline', loss_baseline)
    mean_weights_prev = control.get_weights()

    # Records
    save_record = {'iter':[], 'task_acc':[], 'full_acc':[], 'top_task_acc':[], 'top_full_acc':[], 'loss':[], \
        'mut_str':[], 'spiking':[], 'loss_factors':[]}

    for i in range(par['iterations']):

        trial_info = stim.make_batch()
        spike = control.run_models(trial_info['neural_input'])
        loss  = control.judge_models(trial_info['desired_output'], trial_info['train_mask'])[:par['num_survivors']]
        mean_weights = control.get_weights()

        mutation_strength = par['mutation_strength']*(np.mean(loss)/loss_baseline)
        mutation_strength = np.minimum(par['mutation_strength'], mutation_strength)
        thresholds = [0.1, 0.05, 0.025, 0]
        modifiers  = [1/2, 1/4, 1/8]
        for t in range(len(thresholds))[:-1]:
            if thresholds[t] > mutation_strength > thresholds[t+1]:
                mutation_strength = par['mutation_strength']*np.mean(loss)/loss_baseline * modifiers[t]
                break

        if par['use_weight_momentum']:
            weight_momentum = {}
            for name in mean_weights.keys():
                weight_momentum[name] = par['momentum_scale'] * (mean_weights[name] - mean_weights_prev[name])[np.newaxis,...]
            mean_weights_prev = mean_weights
        else:
            weight_momentum = None

        control.update_constant('mutation_strength', mutation_strength)
        if par['learning_method'] == 'GA':
            control.breed_models(epsilons=weight_momentum)
        elif par['learning_method'] == 'ES':
            control.breed_models_evo_search_with_adam(i)
        else:
            raise Exception('Unknown learning method!')

        if i%par['iters_per_output'] == 0:
            task_accuracy, full_accuracy = control.get_performance()
            loss_dict = control.get_losses()

            task_loss = np.mean(loss_dict['task'][:par['num_survivors']])
            freq_loss = np.mean(loss_dict['freq'][:par['num_survivors']])
            reci_loss = np.mean(loss_dict['reci'][:par['num_survivors']])
            top_task_acc = task_accuracy.max()
            top_full_acc = full_accuracy.max()
            task_acc  = np.mean(task_accuracy[:par['num_survivors']])
            full_acc  = np.mean(full_accuracy[:par['num_survivors']])
            curr_loss = np.mean(loss[:par['num_survivors']])
            spiking   = np.mean(spike[:par['num_survivors']])

            save_record['iter'].append(i)
            save_record['top_task_acc'].append(top_task_acc)
            save_record['top_full_acc'].append(top_full_acc)
            save_record['task_acc'].append(task_acc)
            save_record['full_acc'].append(full_acc)
            save_record['loss'].append(curr_loss)
            save_record['loss_factors'].append(loss_dict)
            save_record['mut_str'].append(mutation_strength)
            save_record['spiking'].append(spiking)
            pickle.dump(save_record, open(par['save_dir']+par['save_fn']+'.pkl', 'wb'))
            if i%(10*par['iters_per_output']) == 0:
                print('Saving weights for iteration {}... ({})'.format(i, par['save_fn']))
                pickle.dump(to_cpu(control.var_dict), open(par['save_dir']+par['save_fn']+'_weights.pkl', 'wb'))


            status_stringA = 'Iter: {:4} | Task Loss: {:5.3f} | Freq Loss: {:5.3f} | Reci Loss: {:5.3f}'.format( \
                i, task_loss, freq_loss, reci_loss)
            status_stringB = ' '*11 + '| Full Loss: {:5.3f} | Mut Str: {:7.5f} | Spiking: {:5.2f} Hz'.format( \
                curr_loss, mutation_strength, spiking)
            status_stringC = ' '*11 + '| Top Acc (Task/Full): {:5.3f} / {:5.3f}  | Mean Acc (Task/Full): {:5.3f} / {:5.3f}'.format( \
                top_task_acc, top_full_acc, task_acc, full_acc)
            print(status_stringA + '\n' + status_stringB + '\n' + status_stringC)

if __name__ == '__main__':
    main()
