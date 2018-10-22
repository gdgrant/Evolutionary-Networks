from utils import *
from parameters import par, update_dependencies
from stimulus import Stimulus

class NetworkController:

    def __init__(self):
        """ Load initial network ensemble state """

        self.make_variables()
        self.make_constants()


    def make_variables(self):
        """ Pull variables into GPU """

        var_names = ['W_in', 'W_out', 'W_rnn', 'b_rnn', 'b_out', 'h_init']
        var_names += ['threshold', 'reset'] if par['cell_type']=='LIF' else []
        self.var_dict = {}
        for v in var_names:
            self.var_dict[v] = to_gpu(par[v+'_init'])/100.


    def make_constants(self):
        """ Pull constants into GPU """

        constant_names = ['alpha_neuron', 'beta_neuron', 'noise_rnn', 'W_rnn_mask', \
            'mutation_rate', 'mutation_strength', 'cross_rate', 'EI_mask', 'loss_baseline', 'dt']
        stp_constants = ['syn_x_init', 'syn_u_init', 'U', 'alpha_stf', 'alpha_std', 'dt_sec']

        constant_names += stp_constants if par['use_stp'] else []
        constant_names += ['adex', 'w_init'] if par['spiking_cell'] == 'adex' else []
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
        syn_x  = self.con_dict['syn_x_init'] * cp.ones([par['n_networks'],par['batch_size'],1], dtype=cp.float16) if par['use_stp'] else 0.
        syn_u  = self.con_dict['syn_u_init'] * cp.ones([par['n_networks'],par['batch_size'],1], dtype=cp.float16) if par['use_stp'] else 0.
        h      = self.var_dict['h_init']     * cp.ones([par['n_networks'],par['batch_size'],1], dtype=cp.float16)
        h_out  = cp.zeros([par['n_networks'],par['batch_size'],1], dtype=cp.float16)
        if par['spiking_cell'] == 'adex':
            w  = self.con_dict['w_init']     * cp.ones([par['n_networks'],par['batch_size'],1], dtype=cp.float16)
            h  = 0.*h + self.con_dict['adex']['V_r']

        self.W_rnn_effective = apply_EI(self.var_dict['W_rnn'], self.con_dict['EI_mask'])

        spiking_means = cp.zeros([par['n_networks']])
        for t in range(par['num_time_steps']):
            if par['cell_type'] == 'rate':
                _, h, syn_x, syn_u = self.recurrent_cell(None, h, input_data[t], syn_x, syn_u)
                self.y[t,...] = cp.matmul(h, self.var_dict['W_out']) + self.var_dict['b_out']

            elif par['cell_type'] == 'LIF':
                h_out, h, syn_x, syn_u = self.LIF_spiking_recurrent_cell(h_out, h, input_data[t], syn_x, syn_u)
                self.y[t,...] = (1-self.con_dict['beta_neuron'])*self.y[t-1,...] \
                              + self.con_dict['beta_neuron']*cp.matmul(h_out, self.var_dict['W_out'])# + self.var_dict['b_out']
                self.y[t,...] = cp.minimum(relu(self.y[t,...]), 5)

            elif par['cell_type'] == 'adex':
                h_out, h, w, syn_x, syn_u = self.AdEx_spiking_recurrent_cell(h_out, h, w, input_data[t], syn_x, syn_u)
                self.y[t,...] = (1-self.con_dict['beta_neuron'])*self.y[t-1,...] \
                              + self.con_dict['beta_neuron']*cp.matmul(h_out, self.var_dict['W_out'])# + self.var_dict['b_out']
                self.y[t,...] = cp.minimum(relu(self.y[t,...]), 5)

            spiking_means += cp.mean(h_out, axis=(1,2))/self.con_dict['dt']/par['num_time_steps']

        self.h_out_mean = spiking_means
        return to_cpu(spiking_means)


    def recurrent_cell(self, h_out, h, rnn_input, syn_x, syn_u):
        """ Process one time step of the hidden layer
            based on the previous state and the current input """

        h_post, syn_x, syn_u = synaptic_plasticity(h_out, syn_x, syn_u, \
            self.con_dict, par['use_stp'], par['n_hidden'])

        h = relu((1-self.con_dict['alpha_neuron'])*h \
            + self.con_dict['alpha_neuron']*(cp.matmul(rnn_input, self.var_dict['W_in']) \
            + cp.matmul(h_post, self.W_rnn_effective) + self.var_dict['b_rnn']) + \
            + cp.random.normal(scale=self.con_dict['noise_rnn'], size=h.shape))

        return None, h, syn_x, syn_u


    def LIF_spiking_recurrent_cell(self, h_out, h, rnn_input, syn_x, syn_u):
        """ Process one time step of the hidden layer
            based on the previous state and the current input,
            using leaky integrate-and-fire spiking """

        h_post, syn_x, syn_u = synaptic_plasticity(h_out, syn_x, syn_u, \
            self.con_dict, par['use_stp'], par['n_hidden'])

        h = (1-self.con_dict['alpha_neuron'])*h \
            + self.con_dict['alpha_neuron']*(cp.matmul(rnn_input, self.var_dict['W_in']) \
            + cp.matmul(h_post, self.W_rnn_effective) + self.var_dict['b_rnn']) + \
            + cp.random.normal(scale=self.con_dict['noise_rnn'], size=h.shape)

        h_out = cp.where(h > self.var_dict['threshold'], cp.ones_like(h_out), cp.zeros_like(h_out))
        h     = (1 - h_out)*h + h_out*self.var_dict['reset']

        return h_out, h, syn_x, syn_u


    def AdEx_spiking_recurrent_cell(self, h_out, V, w, rnn_input, syn_x, syn_u):
        """ Process one time step of the hidden layer
            based on the previous state and the current input,
            using adaptive-exponential spiking """

        h_post, syn_x, syn_u = synaptic_plasticity(h_out, syn_x, syn_u, \
            self.con_dict, par['use_stp'], par['n_hidden'])

        I = cp.matmul(rnn_input, self.var_dict['W_in']) + cp.matmul(h_post, self.W_rnn_effective)# + self.var_dict['b_rnn']
        V, w, h_out = run_adex(V, w, I, self.con_dict['adex'])

        return h_out, V, w, syn_x, syn_u


    def judge_models(self, output_data, output_mask):
        """ Determine the loss and accuracy of each model,
            and rank them accordingly """

        self.output_data = to_gpu(output_data)
        self.output_mask = to_gpu(output_mask)

        self.loss = cross_entropy(self.output_mask, self.output_data, self.y)

        self.freq_loss = 1e-3*cp.abs(self.h_out_mean-20)
        self.loss += self.freq_loss

        self.loss[cp.where(cp.isnan(self.loss))] = self.con_dict['loss_baseline']
        self.rank = cp.argsort(self.loss.astype(cp.float32))

        for name in self.var_dict.keys():
            self.var_dict[name] = self.var_dict[name][self.rank,...]

        return to_cpu(self.loss[self.rank])


    def judge_models_cpu(self, output_data, output_mask):
        """ Determine the loss and accuracy of each model,
            and rank them accordingly, using cpu """

        self.loss = cross_entropy_cpu(output_mask, output_data, to_cpu(self.y))

        self.freq_loss = 1e-4*np.square(self.h_out_mean-20)
        self.loss += self.freq_loss

        self.loss[np.where(np.isnan(self.loss))] = self.con_dict['loss_baseline']
        self.rank = np.argsort(self.loss)

        for name in self.var_dict.keys():
            self.var_dict[name] = self.var_dict[name][self.rank,...]

        return self.loss


    def get_performance(self):
        """ Only output accuracy when requested """

        self.task_accuracy = accuracy(self.y, self.output_data, self.output_mask)
        self.full_accuracy = accuracy(self.y, self.output_data, self.output_mask, inc_fix=True)

        return to_cpu(self.task_accuracy[self.rank]), to_cpu(self.full_accuracy[self.rank])


    def breed_models(self):
        """ Based on the first s networks in the ensemble,
            produce more networks slightly mutated from those s """

        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'],par['n_networks'],par['num_survivors'])
            mate_id = to_gpu(np.random.choice(np.setdiff1d(np.arange(par['num_survivors']), s)))

            self.var_dict[name][indices,...] = cross(self.var_dict[name][s,...], self.var_dict[name][mate_id,...], \
                par['cross_rate'])
            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0], \
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])

        self.var_dict['W_rnn'] *= self.con_dict['W_rnn_mask']


def main():

    print('\nStarting model run: {}'.format(par['save_fn']))
    control = NetworkController()
    stim    = Stimulus()

    # Get loss baseline
    trial_info = stim.make_batch()
    control.run_models(trial_info['neural_input'])
    loss_baseline = np.mean(control.judge_models(trial_info['desired_output'], trial_info['train_mask']))
    control.update_constant('loss_baseline', loss_baseline)

    # Records
    save_record = {'iter':[], 'task_acc':[], 'full_acc':[], 'loss':[], 'mut_str':[]}

    for i in range(par['iterations']):

        trial_info = stim.make_batch()
        h_out = control.run_models(trial_info['neural_input'])
        loss  = control.judge_models(trial_info['desired_output'], trial_info['train_mask'])[:par['num_survivors']]

        mutation_strength = par['mutation_strength']*(np.mean(loss)/loss_baseline)
        mutation_strength = np.minimum(par['mutation_strength'], mutation_strength)
        thresholds = [0.1, 0.05, 0.025, 0]
        modifiers  = [1/2, 1/4, 1/8]
        for t in range(len(thresholds))[:-1]:
            if thresholds[t] > mutation_strength > thresholds[t+1]:
                mutation_strength = par['mutation_strength']*np.mean(loss)/loss_baseline * modifiers[t]
                break

        control.update_constant('mutation_strength', mutation_strength)
        control.breed_models()

        if i%par['iters_per_output'] == 0:
            task_accuracy, full_accuracy = control.get_performance()

            task_acc  = np.mean(task_accuracy[:par['num_survivors']])
            full_acc  = np.mean(full_accuracy[:par['num_survivors']])
            curr_loss = np.mean(loss[:par['num_survivors']])
            spiking   = np.mean(h_out[:par['num_survivors']])*par['dt']*1000

            save_record['iter'].append(i)
            save_record['task_acc'].append(task_acc)
            save_record['full_acc'].append(full_acc)
            save_record['loss'].append(curr_loss)
            save_record['mut_str'].append(mutation_strength)
            save_record['spiking'].append(spiking)
            pickle.dump(save_record, open(par['save_dir']+par['save_fn']+'.pkl', 'wb'))
            if i%(10*par['iters_per_output']) == 0:
                print('Saving weights for iteration {}...'.format(i))
                pickle.dump(to_cpu(control.var_dict), open(par['save_dir']+par['save_fn']+'_weights.pkl', 'wb'))

            status_string = 'Iter: {:4} | Loss: {:5.3f} | Task/Full Acc: {:5.3f} / {:5.3f} | ' +
                'Mut Str: {:5.3f} | Spiking: {:3.0f} Hz'.format(i, curr_loss, task_acc, full_acc, mutation_strength, spiking)
            print(status_string)

if __name__ == '__main__':
    main()
