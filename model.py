from utils import *
from parameters import par, update_dependencies
from stimulus import Stimulus

class NetworkController:

    def __init__(self):
        pass


    def make_variables(self):
        """ Pull variables into GPU """

        var_names = ['W_in', 'W_out', 'W_rnn', 'b_rnn', 'b_out', 'h_init']
        var_names += ['threshold', 'reset'] if par['network_type']=='spiking' else []
        self.var_dict = {}
        for v in var_names:
            self.var_dict[v] = to_gpu(par[v+'_init'])


    def make_constants(self):
        """ Pull constants into GPU """

        constant_names = ['alpha_neuron', 'noise_rnn', 'W_rnn_mask', \
            'mutation_rate', 'mutation_strength', 'cross_rate', 'EI_mask', 'loss_baseline']
        stp_constants = ['syn_x_init', 'syn_u_init', 'U', 'alpha_stf', 'alpha_std', 'dt_sec']

        constant_names += stp_constants if par['use_stp'] else []
        self.con_dict = {}
        for c in constant_names:
            self.con_dict[c] = to_gpu(par[c])


    def update_constant(self, con_name, con):
        """ Update a given constant in the model """

        self.con_dict[con_name] = to_gpu(con)


    def update_mutation_constants(self, strength, rate):
        """ Update the mutation strength and rate """

        self.con_dict['mutation_strength'] = to_gpu(strength)
        self.con_dict['mutation_rate']     = to_gpu(rate)


    def run_models(self, input_data):
        """ Run model based on input data, collecting network outputs into y """

        input_data = to_gpu(input_data)

        self.y = cp.zeros([par['num_time_steps'], par['n_networks'], par['batch_size'], par['n_output']], dtype=cp.float16)
        syn_x  = self.con_dict['syn_x_init'] * cp.ones([par['n_networks'],par['batch_size'],1], dtype=cp.float16)
        syn_u  = self.con_dict['syn_u_init'] * cp.ones([par['n_networks'],par['batch_size'],1], dtype=cp.float16)
        h      = self.var_dict['h_init']     * cp.ones([par['n_networks'],par['batch_size'],1], dtype=cp.float16)
        h_out  = cp.zeros([par['n_networks'],par['batch_size'],1], dtype=cp.float16)

        self.W_rnn_effective = apply_EI(self.var_dict['W_rnn'], self.con_dict['EI_mask'])

        if par['network_type'] == 'rate_based':
            for t in range(par['num_time_steps']):
                _, h, syn_x, syn_u = self.recurrent_cell(None, h, input_data[t], syn_x, syn_u)
                self.y[t,...] = cp.matmul(h, self.var_dict['W_out']) + self.var_dict['b_out']

        elif par['network_type'] == 'spiking':
            for t in range(par['num_time_steps']):
                h_out, h, syn_x, syn_u = self.LIF_spiking_recurrent_cell(h_out, h, input_data[t], syn_x, syn_u)
                self.y[t,...] = (1-self.con_dict['alpha_neuron'])*self.y[t-1,...] \
                              + self.con_dict['alpha_neuron']*(cp.matmul(h_out, self.var_dict['W_out']) + self.var_dict['b_out'])


    def recurrent_cell(self, h_out, h, rnn_input, syn_x, syn_u):
        """ Process one time step of the hidden layer
            based on the previous state and the current input """

        if par['use_stp']:
            syn_x += self.con_dict['alpha_std']*(1-syn_x) - self.con_dict['dt_sec']*syn_u*syn_x*h
            syn_u += self.con_dict['alpha_stf']*(self.con_dict['U']-syn_x) - self.con_dict['dt_sec']*self.con_dict['U']*(1-syn_u)*h
            syn_x = cp.minimum(1., relu(syn_x))
            syn_u = cp.minimum(1., relu(syn_u))
            h_post = syn_u*syn_x*h
        else:
            h_post = h

        h = relu((1-self.con_dict['alpha_neuron'])*h \
            + self.con_dict['alpha_neuron']*(cp.matmul(rnn_input, self.var_dict['W_in']) \
            + cp.matmul(h_post, self.W_rnn_effective) + self.var_dict['b_rnn']) + \
            + cp.random.normal(scale=self.con_dict['noise_rnn'], size=h.shape))

        return None, h, syn_x, syn_u


    def LIF_spiking_recurrent_cell(self, h_out, h, rnn_input, syn_x, syn_u):
        """ Process one time step of the hidden layer
            based on the previous state and the current input,
            using leaky integrate-and-fire spiking """

        if par['use_stp']:
            syn_x += self.con_dict['alpha_std']*(1-syn_x) - self.con_dict['dt_sec']*syn_u*syn_x*h_out
            syn_u += self.con_dict['alpha_stf']*(self.con_dict['U']-syn_x) - self.con_dict['dt_sec']*self.con_dict['U']*(1-syn_u)*h_out
            syn_x = cp.minimum(1., relu(syn_x))
            syn_u = cp.minimum(1., relu(syn_u))
            h_post = syn_u*syn_x*h
        else:
            h_post = h_out

        h = (1-self.con_dict['alpha_neuron'])*h \
            + self.con_dict['alpha_neuron']*(cp.matmul(rnn_input, self.var_dict['W_in']) \
            + cp.matmul(h_post, self.W_rnn_effective) + self.var_dict['b_rnn']) + \
            + cp.random.normal(scale=self.con_dict['noise_rnn'], size=h.shape)

        h_out = cp.where(h > self.var_dict['threshold'], cp.ones_like(h_out), cp.zeros_like(h_out))
        h     = (1 - h_out)*h + h_out*self.var_dict['reset']

        return h_out, h, syn_x, syn_u


    def judge_models(self, output_data, output_mask):
        """ Determine the loss and accuracy of each model,
            and rank them accordingly """

        self.output_data = to_gpu(output_data)
        self.output_mask = to_gpu(output_mask)
        eps = 1e-7

        self.loss = -cp.mean(self.output_mask[...,cp.newaxis]*softmax(self.y)*cp.log(self.output_data+eps), axis=(0,2,3))
        self.loss[cp.where(cp.isnan(self.loss))] = self.con_dict['loss_baseline']

        self.rank = cp.argsort(self.loss.astype(cp.float32))

        for name in self.var_dict.keys():
            self.var_dict[name] = self.var_dict[name][self.rank,...]

        return to_cpu(self.loss)


    def get_performance(self):
        """ Only output accuracy when requested """

        self.task_accuracy = accuracy(self.y, self.output_data, self.output_mask)
        self.full_accuracy = accuracy(self.y, self.output_data, self.output_mask, inc_fix=False)
        return to_cpu(self.task_accuracy), to_cpu(self.full_accuracy)


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
    control.make_variables()
    control.make_constants()

    stim = Stimulus()

    # Get loss baseline
    trial_info = stim.make_batch()
    control.run_models(trial_info['neural_input'])
    loss_baseline = np.mean(control.judge_models(trial_info['desired_output'], trial_info['train_mask']))
    control.update_constant('loss_baseline', loss_baseline)

    # Records
    save_record = {
        'iter'      : [],
        'task_acc'  : [],
        'full_acc'  : [],
        'loss'      : [],
        'mut_str'   : []
    }

    for i in range(par['iterations']):

        trial_info = stim.make_batch()
        control.run_models(trial_info['neural_input'])
        loss = control.judge_models(trial_info['desired_output'], trial_info['train_mask'])

        mutation_strength = par['mutation_strength']*np.mean(loss)/loss_baseline
        if np.mean(loss)/loss_baseline < 0.025:
            mutation_strength = par['mutation_strength']*np.mean(loss)/loss_baseline * (1/8)
        elif np.mean(loss)/loss_baseline < 0.05:
            mutation_strength = par['mutation_strength']*np.mean(loss)/loss_baseline * (1/4)
        elif np.mean(loss)/loss_baseline < 0.1:
            mutation_strength = par['mutation_strength']*np.mean(loss)/loss_baseline * (1/2)
        control.update_mutation_constants(mutation_strength, par['mutation_rate'])
        control.breed_models()

        if i%par['iters_per_output'] == 0:
            task_accuracy, full_accuracy = control.get_performance()

            save_record['iter'].append(i)
            save_record['task_acc'].append(np.mean(task_accuracy[:par['num_survivors']]))
            save_record['full_acc'].append(np.mean(full_accuracy[:par['num_survivors']]))
            save_record['loss'].append(np.mean(loss[:par['num_survivors']]))
            save_record['mut_str'].append(mutation_strength)
            pickle.dump(save_record, open(par['save_dir']+par['save_fn']+'.pkl', 'wb'))

            print('Iter: {:4} | Loss: {:5.3f} | Task Acc: {:5.3f} | Full Acc: {:5.3f} | Mut. Str.: {:5.3f}'.format( \
                i, np.mean(loss[:par['num_survivors']]), np.mean(task_accuracy[:par['num_survivors']]), \
                np.mean(full_accuracy[:par['num_survivors']]), mutation_strength))


if __name__ == '__main__':
    main()
