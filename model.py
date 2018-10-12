from utils import *
from parameters import par, update_dependencies
from stimulus import Stimulus

class NetworkController:

    def __init__(self):
        pass


    def make_variables(self):
        """ Pull variables into GPU """

        var_names = ['W_in', 'W_out', 'W_rnn', 'b_rnn', 'b_out', 'h_init']
        self.var_dict = {}
        for v in var_names:
            self.var_dict[v] = to_gpu(par[v+'_init'])


    def make_constants(self):
        """ Pull constants into GPU """

        constant_names = ['alpha_neuron', 'noise_rnn', 'W_rnn_mask', \
            'mutation_rate', 'mutation_strength', 'cross_rate']
        stp_constants = ['syn_x_init', 'syn_u_init', 'U', 'alpha_stf', 'alpha_std', 'dt_sec']

        constant_names += stp_constants if par['use_stp'] else []
        self.con_dict = {}
        for c in constant_names:
            self.con_dict[c] = to_gpu(par[c])


    def update_mutation_constants(self, strength, rate):
        """ Update the mutation strength and rate """

        self.con_dict['mutation_strength'] = to_gpu(strength)
        self.con_dict['mutation_rate']     = to_gpu(rate)


    def run_models(self, input_data):
        """ Run model based on input data, collecting
            network states into h and network outputs into y """

        input_data = to_gpu(input_data)

        self.h     = cp.zeros([par['num_time_steps'], par['n_networks'], par['batch_size'], par['n_hidden']])
        self.syn_x = cp.zeros([par['num_time_steps'], par['n_networks'], par['batch_size'], par['n_hidden']])
        self.syn_u = cp.zeros([par['num_time_steps'], par['n_networks'], par['batch_size'], par['n_hidden']])

        # Put init in last time step, to be overwritten at end of trial
        self.h[-1,:,:,:]        = self.var_dict['h_init']
        self.syn_x[-1,:,:,:]    = self.con_dict['syn_x_init']
        self.syn_u[-1,:,:,:]    = self.con_dict['syn_u_init']

        # input_data = [time, networks, trials, neurons]
        for t in range(par['num_time_steps']):
            self.h[t,:,:,:], self.syn_x[t,:,:,:], self.syn_u[t,:,:,:] = \
                self.recurrent_cell(self.h[t-1,:,:,:], input_data[t,:,:,:], self.syn_x[t-1,:,:,:], self.syn_u[t-1,:,:,:])

        self.y = cp.matmul(self.h, self.var_dict['W_out']) + self.var_dict['b_out']


    def recurrent_cell(self, h, rnn_input, syn_x, syn_u):
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
            + cp.matmul(h_post, self.var_dict['W_rnn']) + self.var_dict['b_rnn']) + \
            + cp.random.normal(scale=self.con_dict['noise_rnn'], size=h.shape))

        return h, syn_x, syn_u


    def judge_models(self, output_data, output_mask):
        """ Determine the loss and accuracy of each model,
            and rank them accordingly """

        self.output_data = to_gpu(output_data)
        self.output_mask = to_gpu(output_mask)
        eps = 1e-7

        self.loss = -cp.mean(self.output_mask[...,cp.newaxis]*softmax(self.y)*cp.log(self.output_data+eps), axis=(0,2,3))
        self.rank = cp.argsort(self.loss)

        for name in self.var_dict.keys():
            self.var_dict[name] = self.var_dict[name][self.rank,...]

        return to_cpu(self.loss)


    def get_performance(self):
        """ Only output accuracy when requested """

        self.accuracy = accuracy(self.y, self.output_data, self.output_mask)
        return to_cpu(self.accuracy)


    def breed_models(self):
        """ Based on the first s networks in the ensemble,
            produce more networks slightly mutated from those s """

        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'],par['n_networks'],par['num_survivors'])
            m       = to_gpu(np.random.choice(np.setdiff1d(np.arange(par['num_survivors']), s)))

            self.var_dict[name][indices,...] = cross(self.var_dict[name][s,...], self.var_dict[name][m,...], \
                par['cross_rate'])
            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0], \
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])


def main():

    control = NetworkController()
    control.make_variables()
    control.make_constants()

    stim = Stimulus()

    # Get loss baseline
    trial_info = stim.make_batch()
    control.run_models(trial_info['neural_input'])
    loss_baseline = np.mean(control.judge_models(trial_info['desired_output'], trial_info['train_mask']))

    for i in range(par['iterations']):

        trial_info = stim.make_batch()
        control.run_models(trial_info['neural_input'])
        loss = control.judge_models(trial_info['desired_output'], trial_info['train_mask'])

        mutation_strength = par['mutation_strength']*np.mean(loss)/loss_baseline
        control.update_mutation_constants(mutation_strength, par['mutation_rate'])
        control.breed_models()

        if i%par['iters_per_output'] == 0:
            accuracy = control.get_performance()
            print('Iter: {:4} | Loss: {:5.3f} | Acc: {:5.3f} | Mut. Str.: {:5.3f}'.format( \
            i, np.mean(loss[:par['num_survivors']]), np.mean(accuracy[:par['num_survivors']]), \
            mutation_strength))


if __name__ == '__main__':
    main()
