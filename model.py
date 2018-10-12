from utils import *
from parameters import par, update_dependencies


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
            'mutation_rate', 'mutation_strength']
        self.con_dict = {}
        for c in constant_names:
            self.con_dict[c] = to_gpu(par[c])


    def run_models(self, input_data):
        """ Run model based on input data, collecting
            network states into h and network outputs into y """

        input_data = to_gpu(input_data)
        self.h = cp.zeros([par['num_time_steps'], par['n_networks'], par['batch_size'], par['n_hidden']])
        self.h[-1,:,:,:] = self.var_dict['h_init']   # Put init in last time step, to be overwritten at end of trial

        # input_data = [time, networks, trials, neurons]
        for t in range(par['num_time_steps']):
            self.h[t,:,:,:] = self.recurrent_cell(self.h[t-1,:,:,:], input_data[t,:,:,:])

        self.y = cp.matmul(self.h, self.var_dict['W_out']) + self.var_dict['b_out']


    def recurrent_cell(self, h, rnn_input):
        """ Process one time step of the hidden layer
            based on the previous state and the current input """

        h = relu((1-self.con_dict['alpha_neuron'])*h \
            + self.con_dict['alpha_neuron']*(cp.matmul(rnn_input, self.var_dict['W_in']) \
            + cp.matmul(h, self.var_dict['W_rnn']) + self.var_dict['b_rnn']) + \
            + cp.random.normal(scale=self.con_dict['noise_rnn'], size=h.shape, dtype=np.float32))

        return h


    def judge_models(self, output_data, output_mask):
        """ Determine the loss and accuracy of each model,
            and rank them accordingly """

        output_data = to_gpu(output_data)
        output_mask = to_gpu(output_mask)
        eps = 1e-7

        self.loss = -cp.sum(softmax(self.y)*cp.log(output_data+eps), axis=(0,2,3))
        self.rank = cp.argsort(self.loss)

        for name in self.var_dict.keys():
            self.var_dict[name] = self.var_dict[name][self.rank,...]


    def breed_models(self):
        """ Based on the first s networks in the ensemble,
            produce more networks slightly mutated from those s """

        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'],par['n_networks'],par['num_survivors'])
            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0], \
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])


def main():

    control = NetworkController()
    control.make_variables()
    control.make_constants()
    control.run_models(np.ones([par['num_time_steps'],par['n_networks'],par['batch_size'],par['n_input']]))
    control.judge_models(np.ones([par['num_time_steps'],par['n_networks'],par['batch_size'],par['n_output']]),
        np.ones([par['num_time_steps'],par['n_networks'],par['batch_size'],par['n_output']]))
    control.breed_models()



if __name__ == '__main__':
    main()
