



### Record model states while running
    def run_and_record_models(self, input_data):
        """ Run model based on input data, collecting
            network states into h and network outputs into y """

        input_data = to_gpu(input_data)

        self.h     = cp.zeros([par['num_time_steps'], par['n_networks'], par['batch_size'], par['n_hidden']], dtype=cp.float16)
        self.syn_x = cp.zeros([par['num_time_steps'], par['n_networks'], par['batch_size'], par['n_hidden']], dtype=cp.float16)
        self.syn_u = cp.zeros([par['num_time_steps'], par['n_networks'], par['batch_size'], par['n_hidden']], dtype=cp.float16)

        # Put init in last time step, to be overwritten at end of trial
        self.h[-1,:,:,:]        = self.var_dict['h_init']
        self.syn_x[-1,:,:,:]    = self.con_dict['syn_x_init']
        self.syn_u[-1,:,:,:]    = self.con_dict['syn_u_init']

        self.W_rnn_effective = apply_EI(self.var_dict['W_rnn'], self.con_dict['EI_mask'])

        # input_data = [time, networks, trials, neurons]
        for t in range(par['num_time_steps']):
            self.h[t,:,:,:], self.syn_x[t,:,:,:], self.syn_u[t,:,:,:] = \
                self.recurrent_cell(self.h[t-1,:,:,:], input_data[t,:,:,:], self.syn_x[t-1,:,:,:], self.syn_u[t-1,:,:,:])

        self.y = cp.matmul(self.h, self.var_dict['W_out']) + self.var_dict['b_out']
