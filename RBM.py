import utils
import numpy as np
import scipy as sp
import utils
import random as rnd



class RestrictedBoltzmannMachine:
    """Examples of missing features for this class are: a function to plot the weights, L2 norm (or weight decay),
    bias for both hidden and visible units, different formats of hidden units other than binary
    (binomial, gaussian, etc)."""
    def __init__(self, n_input, n_hidden, activation_fun_vis = 'Sigmoid', activation_fun_hid = 'Sigmoid',
                 visible_binom_n = 1, hidden_format = 'binary', v_bias = None, h_bias = None, dropout = True,
                 p_drop = 0.5):
        #Activation should be ReLU or Sigmoid
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.activation_fun_vis = activation_fun_vis
        self.activation_fun_hid = activation_fun_hid
        self.visible_binom_n = visible_binom_n
        self.hidden_format = hidden_format
        self.W = np.zeros([self.n_input, self.n_hidden])

        if v_bias == None:
            self.v_bias = np.zeros([self.n_input])
        else:
            self.v_bias = v_bias
        if h_bias == None:
            self.h_bias = np.zeros([self.n_hidden])
        else:
            self.h_bias = h_bias

        self.persistent_MC_hid = None
        self.momentum = np.zeros(self.W.shape)
        self.dropout_selection = np.ones(self.W.shape)
        self.dropout = dropout
        self.p_drop = p_drop



    def init_weights(self, seed = 123, var = 0.1, fun = 'uniform'):
        np.random.seed(seed)
        if fun == 'uniform':
            self.W = np.random.uniform(0.0 - var/2, 0.0 + var/2, [self.n_input, self.n_hidden])
        elif fun == 'gaussian':
            self.W = np.random.normal(0.0, var, [self.n_input, self.n_hidden])
        else:
            raise(ValueError, "only 'uniform' and 'gaussian' available")


    def activation_visible(self, x):
        if self.activation_fun_vis == 'Sigmoid':
            return(1/(1 + np.exp(-x)))
        else:
            raise(ValueError, "Only 'Sigmoid' implemented so far!")


    def activation_hidden(self, x):
        if self.activation_fun_hid == 'Sigmoid':
            return(1/(1 + np.exp(-x)))
        else:
            raise(ValueError, "Only 'Sigmoid' implemented so far!")


    def visible_to_hidden_signal(self, visible):
        selected_weights = np.multiply(self.W, self.dropout_selection)
        hidden_prob = self.activation_hidden(visible.dot(selected_weights))
        return(hidden_prob)


    def hidden_to_visible_signal(self, hidden):
        selected_weights = np.multiply(self.W, self.dropout_selection)
        hidden_prob = self.activation_visible(hidden.dot(np.transpose(selected_weights)))
        return(hidden_prob)


    def energy(self, visible, hidden):
        energy = np.inner(visible.dot(self.W), hidden) # + np.inner(hidden, self.h_bias) + np.inner(visible, self.v_bias)
        return(energy)


    def loss(self, visible, hidden):
        loss = self.energy(visible, hidden)+ self.lambda_w * np.abs(self.W).sum()
        return(loss)

    def free_energy(self, visible):
        wx_b = visible.dot(self.W) + self.h_bias
        vbias_term = np.inner(visible, self.v_bias)
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis = 1)
        energy = -hidden_term - vbias_term
        return(energy)


    def reconstruction(self, probs):
        U = np.random.uniform(0, 1, [probs.shape[0], probs.shape[1], self.visible_binom_n])
        visible_rebuilt = 0 + sp.less(U, probs.reshape([probs.shape[0], probs.shape[1], -1]))
        return(visible_rebuilt.sum(2))


    def hidden_sampling(self, probs):
        U = np.random.uniform(0, 1, probs.shape)
        hidden = 0+sp.less(U, probs)
        return(hidden)


    def contrastive_divergence(self, visible, cd_num = 1, reconstruction = False):
        if self.dropout:
            self.dropout_selection = np.random.binomial(1, self.p_drop, np.shape(self.W))
        if reconstruction:
            probs = self.visible_to_hidden_signal(visible)
            hidden = self.hidden_sampling(probs)
            cor_dat = np.einsum('ij,ik->ijk', visible, probs)
            for i in range(cd_num):
                recon_visible = self.reconstruction(self.hidden_to_visible_signal(hidden))
                h_probs = self.visible_to_hidden_signal(recon_visible)
                hidden = self.hidden_sampling(h_probs)

            cor_mod = np.einsum('ij,ik->ijk', recon_visible, h_probs)
            gradient = cor_mod.mean(0) - cor_dat.mean(0)

        else:
            probs = self.visible_to_hidden_signal(visible)
            hidden = self.hidden_sampling(probs)
            cor_dat = np.einsum('ij,ik->ijk', visible, probs)
            for i in range(cd_num):
                recon_visible = self.hidden_to_visible_signal(hidden)
                recon_visible = recon_visible*self.visible_binom_n
                h_probs = self.visible_to_hidden_signal(recon_visible)
                hidden = self.hidden_sampling(h_probs)

            cor_mod = np.einsum('ij,ik->ijk', recon_visible, h_probs)
            gradient = cor_mod.mean(0) - cor_dat.mean(0)

        return(gradient)



    def persistent_contrastive_divergence(self, visible, cd_num = 1, reconstruction = False):
        if self.dropout:
            self.dropout_selection = np.random.binomial(1, self.p_drop, np.shape(self.W))

        if reconstruction:
            probs = self.visible_to_hidden_signal(visible)
            hidden = self.hidden_sampling(probs)
            cor_dat = np.einsum('ij,ik->ijk', visible, probs)
            persistent_hidden = self.persistent_MC_hid
            for i in range(cd_num):
                recon_visible = self.hidden_to_visible_signal(persistent_hidden)
                if self.activation_fun_vis == 'Sigmoid':
                    recon_visible = recon_visible*self.visible_binom_n
                h_probs = self.visible_to_hidden_signal(recon_visible)
                hidden = self.hidden_sampling(h_probs)
            self.persistent_MC_hid = hidden
            cor_mod = np.einsum('ij,ik->ijk', recon_visible, h_probs)
            gradient = cor_mod.mean(0) - cor_dat.mean(0)
            self.dropout_selection = np.ones(self.W.shape)

        else:

            probs = self.visible_to_hidden_signal(visible)
            hidden = self.hidden_sampling(probs)
            cor_dat = np.einsum('ij,ik->ijk', visible, probs)
            persistent_hidden = self.persistent_MC_hid
            for i in range(cd_num):
                recon_visible = self.hidden_to_visible_signal(persistent_hidden)
                if self.activation_fun_vis == 'Sigmoid':
                    recon_visible = recon_visible*self.visible_binom_n
                h_probs = self.visible_to_hidden_signal(recon_visible)
                hidden = self.hidden_sampling(h_probs)
            self.persistent_MC_hid = hidden
            cor_mod = np.einsum('ij,ik->ijk', recon_visible, h_probs)
            gradient = cor_mod.mean(0) - cor_dat.mean(0)
            self.dropout_selection = np.ones(self.W.shape)

        return(gradient)


    def initialize_persistent_chain(self, visible):
        probs = self.visible_to_hidden_signal(visible)
        self.persistent_MC_hid = self.hidden_sampling(probs)


    def daydreaming(self, visible, gibbs_steps = 100):
        """Basically Gibbs Sampling with 'live' show of the plots, but for now can only plot 4 images at once"""

        probs = self.visible_to_hidden_signal(visible)
        hidden = self.hidden_sampling(probs)
        for i in range(gibbs_steps):
            visible_p = self.hidden_to_visible_signal(hidden)
            recon_visible = self.reconstruction(visible_p)
            pixels0 = recon_visible
            pixels0 = np.vstack([np.hstack([pixels0[0].reshape([28,28]), pixels0[1].reshape([28,28])]),
                                np.hstack([pixels0[2].reshape([28,28]), pixels0[3].reshape([28,28])])])
            utils.show(pixels0, figure_n=2, pause_time=0.005)
            h_probs = self.visible_to_hidden_signal(recon_visible)
            hidden = self.hidden_sampling(h_probs)
        visible_p = self.hidden_to_visible_signal(hidden)
        recon_visible = self.reconstruction(visible_p)

        return(recon_visible)


    #Remember to ad regularization at some point
    def train(self, data, data_test, test_dim, alpha = 0.1, lambda_w = 0.0,
              batch_dim = 50, batches_n = 20, epochs = 10, seed = 123, recon_logical = False,
              persistent = True, initialize_persistent = True, momentum_coef = 0.5, daydream = True,
              gibbs_lag = 1):
        np.random.seed(seed)
        self.lambda_w = lambda_w
        for j in range(1, epochs + 1):
            for i in range(batches_n):
                #Select a batch of data, one epoch doesn't go through the whole data but through a random selection of
                #k batches of n elements each
                idx = np.random.choice(np.arange(np.shape(data)[0]), batch_dim)
                batch = data[idx, :]

                #depending on the method chosen, the gradient approximation is estimated
                if persistent:
                    #The persistent markov chain is initialized at the beginning of the training only.
                    if initialize_persistent:
                        self.initialize_persistent_chain(batch)
                        initialize_persistent = False
                    grad = self.persistent_contrastive_divergence(batch, cd_num = j , reconstruction=recon_logical) +\
                           self.lambda_w*np.sign(self.W)

                else:
                    grad = self.contrastive_divergence(batch, cd_num = j, reconstruction = recon_logical) +\
                           self.lambda_w*np.sign(self.W)

                self.momentum = momentum_coef*self.momentum + grad
                self.W = self.W - alpha*self.momentum

            data_test_idx = np.random.choice(np.arange(np.shape(data_test)[0]), test_dim)
            data_test_block = data_test[data_test_idx,:]
            print(self.energy(visible = data_test_block, hidden = data_test_block.dot(self.W)).mean())
            self.dropout_selection = np.ones(self.W.shape)
            if daydream:
                k = [rnd.randint(0, 5000), rnd.randint(0, 5000), rnd.randint(0, 5000), rnd.randint(0, 5000)]
                pixels0 = data_test[k]
                pixels0 = np.vstack([np.hstack([pixels0[0].reshape([28,28]), pixels0[1].reshape([28,28])]),
                                     np.hstack([pixels0[2].reshape([28,28]), pixels0[3].reshape([28,28])])])
                utils.show(pixels0, figure_n=1, pause_time=0.005)
                dat = data_test[k]
                self.daydreaming(dat,  gibbs_steps=gibbs_lag)



    