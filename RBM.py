import utils
import numpy as np
import scipy as sp
import utils
import random as rnd
import matplotlib.pyplot as plt



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
        self.dropout = dropout
        self.p_drop = p_drop
        self.obj = []



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
            return(1 / (1 + np.exp(-x)))
        else:
            raise(ValueError, "Only 'Sigmoid' implemented so far!")


    def activation_hidden(self, x):
        if self.activation_fun_hid == 'Sigmoid':
            return(1 / (1 + np.exp(-x)))
        else:
            raise(ValueError, "Only 'Sigmoid' implemented so far!")


    def visible_to_hidden_signal(self, visible):
        #Function to calculate hidden probabilities from visible state
        hidden_prob = self.activation_hidden(visible.dot(self.W))
        return(hidden_prob)


    def hidden_to_visible_signal(self, hidden):
        #Function to calculate visible probabilities from hidden state
        hidden_prob = self.activation_visible(hidden.dot(np.transpose(self.W)))
        return(hidden_prob)


    def config_goodness(self, visible, hidden = None):
        if hidden is None:
            hidden = self.activation_hidden(np.transpose(visible.dot(self.W)))
        goodness = -visible.dot(self.W).dot(hidden)
        return(goodness.mean())

    def reconstruction(self, probs):
        U = np.random.uniform(0, 1, [probs.shape[0], probs.shape[1], self.visible_binom_n])
        visible_rebuilt = 0 + sp.less(U, probs.reshape([probs.shape[0], probs.shape[1], -1]))
        return(visible_rebuilt.sum(2))


    def hidden_sampling(self, probs):
        U = np.random.uniform(0, 1, probs.shape)
        hidden = 0+sp.less(U, probs)
        return(hidden)


    def contrastive_divergence(self, visible, persistent = True, cd_num = 1, reconstruction = False):
        if self.dropout:
            dropout_selection = np.random.binomial(1, 1 - self.p_drop, self.n_hidden)


        probs = self.visible_to_hidden_signal(visible)
        hidden = dropout_selection * self.hidden_sampling(probs)
        cor_dat = np.einsum('ij,ik->ijk', visible, probs)
        persistent_hidden = self.persistent_MC_hid
        for i in range(cd_num):
            #Calculate probabilities of the next visible layer
            if persistent:
                #If using persistent CD, those come from the persistent underlying MC
                recon_visible = self.hidden_to_visible_signal(self.persistent_MC_hid)
            elif not persistent:
                # If using classical CD, those come from the currently calculated hidden value
                recon_visible = self.hidden_to_visible_signal(hidden)
            else:
                raise(ValueError, "persistent has to be either True or False, no other value admitted.")

            #Additional randomness by reconstructing the visible layer every time
            if reconstruction:
                recon_visible = self.reconstruction(recon_visible)

            #Scale a probability up by the N of the binomial
            elif (not reconstruction) &  (self.activation_fun_vis == 'Sigmoid'):
                recon_visible = recon_visible * self.visible_binom_n

            #Raise error for any other implementation
            else:
                raise(ValueError, "reconstuction != True & self.activation_fun_vis != Sigmoid not available!")

            #Calculate probabilities of next hidden layer
            h_probs = self.visible_to_hidden_signal(recon_visible)
            #Sample from those probabilities
            hidden = self.hidden_sampling(h_probs)


        self.persistent_MC_hid = hidden

        cor_mod = np.einsum('ij,ik->ijk', recon_visible, h_probs)
        gradient = cor_mod.mean(0) - cor_dat.mean(0)

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
              gibbs_lag = 1, loss_plot = True, cd_num = 10):
        np.random.seed(seed)
        self.lambda_w = lambda_w
        for j in range(1, epochs + 1):
            for i in range(batches_n):
                #Select a batch of data, one epoch doesn't go through the whole data but through a random selection of
                #k batches of n elements each
                idx = np.random.choice(np.arange(np.shape(data)[0]), batch_dim)
                batch = data[idx, :]

                #depending on the method chosen, the gradient approximation is estimated
                #The underlying persistent MC is initialized if necessary
                if initialize_persistent or (persistent and (self.persistent_MC_hid is None)):
                    self.initialize_persistent_chain(batch)
                    initialize_persistent = False


                #Get the gradient
                grad = self.contrastive_divergence(batch, persistent, cd_num, recon_logical) +\
                        self.lambda_w * np.sign(self.W)

                #Use momentum to speed up training
                self.momentum = momentum_coef * self.momentum + grad
                #Update the weights
                self.W = self.W - alpha * self.momentum

            data_test_idx = np.random.choice(np.arange(np.shape(data_test)[0]), test_dim)
            data_test_block = data_test[data_test_idx,:]

            goodness_of_config = self.config_goodness(visible = data_test_block, hidden = None)
            if loss_plot:
                self.obj.append(goodness_of_config)
                self.plot_obj(self.obj, test_dim)

            print(goodness_of_config)
            if daydream:
                k = [rnd.randint(0, 5000), rnd.randint(0, 5000), rnd.randint(0, 5000), rnd.randint(0, 5000)]
                pixels0 = data_test[k]
                pixels0 = np.vstack([np.hstack([pixels0[0].reshape([28,28]), pixels0[1].reshape([28,28])]),
                                     np.hstack([pixels0[2].reshape([28,28]), pixels0[3].reshape([28,28])])])
                utils.show(pixels0, figure_n=1, pause_time=0.005)
                dat = data_test[k]
                self.daydreaming(dat,  gibbs_steps=gibbs_lag)


    def plot_obj(self, obj, test_dim):
        #Obj are the values of the objective function, i.e. configuration goodness
        idx = np.arange(len(obj))
        fig = plt.figure(3)
        plt.plot(idx, obj)
        fig.suptitle("Configuration Goodness over epoch for " + str(test_dim) + " test elements.")
        plt.ylabel("Configuration Goodness")
        plt.xlabel("Epoch")
        fig.show()
        plt.pause(0.01)

    