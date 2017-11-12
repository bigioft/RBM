import utils
import numpy as np
import RideByMe

training_data = list(utils.read(dataset = 'training', img_and_lbl = False, only_lbl = False, path = './mnist'))
test_data = list(utils.read(dataset = 'testing', img_and_lbl = False, only_lbl = False, path = './mnist'))

td_reshaped = np.array(training_data).reshape([60000, 28*28])
testd_reshaped = np.array(test_data).reshape([10000, 28*28])


RBM = RideByMe.RestrictedBoltzmannMachine(28*28, 500, visible_binom_n = 255, activation_fun_vis = 'Sigmoid',
                                          activation_fun_hid = 'Sigmoid', hidden_format = 'binary', dropout = True, p_drop = 0.5)
RBM.init_weights(seed = 123, var = 0.01, fun = 'gaussian')

RBM.train(data = td_reshaped, data_test = testd_reshaped, test_dim = 500, alpha = 0.0001, lambda_w=0.0,
          batch_dim = 10, batches_n = 500, epochs = 4, seed = 345, recon_logical = False,
          persistent = True, initialize_persistent = True, momentum_coef = 0.5, daydream = False, gibbs_lag = 0)


#Additional training with plots
RBM.train(data = td_reshaped, data_test = testd_reshaped, test_dim = 500, alpha = 0.00005, lambda_w=0.0000001,
          batch_dim = 20, batches_n = 50, epochs = 4, seed = 345, recon_logical = False,
          persistent = True, initialize_persistent = True, momentum_coef = 0.9, daydream = True, gibbs_lag = 20)


#Show some daydreaming
RBM.daydreaming(testd_reshaped[[1,2,3,4]], gibbs_steps = 100)