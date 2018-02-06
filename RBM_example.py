import utils
import numpy as np
import RBM

training_data = list(utils.read(dataset = 'training', img_and_lbl = False, only_lbl = False, path = './mnist'))
test_data = list(utils.read(dataset = 'testing', img_and_lbl = False, only_lbl = False, path = './mnist'))

td_reshaped = np.array(training_data).reshape([60000, 28*28])
testd_reshaped = np.array(test_data).reshape([10000, 28*28])


RBM = RBM.RestrictedBoltzmannMachine(28*28, 500, visible_binom_n = 255, activation_fun_vis = 'Sigmoid',
                                          activation_fun_hid = 'Sigmoid', hidden_format = 'binary', dropout = True, p_drop = 0.3)
RBM.init_weights(seed = 123, var = 0.0001, fun = 'gaussian')


RBM.train(data = td_reshaped, data_test = testd_reshaped, test_dim = 5000, alpha = 0.000005, lambda_w=0.0,
          batch_dim = 50, batches_n = 200, epochs = 10, seed = 345, recon_logical = False, persistent = False,
          initialize_persistent = False, momentum_coef = 0.5, daydream = False, gibbs_lag = 0, loss_plot =True,
          cd_num = 10)

#Additional training with plots
RBM.train(data = td_reshaped, data_test = testd_reshaped, test_dim = 500, alpha = 0.000005, lambda_w=0.0,
          batch_dim = 50, batches_n = 10, epochs = 4, seed = 345, recon_logical = False, persistent = False,
          initialize_persistent = False, momentum_coef = 0.9, daydream = True, gibbs_lag = 20, loss_plot = False)


#Show some daydreaming
RBM.daydreaming(testd_reshaped[[1,2,3,4]], gibbs_steps = 100)