import numpy as onp
import jax.numpy as np
import pandas as pd

from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.nn import relu
from jax.config import config

import itertools
from functools import partial
from tqdm import trange
import matplotlib.pyplot as plt

from NN_modules_tanh import singleBranchTrain, DeepONet_2ST

import scipy
import scipy.io

load_path   = "/share/deeponet2st/yshin8/don_2st_choo/new_data2"
save_path   = "/share/deeponet2st/yshin8/don_2st_choo/save"

# config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')
# config.update('XLA_PYTHON_CLIENT_PREALLOCATE', False)
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

t_batch = 1225
#t_batch  = 2447
b_batch  = 2447


data_ALL = scipy.io.loadmat(f"{load_path}/ExChoo_DATA_NEW2.mat")
F_tr     = data_ALL['F_tr']
U_tr     = data_ALL['y_tr']
F_tt     = data_ALL['F_tt']
U_tt     = data_ALL['y_tt']
Nx_grid  = data_ALL['Nx_grid']

[N_train, mx] = F_tr.shape
[my, N_test] = U_tt.shape
N_all    = N_test + N_train

from datetime import datetime
now       = datetime.now()
train_num = now.strftime("%Y%m%d_%H%M%S")

t_seed_list = [1234, 9545, 2808, 8806, 1317, 7046, 6801, 5688, 8339]
b_seed_list = [4321, 6006, 2083, 4532, 8569, 1217, 8730,   76, 640]

all_training_properties = {
    "t_epochs": 100000,
    "t_batch":t_batch,
    "t_lr": 1e-3,
    "t_scheduler_step": 1000,
    "t_scheduler_rate": 0.99,
    "t_seed": 1234,
    "t_activation": 'tanh',
    "b_epochs": 100000,
    "b_batch": b_batch,
    "b_lr": 1e-3,
    "b_scheduler_step": 1000,
    "b_scheduler_rate": 0.99,
    "b_seed": 4321,
    "b_activation": 'tanh',
    "N_train": N_train,
    "N_test": N_test,
    "N": 400,
    "trunk_layers": [2, 400, 400, 400, 400, 400],
    "branch_layers": [mx, 401, 401, 401],
    "x_grid": Nx_grid,
    "my": my,
    "F_tr": F_tr,
    "init_scheme": "Xavier"
}


### define minibatches
onp.random.seed(all_training_properties['t_seed'])
num_complete_batches, leftover = divmod(N_train, t_batch)
num_batches = num_complete_batches + bool(leftover)
def data_stream():
    rng = onp.random.RandomState(0)
    while True:
        perm = rng.permutation(N_train) # compute a random permutation
        for i in range(num_batches):
            batch_idx = perm[i * t_batch:(i + 1) * t_batch]
            yield batch_idx, U_tr[:,batch_idx]
# define the batches generator
batches = data_stream()

# Train
full_info = (np.arange(0,N_train), U_tr)
model      = DeepONet_2ST(all_training_properties)
opt_params = model.train(num_batches, batches, full_info)
A_iter, trunk_params = opt_params
t_loss_vec = model.loss_log


# LSQ predict
T_mat      = model.trunk_eval(trunk_params, Nx_grid)
T1_mat     = onp.concatenate((T_mat, onp.ones((my, 1))), axis=1)
A_st       = scipy.linalg.lstsq(T1_mat, U_tr)[0]
q_st, r_st = scipy.linalg.qr(T1_mat, mode='economic')
A_target   = r_st @ A_st

# Train
Branch_model  = singleBranchTrain(all_training_properties)
opt_b_params  = Branch_model.train(A_target.T)
B_net_twoStep = onp.array(Branch_model.branch_eval(opt_b_params, F_tt))
b_loss_vec    = Branch_model.loss_log
ONet_2ST_pred = q_st @ B_net_twoStep.T

A_tt_lsq      = scipy.linalg.lstsq(T1_mat, U_tt)[0]
ONet_LSQ_pred = T1_mat @ A_tt_lsq

# Compute relative l2 error
for i in range(N_test):
  err_i_LSQ = onp.linalg.norm(ONet_LSQ_pred[:,i][:,None] - U_tt[:,i][:,None])/onp.linalg.norm(U_tt[:,i][:,None])
  err_i_2ST = onp.linalg.norm(ONet_2ST_pred[:,i][:,None] - U_tt[:,i][:,None])/onp.linalg.norm(U_tt[:,i][:,None])
  if i == 0:
    err_LSQ_vec = err_i_LSQ
    err_2ST_vec = err_i_2ST
  else:
    err_LSQ_vec = onp.hstack((err_LSQ_vec, err_i_LSQ))
    err_2ST_vec = onp.hstack((err_2ST_vec, err_i_2ST))

depth_trk = len(all_training_properties["trunk_layers"])-1
depth_bch = len(all_training_properties["branch_layers"])-1
width_trk = all_training_properties["trunk_layers"][1]
width_bch = all_training_properties["branch_layers"][1]
t_epochs  = all_training_properties["t_epochs"]
b_epochs  = all_training_properties["b_epochs"]
N         = all_training_properties["N"]

archi_trk = "Trk[L{:.0f}".format(depth_trk) + "W{:.0f}".format(width_trk) + "]"
archi_bch = "Bch[L{:.0f}".format(depth_bch) + "W{:.0f}".format(width_bch) + "]"
architect = "N{:.0f}".format(N) + archi_trk + archi_bch

scipy.io.savemat(f"{save_path}/ExChooNEW2_2ST_bat_{architect}_{train_num}_DATA.mat", \
                 {'all_training_properties':all_training_properties, \
                  't_train_err':t_loss_vec, 'b_train_err':b_loss_vec, \
                  'test_LSQ_err':err_LSQ_vec, 'test_2ST_err':err_2ST_vec, \
                  'B_net_twoStep':B_net_twoStep, 'B_net_LSQ': A_tt_lsq, \
                  'QTmat1':q_st, 'RTmat1':r_st, 'T1mat1':T1_mat, 'Atarget':A_target})
