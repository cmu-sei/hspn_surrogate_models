import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.nn import relu
from jax.config import config
import itertools
from functools import partial
from tqdm import trange
import tqdm

# Define the neural net
def MLP(layers, init_scheme="Xavier", activation=np.tanh):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          if init_scheme == "Xavier":
            stddev = 1. / np.sqrt((d_in + d_out) / 2.)
          elif init_scheme == "He":
            stddev     = 1. / np.sqrt( d_in / 2.)
          W = stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
  def apply(params, inputs):
      for W, b in params[:-1]:
          outputs = np.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply


# Define the neural net
def A_MLP(N,N_train):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key):
          k1, k2 = random.split(key)
          #   glorot_stddev = 1. / np.sqrt((N + N_train) / 2.)
          zero_stddev = 0  
          W = zero_stddev * random.normal(k1, (N, N_train))
          return W
      key, *keys = random.split(rng_key, 1)
      params = init_layer(key)
      return params
  def apply(params):
      outputs = params
      return outputs
  return init, apply


# Define the model
class DeepONet_2ST:
    def __init__(self, all_training_properties):
        # Network initialization and evaluation functions
        # trunk_layers, N, N_train, x_grid, seed, lr, d_step, d_rate
        self.trunk_layers = all_training_properties["trunk_layers"]
        self.N            = all_training_properties["N"]
        self.N_train      = all_training_properties["N_train"]
        self.x_grid       = all_training_properties["x_grid"]
        self.t_seed       = all_training_properties["t_seed"]
        self.t_lr         = all_training_properties["t_lr"]
        self.t_d_step     = all_training_properties["t_scheduler_step"]
        self.t_d_rate     = all_training_properties["t_scheduler_rate"]
        self.t_epochs     = all_training_properties["t_epochs"]
        self.my           = all_training_properties["my"]
        self.batch_size   = all_training_properties["t_batch"]

        self.A_init, self.A_apply = A_MLP(self.N + 1, self.N_train)
        self.trunk_init, self.trunk_apply = MLP(self.trunk_layers, activation=np.tanh)

        # Initialize
        rng_key      = random.PRNGKey(self.t_seed)
        A_params     = self.A_init(rng_key)

        rng_key, *subkeys = random.split(rng_key)
        trunk_params = self.trunk_init(rng_key)
        params       = (A_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(self.t_lr,
                                                                       decay_steps=self.t_d_step,
                                                                       decay_rate=self.t_d_rate))
        self.opt_state = self.opt_init(params)

        self.itercount = itertools.count()
        self.loss_log = []


    # Define opeartor net
    def operator_net(self, params, batch_index):
        A_params, trunk_params = params
        B = self.A_apply(A_params)  # N  x N_train
        B = B[:,batch_index]
        T = vmap(self.trunk_apply, (None, 0))(trunk_params, self.x_grid)
        outputs = T @ B[:-1, :] + np.ones((self.my, 1)) @ B[-1, :][None, :]  # my x N_train
        return outputs

    # Define loss
    def loss(self, params, batches):
        batch_index, Udata = batches
        pred = self.operator_net(params, batch_index)
        # Compute loss
        loss = np.mean((Udata.flatten() - pred.flatten()) ** 2)
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batches):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batches)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, num_batches, batches, full_info):
        pbar = trange(self.t_epochs)
        min_loss = 1e+8
        # Main training loop
        for it in pbar:
            # batch = next(data)
            for _ in range(num_batches):
                self.opt_state = self.step(next(self.itercount), self.opt_state, next(batches))

            if (it % 10 == 0) or (it == self.t_epochs-1):
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.loss(params, full_info)
                if min_loss > loss_value:
                    min_loss = loss_value
                    opt_params = params

                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'it': '{0:d}'.format(it), 'Loss': '{0:1.4e}'.format(loss_value), 'minLoss': '{0:1.4e}'.format(min_loss)})

        return opt_params

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def trunk_eval(self, trunk_params, Y_star):
        T = vmap(self.trunk_apply, (None, 0))(trunk_params, Y_star)
        return T
    


# Define the model
class singleBranchTrain:
    def __init__(self, all_training_properties):
        # Network initialization and evaluation functions
        # branch_layers, F_tr, seed, lr, d_step, d_rate
        self.branch_layers = all_training_properties["branch_layers"]
        self.F_tr         = all_training_properties["F_tr"]
        self.b_seed       = all_training_properties["b_seed"]
        self.b_lr         = all_training_properties["b_lr"]
        self.b_d_step     = all_training_properties["b_scheduler_step"]
        self.b_d_rate     = all_training_properties["b_scheduler_rate"]
        self.b_epochs     = all_training_properties["b_epochs"]

        self.branch_init, self.branch_apply = MLP(self.branch_layers, activation=np.tanh)

        # Initialize
        params = self.branch_init(rng_key=random.PRNGKey(self.b_seed))

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(self.b_lr,
                                                                       decay_steps=self.b_d_step,
                                                                       decay_rate=self.b_d_rate))
        self.opt_state = self.opt_init(params)

        self.itercount = itertools.count()
        # Logger
        self.loss_log = []

    # Define loss
    def loss(self, params, singleA):
        # pred = self.branch_apply(params, F_tr.T) # 1 x 900
        pred = vmap(self.branch_apply, (None, 0))(params, self.F_tr)
        # Compute loss
        loss = np.mean((singleA.flatten() - pred.flatten()) ** 2)
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, singleA):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, singleA)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, singleA):
        pbar = trange(self.b_epochs)
        min_loss = 1e+8
        # Main training loop
        for it in pbar:
            # batch = next(data)
            self.opt_state = self.step(next(self.itercount), self.opt_state, singleA)

            if it % 10 == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.loss(params, singleA)
                if min_loss > loss_value:
                    min_loss = loss_value
                    opt_params = params

                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss': '{0:1.4e}'.format(loss_value), 'minLoss': '{0:1.4e}'.format(min_loss)})

        return opt_params

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def branch_eval(self, params, F_star):
        B = vmap(self.branch_apply, (None, 0))(params, F_star)
        return B


# Define the model
class DeepONet_ALL:
    def __init__(self, all_training_properties):
        # Network initialization and evaluation functions
        self.trunk_layers = all_training_properties["trunk_layers"]
        self.x_grid       = all_training_properties["x_grid"]
        self.t_seed       = all_training_properties["t_seed"]
        self.b_seed       = all_training_properties["b_seed"]
        self.lr           = all_training_properties["lr"]
        self.d_step       = all_training_properties["scheduler_step"]
        self.d_rate       = all_training_properties["scheduler_rate"]
        self.epochs       = all_training_properties["epochs"]
        self.branch_layers = all_training_properties["branch_layers"]
        self.F_tr          = all_training_properties["F_tr"]
        self.my            = all_training_properties["my"]

        self.branch_init, self.branch_apply = MLP(self.branch_layers, activation=np.tanh)
        self.trunk_init,  self.trunk_apply  = MLP(self.trunk_layers, activation=np.tanh)

        # Initialize
        trunk_params  = self.trunk_init(rng_key=random.PRNGKey(self.t_seed))
        branch_params = self.branch_init(rng_key=random.PRNGKey(self.b_seed))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(self.lr,
                                                                       decay_steps=self.d_step,
                                                                       decay_rate=self.d_rate))
        self.opt_state = self.opt_init(params)

        self.itercount = itertools.count()
        self.loss_log = []


    # Define opeartor net
    def operator_net(self, params):
        branch_params, trunk_params = params
        B = vmap(self.branch_apply, (None, 0))(branch_params, self.F_tr).T
        T = vmap(self.trunk_apply, (None, 0))(trunk_params, self.x_grid)
        outputs = T @ B[:-1, :] + np.ones((self.my, 1)) @ B[-1, :][None, :]  # my x N_train
        return outputs

    # Define loss
    def loss(self, params, Udata):
        pred = self.operator_net(params)
        # Compute loss
        loss = np.mean((Udata.flatten() - pred.flatten()) ** 2)
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, Udata):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, Udata)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, Udata):
        pbar = trange(self.epochs)
        min_loss = 1e+8
        # Main training loop
        for it in pbar:
            # batch = next(data)
            self.opt_state = self.step(next(self.itercount), self.opt_state, Udata)

            if it % 10 == 0:
                params = self.get_params(self.opt_state)

                # Compute loss
                loss_value = self.loss(params, Udata)
                if min_loss > loss_value:
                    min_loss = loss_value
                    opt_params = params

                # Store loss
                self.loss_log.append(loss_value)

                # Print loss during training
                pbar.set_postfix({'Loss': '{0:1.4e}'.format(loss_value), 'minLoss': '{0:1.4e}'.format(min_loss)})

        return opt_params

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def trunk_eval(self, trunk_params, Y_star):
        T = vmap(self.trunk_apply, (None, 0))(trunk_params, Y_star)
        return T
    
    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def branch_eval(self, branch_params, F_star):
        T = vmap(self.branch_apply, (None, 0))(branch_params, F_star)
        return T