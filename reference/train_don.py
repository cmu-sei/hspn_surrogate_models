import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
import sys
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"] ="2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dropout, Reshape, Conv2D, PReLU, Flatten, Dense, Activation, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError
import horovod.tensorflow as hvd


NUM_GPUs = 4
INPUT_FOLDER = "/p/work1/projects/FrontierX/don_volume_data" 
OUTPUT_FOLDER = "/p/work1/projects/FrontierX/hspn/data/don_volume_data"
MODEL_FOLDER = "/p/work1/projects/FrontierX/hspn/models/don_volume_model" 

class DeepONet_Model(tf.keras.Model):
    def __init__(self, Par, optimizer):
        super(DeepONet_Model, self).__init__()
        np.random.seed(1234)
        tf.random.set_seed(1234)
        self.latent_dim = Par['latent_dim']
        self.Par = Par
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.fc_loss_list = []
        self.optimizer = optimizer
        self.branch_net = self.build_branch_net()
        self.branch_net.build((None, Par['m'] ))

        self.trunk_net  = self.build_trunk_net()
        self.trunk_net.build((None, 3)) 

        print(self.branch_net.summary())
        print(self.trunk_net.summary())

    def build_branch_net(self):
        model = Sequential(name='Branch_Network')
        model.add(Dense(100, activation='elu'))
        model.add(Dense(100, activation='elu')) 
        model.add(Dense(100, activation='elu'))
        model.add(Dense(100, activation='elu')) 
        model.add(Dense(self.latent_dim))
        return model

    def build_trunk_net(self):
        model = Sequential(name='Trunk_Network')
        model.add( Dense(100, activation='relu'))
        model.add( Dense(100, activation='relu'))
        model.add( Dense(100, activation='relu'))   
        model.add( Dense(100, activation='relu'))
        model.add( Dense(100, activation='relu'))    
        model.add(Dense(self.latent_dim))
        return model


    @tf.function(jit_compile=True)
    def call(self, X_func, X_loc):
        #X_func -> [BS, 1]
        #X_loc  -> [n_t, 3]
        y_func = X_func
        y_func = self.branch_net(y_func)
        y_loc = X_loc
        y_loc = self.trunk_net(y_loc)
        Y = tf.einsum("ij, kj-> ik", y_func, y_loc)
        return(Y)

    @tf.function(jit_compile=True)
    def loss(self, y_pred, y_train):
        y_pred = tf.reshape(y_pred, (-1, 1))
        y_train = tf.reshape(y_train,(-1, 1))
        

        #-------------------------------------------------------------#
        #Total Loss
        train_loss =  tf.reduce_mean(tf.square(y_pred - y_train ))/tf.reduce_mean(tf.square(y_train )) 
        #-------------------------------------------------------------#
        return([train_loss])




def tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)

@tf.function()
def train(don_model, X_func, X_loc, y, first_batch):
    with tf.GradientTape() as tape:
        y_hat  = don_model(X_func, X_loc)
        loss   = don_model.loss(y_hat, y)[0]

    tape = hvd.DistributedGradientTape(tape)    
    gradients = tape.gradient(loss, don_model.trainable_variables)
    don_model.optimizer.apply_gradients(zip(gradients, don_model.trainable_variables))
    if first_batch:
        hvd.broadcast_variables(don_model.variables, root_rank=0)
        hvd.broadcast_variables(don_model.optimizer.variables(), root_rank=0)
    return(loss)


def sample_train(X_func_ls, X_loc_ls, y_ls, batch_size):
     X_func = X_func_ls[0:18, :] 
     X_loc  = X_loc_ls
     y = y_ls[0:18, :] 
     idx = np.arange(y.shape[0])
     #np.random.shuffle(idx)
     X_loc = X_loc
     y = y[idx, :]
     X_func = X_func_ls[idx, :]
     return tensor(X_func), tensor(X_loc[:batch_size]), tensor(y[:,:batch_size])


def sample_test(X_func_ls, X_loc_ls, y_ls, batch_size):
     X_func = X_func_ls[18:21, :] 
     X_loc  = X_loc_ls
     y = y_ls[18:21, :] 
     idx = np.arange(y.shape[0])
     #np.random.shuffle(idx)
     X_loc = X_loc
     y = y[idx, :]
     X_func = X_func_ls[idx, :]
     return tensor(X_func), tensor(X_loc[:batch_size]), tensor(y[:,:batch_size])




def main():
    #init horovod
    hvd.init()
    print("Ammended!")
    rank = hvd.rank()
    size = hvd.size()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"gpus: {gpus}")

    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    Par = {}
    aoa = np.load(f'{INPUT_FOLDER}/aoa_total.npy', allow_pickle=True)
    aoa = aoa.reshape(-1, 1)
    if rank==0:
        print(f"AoA is : {aoa.shape}")
    
    # split the data
    f_name = f"{OUTPUT_FOLDER}/trunk_in_" + str(rank) + ".npy"
    out_name = f"{OUTPUT_FOLDER}/y_out_" + str(rank) + ".npy"
    
    xyz = np.load(f_name, allow_pickle=True)
    rho = np.load(out_name , allow_pickle=True)


    X_func_ls = aoa  #[13,1]
    X_loc_ls = xyz   #[50M, 3]


    if hvd.rank==0:
        print(f"branch shape: {X_func_ls.shape} and Trunk Shape: {X_loc_ls.shape}")
        print(f"mn: {np.min(X_loc_ls, axis=0)}, mx: {np.max(X_loc_ls, axis=0)}")
    y_ls = rho       #[13,50M, 1]
    
    
    if hvd.rank==0:
        print('X_func: ', X_func_ls.shape, ', min: ', np.min(X_func_ls), ', max: ', np.max(X_func_ls) )
        print('X_loc: ', X_loc_ls.shape)

    X_loc_ls = X_loc_ls
    
    Par['address'] =f"{MODEL_FOLDER}/don" 
    Par['latent_dim'] = 25
    Par['m'] = X_func_ls.shape[1]

    print(Par['address'])
    print('------\n')
    lr = 5e-04

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # Horovod distributed optimizer
    # synchronizes gradient of the loss function
    opt = hvd.DistributedOptimizer(opt)
    
    don_model = DeepONet_Model(Par, opt)
    #don_model.load_weights(Par["address"] + "/Run_2" + "/model_"+str(50000))
    n_epochs =   50001
    batch_size = 12690717 
       
    print("DeepONet Training Begins")
    X_func_train, X_loc_train, y_train = sample_train(X_func_ls, X_loc_ls, y_ls, batch_size)
    if rank==0:
        print(f"Shape_0: {tf.shape(X_func_train)} and Rank: {rank}")
        print(f"Shape_1: {tf.shape(X_loc_train)} and Rank: {rank}")
        print(f"Shape_2: {tf.shape(y_train)} and Rank: {rank}")
    # sys.exit()
    X_func_test, X_loc_test, y_test = sample_test(X_func_ls, X_loc_ls, y_ls, batch_size)
    train_loss_ls = []
    val_loss_ls = []
            
    begin_time = time.time()
    t1 = time.time()
    for i in range(n_epochs+1):
        loss = train(don_model, tensor(X_func_train), tensor(X_loc_train), tensor(y_train), i==0)
        avg_loss = hvd.allreduce(loss, average=True)
       
        if (i%500 == 0) and (rank==0):
            print("saving model")       
            don_model.save_weights(Par['address'] + "/model_"+str(i))
        # if rank == 0:
        #     print("epoch:" + str(i) + ", Avg Loss:" + "{:.3e}".format(avg_loss) +  ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )

        
        if (i%500 == 0):    
            y_pred = don_model(X_func_train, X_loc_train)
            local_loss = don_model.loss(y_pred, y_train)[0]
            avg_loss = hvd.allreduce(local_loss, average=True)
            train_loss = avg_loss.numpy()
            train_loss_ls.append(train_loss)
            y_pred = don_model(X_func_test, X_loc_test)
            local_val_loss = don_model.loss(y_pred, y_test)[0]
            val_loss = hvd.allreduce(local_val_loss, average=True)
            val_loss_ls.append(val_loss.numpy())

            if rank==0:
                print("Rank:" + str(rank), "epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + ", Val Loss:" + "{:.3e}".format(val_loss) + ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )

            don_model.index_list.append(i)
            don_model.train_loss_list.append(train_loss)
            don_model.val_loss_list.append(val_loss)
        


    #Convergence plot
    if rank==0:
        index_list = don_model.index_list
        train_loss_list = don_model.train_loss_list
        val_loss_list = don_model.val_loss_list
        np.savez(Par['address']+'/convergence_data', index_list=index_list, train_loss_list=train_loss_list, val_loss_list=val_loss_list)

            
    #     plt.close()
        fig = plt.figure(figsize=(10,7))
        plt.plot(index_list, train_loss_list, label="train", linewidth=2)
        plt.plot(index_list, val_loss_list, label="val", linewidth=2)
        plt.legend(fontsize=16)
        plt.yscale('log')
        plt.xlabel("Epoch", fontsize=18)
        plt.ylabel("MSE", fontsize=18)
        plt.savefig(Par["address"] + "/convergence.png", dpi=800)
        plt.close()
        
        val_loss_list[0]=1000.0
        don_model_number = index_list[np.argmin(val_loss_list)]
        print('Best DeepONet model: ', don_model_number)
        print('--------Complete--------')


main()

