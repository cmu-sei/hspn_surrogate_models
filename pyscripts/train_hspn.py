import numpy as np
import matplotlib.pyplot as plt
import time
#from sklearn.preprocessing import StandardScaler
import pickle
#import matplotlib
#matplotlib.rc('xtick', labelsize=16)
#matplotlib.rc('ytick', labelsize=16)
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

def main(config_file):
    
    # get configuration
    config = utils.read_yaml(config_file)
    rp = config.get('run_parameters')
    hp = config.get('hyperparameters')
    md = config.get('model')
    dt = config.get('data')
    
    
    #init horovod
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"gpus: {gpus}")

    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
 
    #get data
    train_branch_input, train_trunk_input, train_exp_output = utils.get_data(dt,'train', rank)
    valid_branch_input, valid_trunk_input, valid_exp_output = utils.get_data(dt,'valid', rank)

    #    X_func_ls = aoa ---> train_branch_input
    #    X_loc_ls = xyz  ---> train_trunk_input
    #    y_ls = rho      ---> train_don_output # [13,50M, 1]
    
    # Horovod distributed optimizer
    # synchronizes gradient of the loss function
    opt = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
    opt = hvd.DistributedOptimizer(opt)
    
    don_model = DeepONet_Model(md, 
                               opt, 
                               train_branch_input.shape[1], 
                               train_trunk_input.shape[1])
    
       
    print("DeepONet Training Begins")
    X_func_train, X_loc_train, y_train = utils.batch(train_branch_input,
                                                      train_trunk_input, 
                                                      train_exp_output, 
                                                      hp.batch_size)
    #if rank==0:
    #    print(f"Shape_0: {tf.shape(X_func_train)} and Rank: {rank}")
    #    print(f"Shape_1: {tf.shape(X_loc_train)} and Rank: {rank}")
    #    print(f"Shape_2: {tf.shape(y_train)} and Rank: {rank}")
    # sys.exit()
    
    X_func_test, X_loc_test, y_test = utils.batch(valid_branch_input, 
                                                  valid_trunk_input, 
                                                  valid_exp_output, 
                                                  hp.batch_size)
    train_loss_ls = []
    val_loss_ls = []
            
    begin_time = time.time()
    t1 = time.time()
    for i in range(hp.n_epochs+1):
        loss = train(don_model, train_branch_input, train_trunk_input, train_exp_output, i==0)
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
        

'''
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
'''

if __name__=="__main__":
    if nargs <1 :
        print('Too few arguments!')
    if nargs >1:
        print('Too many arguments!')
        
    print('Running')
    main(sys.argv[0])

