import argparse
import logging
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] ="2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dropout, Reshape, Conv2D, PReLU, Flatten, Dense, Activation, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.mixed_precision import experimental as mixed_precision, LossScaleOptimizer

import horovod.tensorflow as hvd

from hspn import utils as utils
from hspn.don import DeepONet_Model

# for logging
logging.basicConfig()
logger = logging.getLogger("training script")
formatter = logging.Formatter('%(name)s: %(asctime)s: %(levelname)s: %(message)s')
shandler = logging.StreamHandler()
shandler.setFormatter(formatter)
t = time.gmtime()
datetime = f'{t.tm_year}-{t.tm_yday}-{t.tm_hour}-{t.tm_min}-{t.tm_sec}'
fhandler = logging.FileHandler(f"logfiles/train_hspn-{datetime}.log")
fhandler.setFormatter(formatter)

logger.addHandler(shandler)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG) 


def setup_gpus():
    hvd.init()    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.debug(f"GPUs setup: {gpus}")
        except RuntimeError as e:
            logger.error(f"Error setting up GPUs: {e}")
        except IndexError as e:
            logger.error(f"Error: {e}")
            logger.error(f"hvd.local_rank() = {hvd.local_rank()}, but gpus list has only {len(gpus)} elements.")

    else:
        logger.warning("No GPUs found.")


def setup_mixed_precision(hp):
    
    base_optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    optimizer = LossScaleOptimizer(base_optimizer)
    optimizer = hvd.DistributedOptimizer(optimizer)
    return optimizer

@tf.function()
def train(model, x_branch, x_trunk, y_true, first_batch):
    
    with tf.GradientTape() as tape:
        
        y_pred = model(x_branch, x_trunk)
        y_true = tf.cast(y_true, dtype=tf.float16)
        y_pred = tf.cast(y_pred, dtype=tf.float16)  
        local_loss = model.loss(y_pred, y_true)[0]

    tape = hvd.DistributedGradientTape(tape)    
    gradients = tape.gradient(local_loss, model.trainable_variables)
    
    # ignore gradient accumulation for now.
    #if accumulate_count ==0 :
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if first_batch:
        # Check variable consistency
        for var in model.variables:
            print(f"Variable name: {var.name}, Shape: {var.shape}, Dtype: {var.dtype}")
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(model.optimizer.variables(), root_rank=0)
    
    return(local_loss)
    
   # Accumulate gradients across iterations
     # TBD
    '''
    if batch_no % accumulate_count == 0 :
        accumulated_gradients = [tf.zeros_like(grad) for grad in gradients] 
    for i, grad in enumerate(gradients):
        accumulated_gradients[i] += grad
    
    # Apply accumulated gradients every `accumulate_count` iterations
    if (batch_no + 1) % accumulate_count == 0:
        averaged_gradients = [gd/accumulate_count for gd in accumulated_gradients]
        model.optimizer.apply_gradients(zip(averaged_gradients, model.trainable_variables))
        for i, _ in enumerate(accumulated_gradients):
            accumulated_gradients[i] = tf.zeros_like(gradients[i])
    return loss
    '''



@tf.function()
def get_loss(model, x_branch, x_trunk, y_true):
    y_pred = model(x_branch, x_trunk)
    y_true = tf.cast(y_true, dtype=tf.float16)
    y_pred = tf.cast(y_pred, dtype=tf.float16)  
    local_loss = model.loss(y_pred, y_true)[0]
    avg_loss = hvd.allreduce(local_loss, average=True)
    return avg_loss
    

def main(rp, hp, md, dt):
    setup_gpus()
    
    ### Distributed optimizer
    opt = setup_mixed_precision(hp)
    
    ### Data loading
    train_bin, train_tin, train_y_val = utils.get_data(dt, 'train', hvd.rank())
    valid_bin, valid_tin, valid_y_val = utils.get_data(dt, 'valid', hvd.rank())


    ### Data batching
    train_batcher = utils.batcher(train_bin, train_tin, train_y_val, hp.batch_size, hp.batch_size)
    


    ### Model creation
    don_model = DeepONet_Model(md, opt, train_bin.shape[1], train_tin.shape[1])
    logger.info('DeepONet model created')


    early_stopping = utils.EarlyStopping(patience=hp.patience, min_delta=hp.min_delta)
    begin_time = time.time()
    stop_training = False
    accumulated_gradients = []
    for i in range(hp.n_epochs+1):
        #train model on next batch
        x_branch, x_trunk, y_true = next(train_batcher)
        
        local_loss = train(don_model, 
                     x_branch, 
                     x_trunk, 
                     y_true, 
                     i==0)
        loss = hvd.allreduce(local_loss, average=True)
    
        if i%hp.interval == 0:
            train_loss = get_loss(don_model, train_bin, train_tin, train_y_val).numpy()
            valid_loss = get_loss(don_model, valid_bin, valid_tin, valid_y_val).numpy()
            
            if hvd.rank() == 0:
                don_model.index_list.append(i)
                don_model.train_loss_list.append(train_loss)
                don_model.val_loss_list.append(valid_loss)
                            
                name = f"{rp.model_dir}/{rp.prepend}_model_{i:06}_vloss-{valid_loss:0.3f}"
                logging.info("saving model")       
                don_model.save_weights(name)
                logger.info(f"Rank:{hvd.rank()} epoch:{i}  Train Loss:{train_loss:.3e}, Val Loss:{valid_loss:.3e}, elapsed time:{int(time.time()-begin_time)} s" )

                if early_stopping(valid_loss):
                    logger.info(f"Early stopping at epoch {i}")
                    stop_training = True

            # Broadcast the early stopping signal to all ranks
            stop_training = hvd.broadcast(stop_training, root_rank=0)

        if stop_training:
            break


def args_to_configs(args):
    if args.acfg and not (args.rcfg or args.hcfg or args.mcfg or args.dcfg):
        logger.debug('Single file')
        config = utils.read_yaml(args.acfg)
        rp = config.get('run_parameters')
        hp = config.get('hyperparameters')
        md = config.get('model')
        dt = config.get('data')
    elif not args.acfg and (args.rcfg and args.hcfg and args.mcfg and args.dcfg):
        logger.debug('Multiple files')
        rp = utils.read_yaml(args.rcfg)
        hp = utils.read_yaml(args.hcfg)
        md = utils.read_yaml(args.mcfg)
        dt = utils.read_yaml(args.dcfg)
    else:
        logger.error("Incompatible arguments")
        sys.exit()
    return rp, hp, md, dt
            
if __name__=="__main__":

    
    #Define input parameters
    usage='%(prog)s <functional argument> <ouput target argument>'
    description='DeepONet training tool'
    parser = argparse.ArgumentParser(usage=usage,description=description)

    arg_names = {'all':'a','run':'r','hyperparameter':'y','model':'m','data':'d'}
    
    for full, let in arg_names.items():
        parser.add_argument(f'-{let}',
                            f'--{full[:3]}',
                            dest=f'{full[0]}cfg',
                            help=f'{full} paramter config file',
                            metavar="<{full} config yaml>",
                            required=False)
   
    args = parser.parse_args()
    # read in yaml files
    rp, hp, md, dt = args_to_configs(args)
    
    main(rp, hp, md, dt)


