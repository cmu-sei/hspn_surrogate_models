import logging
import numpy as np
import yaml
import tensorflow as tf
from yaml import Loader

logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO)

class configer:
    def __init__(self, document_type, name, **kwargs):
        self.type = document_type
        self.name = name
        self.__dict__.update(**kwargs)
    def __repr__(self):
        return f'<configer: type - {self.type}, name - {self.name}>'
                

def tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)


def read_yaml(yaml_file):
    stream = open(yaml_file)
    docs = yaml.load_all(stream, Loader)
    
    must_have_configs = {'run_parameters','data','hyperparameters','model'}
    configs = dict()
    for doc in docs:
        
        dtype = doc.get('document_type')
        dname = doc.get('name')
        cfg = configer(**doc)
        
        configs.update({dname: cfg})
        sname = set([dname])
        l_set = len(sname.union(must_have_configs))
        if l_set == len(must_have_configs):
            must_have_configs = must_have_configs.difference(sname)
    if len(must_have_configs) ==0:
        logger.info(" utils.read_yaml: Config setting succeeded.")
    else:
        logger.warning('utils.read_yaml: Oh no! you dont have the right configs')
        
    return configs


def get_data(config, data_type, horovod_rank):
    #the horovod rank tells you which file to get.
    
    branch = np.load(f'{config.data_dir}/{config.branch[0]}')
    if not config.duplicate:
        branch = np.load(f'{config.data_dir}/{config.branch_input_files[horovod_rank]}',allow_pickle=True)
    trunk = np.load(f'{config.data_dir}/{config.trunk_input_files[horovod_rank]}', allow_pickle=True)
    y = np.load(f'{config.data_dir}/{config.don_output_files[horovod_rank]}', allow_pickle=True)
    
    branch = tensor(branch)
    trunk = tensor(trunk)
    y = tensor(y)
    bidx0 = config.branch_valid_idx_0
    tidx0 = config.trunk_valid_idx_0
    
    if data_type == "train":
        return branch[:bidx0,:], trunk[:bidx0, :tidx0], y[:bidx0, :tidx0,:]
    if data_type == "valid":
        return branch[bidx0:,:], trunk[bidx0:, tidx0:], y[bidx0:, tidx0:,:] 

    

def sample_train(X_func_ls, X_loc_ls, y_ls, batch_size):
     X_func = X_func_ls[0:18, :] 
     X_loc  = X_loc_ls
     y = y_ls[0:18, :] 
     idx = np.arange(y.shape[0])
     #np.random.shuffle(idx)
     X_loc = X_loc
     y = y[idx, :]
     X_func = X_func_ls[idx, :]
     return tensor(X_func), tensor(X_loc[:batch_size]), tensor(y[:batch_size])


def sample_test(X_func_ls, X_loc_ls, y_ls, batch_size):
     X_func = X_func_ls[18:21, :] 
     X_loc  = X_loc_ls
     y = y_ls[18:21, :] 
     idx = np.arange(y.shape[0])
     #np.random.shuffle(idx)
     X_loc = X_loc
     y = y[idx, :]
     X_func = X_func_ls[idx, :]
     return tensor(X_func), tensor(X_loc[:batch_size]), tensor(y[:batch_size])

