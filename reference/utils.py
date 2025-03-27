import logging

import numpy as np
import tensorflow as tf
import yaml
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
        return f"<configer: type - {self.type}, name - {self.name}>"


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch = 0

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                return True
        return False


def tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)


def read_yaml(yaml_file):
    stream = open(yaml_file)
    docs = yaml.load_all(stream, Loader)

    must_have_configs = {"run_parameters", "data", "hyperparameters", "model"}
    configs = dict()
    for doc in docs:
        dtype = doc.get("document_type")
        dname = doc.get("name")
        cfg = configer(**doc)

        configs.update({dname: cfg})
        sname = set([dname])
        l_set = len(sname.union(must_have_configs))
        if l_set == len(must_have_configs):
            must_have_configs = must_have_configs.difference(sname)
    if len(must_have_configs) == 0:
        logger.info(" utils.read_yaml: Config setting succeeded.")
    else:
        logger.warning("utils.read_yaml: Oh no! you dont have the right configs")

    return configs


def get_data(config, data_type: str = "train", horovod_rank: int = 0):
    # the horovod rank tells you which file to get.

    branch = np.load(f"{config.data_dir}/{config.branch_input_files[0]}")
    if not config.duplicate:
        branch = np.load(
            f"{config.data_dir}/{config.branch_input_files[horovod_rank]}",
            allow_pickle=True,
        )
    trunk = np.load(f"{config.data_dir}/{config.trunk_input_files[horovod_rank]}", allow_pickle=True)
    y = np.load(f"{config.data_dir}/{config.don_output_files[horovod_rank]}", allow_pickle=True)

    logger.info(f"Branch shape:\t{branch.shape}")
    logger.info(f"Trunk shape:\t{trunk.shape}")
    logger.info(f"Output shape:\t{y.shape}")

    if y.shape[0] != branch.shape[0]:
        logger.error("First dimension of Y should be the same as first dimension of Branch")
    if y.shape[1] != trunk.shape[0]:
        logger.error("Second dimension of Y should be the same as first dimension of Trunk")

    branch = tensor(branch)
    trunk = tensor(trunk)
    y = tensor(y)

    bidx0 = config.branch_valid_idx_0
    tidx0 = config.trunk_valid_idx_0

    if data_type.lower() == "train":
        return branch[:bidx0], trunk[:tidx0], y[:bidx0, :tidx0]
    if data_type.lower() == "valid":
        return branch[bidx0:], trunk[tidx0:], y[bidx0:, tidx0:]


def shuffle(idx=0, *args):
    x = args[idx]
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    return [tf.gather(unshuffled_x, shuffled_indices) for unshuffled_x in args]


def batcher(branch: tf.Tensor, trunk: tf.Tensor, y: tf.Tensor, batch_branch: int = -1, batch_trunk: int = -1):
    """
    Batches the input tensors 'branch', 'trunk', and 'y' based on the specified batch sizes.

    Args:
        branch (tf.Tensor): Tensor with shape [a, b, ...].
        trunk (tf.Tensor): Tensor with shape [c, d, ...].
        y (tf.Tensor): Tensor with shape [a, c, ...].
        batch_branch (int): Batch size for 'branch'.
        batch_trunk (int): Batch size for 'trunk'.

    Returns:
        Generator: Yields batches of 'branch', 'trunk', and 'y'.
    """
    # If batch sizes are not provided, set them to the sizes of the tensors
    if batch_branch == -1:
        batch_branch = branch.shape[0]
    if batch_trunk == -1:
        batch_trunk = trunk.shape[0]

    num_batches_branch = -(-branch.shape[0] // batch_branch)
    num_batches_trunk = -(-trunk.shape[0] // batch_trunk)

    i = j = 0
    while True:
        start_branch = i * batch_branch
        end_branch = min((i + 1) * batch_branch, branch.shape[0])
        start_trunk = j * batch_trunk
        end_trunk = min((j + 1) * batch_trunk, trunk.shape[0])

        yield (
            branch[start_branch:end_branch],
            trunk[start_trunk:end_trunk],
            y[start_branch:end_branch, start_trunk:end_trunk],
        )

        if i == 0:
            j = (j + 1) % num_batches_trunk
        i = (i + 1) % num_batches_branch


def sample_test(X_func_ls, X_loc_ls, y_ls, batch_size):
    X_func = X_func_ls[18:21, :]
    X_loc = X_loc_ls
    y = y_ls[18:21, :]
    idx = np.arange(y.shape[0])
    # np.random.shuffle(idx)
    X_loc = X_loc
    y = y[idx, :]
    X_func = X_func_ls[idx, :]
    return tensor(X_func), tensor(X_loc[:batch_size]), tensor(y[:batch_size])
