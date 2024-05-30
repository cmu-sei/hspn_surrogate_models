import logging 
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dropout, Reshape, Conv2D, PReLU, Flatten, Dense, Activation, MaxPooling2D, BatchNormalization
#from tensorflow.keras.losses import MeanSquaredError
#import horovod.tensorflow as hvd


logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO)
class DeepONet_Model(tf.keras.Model):
    def __init__(self,model, optimizer, branch_sensors, trunk_dim):
        super(DeepONet_Model, self).__init__()
        #np.random.seed(model.random_seed)
        tf.random.set_seed(model.random_seed)
        self.latent_dim = model.latent_dim
        self.optimizer = optimizer
        
        self.branch_net = self.build_net(model.branch,"branch")
        self.branch_net.build((None, branch_sensors))
        
        self.trunk_net  = self.build_net(model.trunk,"trunk")
        self.trunk_net.build((None, trunk_dim)) 
        
        self.einsum = model.einsum
        
        
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.fc_loss_list = []

    def build_net(self, net_config, name):
        struct_file = net_config.get('structure_file')
        if struct_file is not None:
            return self.build_net_with_instructions(struct_file)
        
        return self.build_net_with_shape(net_config, name)
    
    def build_net_with_instructions(self, struct_file, name):
        #not implemented
        logger.error('build_net_with_instructions not implemented')    
        pass
    
    def build_net_with_shape(self, net_config, name):
        width = net_config.get('width') 
        depth = net_config.get('depth')
        activation = net_config.get('activation') 
        
        model = Sequential(name=name)
        
        for i in range(depth):
            model.add(Dense(width, activation=activation))

        model.add(Dense(self.latent_dim))
        return model


    @tf.function(jit_compile=True)
    def call(self, branch_input, trunk_input):
        #X_func -> [number_of_sensor_points, number_of_function_sensor_dimensions]
        #X_loc  -> [number_of_points, number_of_spatial_dimensions]
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        prediction = tf.einsum(self.einsum,branch_output, trunk_output)
        return(prediction)

    @tf.function(jit_compile=True)
    def loss(self, y_pred, y_train):
        y_pred = tf.reshape(y_pred, (-1, 1))
        y_train = tf.reshape(y_train,(-1, 1))
    
        #-------------------------------------------------------------#
        #Total Loss
        train_loss =  tf.reduce_mean(tf.square(y_pred - y_train ))/tf.reduce_mean(tf.square(y_train )) 
        #-------------------------------------------------------------#
        return([train_loss])
