import logging
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import (
    Dropout,
    Reshape,
    Conv2D,
    PReLU,
    Flatten,
    Dense,
    Activation,
    MaxPooling2D,
    BatchNormalization,
)

# from tensorflow.keras.losses import MeanSquaredError
# import horovod.tensorflow as hvd


logging.basicConfig()
logger = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO)


class DeepONet_Model(tf.keras.Model):
    def __init__(self, model, optimizer, branch_sensors, trunk_dim):
        super(DeepONet_Model, self).__init__()
        tf.random.set_seed(model.random_seed)
        self.latent_dim = model.latent_dim
        self.optimizer = optimizer


        self.einsum = model.einsum

        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.fc_loss_list = []
        
        # Build models
        self.branch_net = self.build_net(model.branch, "branch")
        self.trunk_net = self.build_net(model.trunk, "trunk")
        
        self.branch_net.build((None, branch_sensors))
        self.trunk_net.build((None, trunk_dim))
        
        # Compile models
        self.branch_net.compile(optimizer=self.optimizer)
        self.trunk_net.compile(optimizer=self.optimizer)

    def __repr__(self):
        repr_str = 'Branch: \n'
        for layer in self.branch_net.layers:
            if isinstance(layer, Dense):
                repr_str += f"\t dense: {layer.input_shape[1]} x {layer.output_shape[1]}\n"
            else:
                repr_str += f"{layer.get_config()['name']}\n"

        repr_str += 'Trunk: \n'
        for layer in self.trunk_net.layers:
            if isinstance(layer, Dense):
                repr_str += f"\t dense: {layer.input_shape[1]} x {layer.output_shape[1]}\n"
            else:
                repr_str += f"{layer.get_config()['name']}\n"
        repr_str += f'Einsum: {self.einsum}'

        return repr_str
        
            
    def build_net(self, net_config, name):
        struct_file = net_config.get("structure_file")
        if struct_file is not None:
            return self.build_net_with_instructions(struct_file)

        return self.build_net_with_shape(net_config, name)

    def build_net_with_instructions(self, struct_file, name):
        try:
            with open(struct_file, 'r') as file:
                json_config = json.load(file)
            
            model = tf.keras.models.model_from_json(json.dumps(json_config))
            model._name = name

            # Check if the final dense layer matches latent_dim
            if model.layers[-1].units != self.latent_dim:
                model.add(Dense(self.latent_dim))

            return model

        except Exception as e:
            logger.error(f"Error in building network from instructions: {e}")
            raise


    def build_net_with_shape(self, net_config, name):
        width = net_config.get("width")
        depth = net_config.get("depth")
        activation = net_config.get("activation")

        model = Sequential(name=name)

        for i in range(depth):
            model.add(Dense(width, activation=activation))

        model.add(Dense(self.latent_dim))
        return model

    @tf.function(jit_compile=True)
    def call(self, branch_input, trunk_input):
        # X_func -> [number_of_sensor_points, number_of_function_sensor_dimensions]
        # X_loc  -> [number_of_points, number_of_spatial_dimensions]
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        prediction = tf.einsum(self.einsum, branch_output, trunk_output)
        return prediction

    @tf.function(jit_compile=True)
    def loss(self, y_pred, y_train):
        y_pred = tf.reshape(y_pred, (-1, 1))
        y_train = tf.reshape(y_train, (-1, 1))

        # -------------------------------------------------------------#
        # Total Loss
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train)) / tf.reduce_mean(
            tf.square(y_train)
        )
        # -------------------------------------------------------------#
        return [train_loss]
