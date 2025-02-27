"""DeepONet Model Implementation."""

import json
import logging
from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

logger = logging.getLogger(__name__)


class DeepONetModel(tf.keras.Model):
    """DON implementation.

    Attributes:
        latent_dim: Intenal latent space dim
        einsum: Einsum pattern for combining branch and trunk outputs
        branch_net: Network for processing input functions
        trunk_net: Network for processing spatial points
        history: Dict with training history
    """

    def __init__(self, config: Dict[str, Any], branch_dim: int, trunk_dim: int) -> None:
        """Sets up model building subnets.

        Args:
            config: Configuration dictionary with model parameters
            branch_dim: Dimension of branch network input
            trunk_dim: Dimension of trunk network input
        """
        super(DeepONetModel, self).__init__()

        self.latent_dim = config.get("latent_dim", 20)
        self.einsum = config.get("einsum", "ij,ik->ik")
        self.random_seed = config.get("random_seed", 42)

        tf.random.set_seed(self.random_seed)

        self.history = {"epochs": [], "train_loss": [], "val_loss": []}

        self.branch_net = self._build_net(config.get("branch", {}), "branch", branch_dim)
        self.trunk_net = self._build_net(config.get("trunk", {}), "trunk", trunk_dim)

        logger.info(f"DeepONet model created with latent_dim={self.latent_dim}")
        logger.debug(f"Model structure:\n{self}")

    def _build_net(self, net_config: Dict[str, Any], name: str, input_dim: int) -> Model:
        """Build a network from config.

        Supports two methods of network creation:
        1. From a structure file (JSON)
        2. From shape parameters (width, depth, activation)

        Args:
            net_config: Configuration for the network
            name: Name of the network
            input_dim: Dimension of the input

        Returns:
            Model: Built Keras model

        Raises:
            ValueError: If the network configuration is invalid
        """
        structure_file = net_config.get("structure_file")

        if structure_file:
            return self._build_from_file(structure_file, name, input_dim)
        else:
            return self._build_from_params(net_config, name, input_dim)

    def _build_from_file(self, structure_file: str, name: str, input_dim: int) -> Model:
        """Build network from tf JSON.

        Args:
            structure_file: Path to JSON file with network structure
            name: Name of the network
            input_dim: Dimension of the input

        Returns:
            Model: Built Keras model

        Raises:
            FileNotFoundError: If the structure file does not exist
            json.JSONDecodeError: If the structure file is not valid JSON
        """
        try:
            with open(structure_file, "r") as f:
                model_config = json.load(f)

            model = tf.keras.models.model_from_json(json.dumps(model_config))
            assert model is not None, "model_from_json returned nothing"
            model._name = name

            if model.layers[0].input_shape[1] != input_dim:
                logger.warning(
                    f"Input dimension mismatch in {name} network. "
                    f"Expected {input_dim}, got {model.layers[0].input_shape[1]}."
                )

                # Build new model w correct input shape
                inputs = Input(shape=(input_dim,), name=f"{name}_input")
                x = inputs

                # Skip first/input layer and rebuild
                for layer in model.layers[1:]:
                    x = layer(x)

                model = Model(inputs=inputs, outputs=x, type=tf.float32)

            if model.layers[-1].output_shape[1] != self.latent_dim:
                logger.info(
                    f"Adding output layer to match latent dimension "
                    f"({model.layers[-1].output_shape[1]} -> {self.latent_dim})"
                )

                outputs = Dense(self.latent_dim, name=f"{name}_output")(model.layers[-1].output)

                model = Model(inputs=model.input, outputs=outputs, name=name)

            return model

        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading structure file {structure_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error building network from file: {e}")
            raise ValueError(f"Failed to build network from file: {e}")

    def _build_from_params(self, net_config: Dict[str, Any], name: str, input_dim: int) -> Model:
        """Build network from structure parameters.

        Args:
            net_config: Config dict with width, depth, and activation
            name: Name of the network
            input_dim: Dimension of the input

        Returns:
            Model: Built Keras model
        """
        width = net_config.get("width")
        depth = net_config.get("depth")
        activation = net_config.get("activation")
        assert width is not None
        assert depth is not None
        assert activation is not None

        inputs = Input(shape=(input_dim,), name=f"{name}_input")
        x = inputs  # current layer
        for _ in range(depth):  # hidden layers
            x = Dense(width, activation=activation)(x)
        outputs = Dense(self.latent_dim)(x)  # Output layer
        model = Model(inputs=inputs, outputs=outputs, name=name)
        logger.info(f"Built {name} network with {depth} layers of width {width}")
        return model

    @tf.function
    def call(self, inputs, training=None):
        """Forward pass.

        Handles different batch structures between branch and trunk inputs.
        - branch_input: [batch_size, branch_dim] - Batched function representations
        - trunk_input: [trunk_points, trunk_dim] - All spatial points (not batched)

        Args:
            inputs: Tuple of (branch_input, trunk_input) tensors
            training: Whether the call is in training mode

        Returns:
            tf.Tensor: Model output with shape [batch_size, trunk_points]
        """
        branch_input, trunk_input = inputs

        branch_output = self.branch_net(branch_input, training=training)  # [batch_size, latent_dim]
        trunk_output = self.trunk_net(trunk_input, training=training)  # [trunk_points, latent_dim]
        preds = tf.einsum(self.einsum, branch_output, trunk_output)  # [batch_size, trunk_points]
        return preds

    def compute_loss(self, y_pred, y_true):
        """Calculate loss.

        Args:
            y_pred: Predicted values tensor [batch_size, trunk_points]
            y_true: Target values tensor [batch_size, trunk_points]

        Returns:
            tf.Tensor: Loss value
        """
        # normalized mean squared error
        epsilon = 1e-8
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        norm = tf.reduce_mean(tf.square(y_true)) + epsilon

        return mse / norm

    def save_history(self, filepath: str) -> None:
        """Save training history to a JSON file.

        Args:
            filepath: Path to save the history file
        """
        try:
            with open(filepath, "w") as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Model history saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model history: {e}")

    def __repr__(self) -> str:
        """Get string representation of the model architecture.

        Returns:
            str: Formatted string showing model structure
        """
        repr_str = "DeepONet Model\n"
        repr_str += f"Latent dimension: {self.latent_dim}\n"
        repr_str += f"Einsum pattern: {self.einsum}\n\n"

        repr_str += "Branch Network:\n"
        self.branch_net.summary(print_fn=lambda s: repr_str + s + "\n")

        repr_str += "\nTrunk Network:\n"
        self.trunk_net.summary(print_fn=lambda s: repr_str + s + "\n")

        return repr_str
