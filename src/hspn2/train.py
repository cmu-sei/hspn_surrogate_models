"""DeepONet Training.

Will support:
- Single-GPU and distributed training (Horovod)
- Mixed precision
- Early stopping checkpointing
- Training and validation metric tracking
"""

import os
import time
import logging
from typing import Dict, List, Optional

import tensorflow as tf
import numpy as np
from omegaconf import DictConfig

try:
    import horovod.tensorflow as hvd

    HOROVOD_AVAILABLE = True
except ImportError:
    HOROVOD_AVAILABLE = False

from model import DeepONetModel
from dataloader import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for DeepONet models.

    Handles the training process including:
    - Model compilation
    - Mixed precision configuration
    - Single-GPU or distributed training
    - Early stopping
    - Model checkpointing

    Attributes:
        config: Training configuration
        model: DeepONet model to train
        optimizer: TensorFlow optimizer
        use_mixed_precision: Whether to use mixed precision training
        distributed: Whether to use distributed training
        rank: Process rank in distributed training
        data_loader: DataLoader for training and validation data
    """

    def __init__(self, config: DictConfig, model: Optional[DeepONetModel] = None) -> None:
        """Initialize the trainer.

        Args:
            config: Training configuration
            model: Optional pre-initialized DeepONet model otherwise built from config
        """
        self.config = config
        self.optimizer = None
        self.use_mixed_precision = config.training.get("mixed_precision", False)
        self.distributed = config.training.get("distributed", False) and HOROVOD_AVAILABLE
        self.rank = 0

        self._setup_environment()
        self._setup_hardware()
        self.data_loader = DataLoader(config.data)

        if model is None:  # build from config
            branch_shape, trunk_shape, _ = self.data_loader.get_shapes()
            self.model = DeepONetModel(config=self.config.model, branch_dim=branch_shape[1], trunk_dim=trunk_shape[1])
            logger.info("DeepONet model created")
        else:
            self.model = model

        self._setup_optimizer()
        self.early_stopping = EarlyStopping(
            patience=config.training.get("patience", 10), min_delta=config.training.get("min_delta", 0.0001)
        )

    def _setup_environment(self) -> None:
        """Configure TensorFlow env vars, logging, etc."""
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        if self.config.training.get("xla_compilation", False):
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
            logger.info("XLA compilation enabled")

        log_level = self.config.get("log_level", "INFO")
        logging.getLogger().setLevel(getattr(logging, log_level))
        logger.info(f"Environment configured. TensorFlow version: {tf.__version__}")

    def _setup_hardware(self) -> None:
        """Configure hardware for training.

        Sets up:
        - GPU memory growth
        - Distributed training with Horovod
        - Mixed precision
        """
        if self.distributed:
            self._setup_distributed()
        else:
            self._setup_single_device()

        if self.use_mixed_precision:
            self._setup_mixed_precision()

    def _setup_distributed(self) -> None:
        """Configure distributed training with Horovod."""
        hvd.init()
        self.rank = hvd.rank()

        gpus = tf.config.experimental.list_physical_devices("GPU")

        if gpus:
            try:
                # Make only one gpu visible to each process
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                logger.info(f"Rank {self.rank}: Using GPU {hvd.local_rank()}")
            except Exception as e:
                logger.error(f"Error setting up distributed GPUs: {e}")
                raise
        else:
            logger.warning(f"Rank {self.rank}: No GPUs found, using CPU")

    def _setup_single_device(self) -> None:
        """Configure single device training."""
        gpus = tf.config.experimental.list_physical_devices("GPU")

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                logger.info(f"Using {len(gpus)} GPU(s)")
            except Exception as e:
                logger.error(f"Error setting up GPUs: {e}")
                raise
        else:
            logger.info("No GPUs found, using CPU")

    def _setup_mixed_precision(self) -> None:
        """Configure mixed precision training."""
        try:
            # Use mixed precision policy
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")
        except Exception as e:
            logger.error(f"Error setting up mixed precision: {e}")
            self.use_mixed_precision = False
            logger.warning("Falling back to full precision training")

    def _setup_optimizer(self) -> None:
        """Configure the optimizer for training."""
        learning_rate = self.config.training.get("learning_rate", 0.001)
        optimizer_name = self.config.training.get("optimizer", "adam")

        if optimizer_name.lower() == "adam":
            base_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == "sgd":
            momentum = self.config.training.get("momentum", 0.9)
            base_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        else:
            logger.warning(f"Unknown optimizer {optimizer_name}, defaulting to Adam")
            base_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)  # pyright: ignore
        else:
            optimizer = base_optimizer

        if self.distributed:
            optimizer = hvd.DistributedOptimizer(optimizer)

        self.optimizer = optimizer
        logger.info(f"Optimizer configured: {optimizer_name} with lr={learning_rate}")

    @tf.function
    def _train_step(self, branch_batch, trunk_input, y_true, first_batch=False):
        """Execute one training step.

        Handles different batch structures between branch and trunk inputs.

        Args:
            branch_batch: Batch of branch inputs [batch_size, branch_dim]
            trunk_input: All trunk inputs [trunk_points, trunk_dim]
            y_true: Target outputs for this batch [batch_size, trunk_points]
            first_batch: Whether this is the first batch (for variable broadcasting)

        Returns:
            tf.Tensor: Loss value
        """
        with tf.GradientTape() as tape:
            y_pred = self.model((branch_batch, trunk_input), training=True)

            # y_pred and y_true both have shape [batch_size, trunk_points]?
            loss = self.model.compute_loss(y_pred, y_true)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.distributed and first_batch:
            hvd.broadcast_variables(self.model.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        return loss

    @tf.function
    def _evaluate_step(self, branch_input: tf.Tensor, trunk_input: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """Execute one evaluation step.

        Args:
            branch_input: Input tensor for branch network
            trunk_input: Input tensor for trunk network
            y_true: Target output tensor

        Returns:
            tf.Tensor: Loss value
        """
        y_pred = self.model((branch_input, trunk_input), training=False)
        return self.model.compute_loss(y_pred, y_true)

    def train(self, epochs: Optional[int] = None, batch_size: Optional[int] = None) -> Dict[str, List[float]]:
        """Train loop.

        Args:
            epochs: Number of epochs to train (overrides config)
            batch_size: Batch size for training (overrides config)

        Returns:
            Dict[str, List[float]]: Training history
        """
        epochs = epochs or self.config.training.get("epochs", 100)
        batch_size = batch_size or self.config.training.get("batch_size", 32)
        save_interval = self.config.training.get("save_interval", 10)

        logger.info("Loading training and validation data")
        train_data = self.data_loader.get_train_data(batch_size=batch_size)
        val_data = self.data_loader.get_val_data()

        branch_shape, trunk_shape, output_shape = self.data_loader.get_shapes()
        logger.info(f"Data shapes: branch={branch_shape}, trunk={trunk_shape}, output={output_shape}")

        start_time = time.time()
        best_val_loss = float("inf")
        stop_training = False

        if self.rank == 0:
            checkpoint_dir = self.config.training.get("checkpoint_dir", "./checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(f"Starting training for {epochs} epochs")
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train_losses = []

            for batch_idx, (branch_batch, trunk_batch, y_batch) in enumerate(train_data):
                loss = self._train_step(branch_batch, trunk_batch, y_batch, first_batch=(epoch == 1 and batch_idx == 0))
                if self.distributed:  # agg loss across workers
                    loss = hvd.allreduce(loss)
                train_losses.append(loss.numpy())

            avg_train_loss = np.mean(train_losses)

            # Validation
            branch_val, trunk_val, y_val = val_data
            val_loss = self._evaluate_step(branch_val, trunk_val, y_val)

            if self.distributed:
                val_loss = hvd.allreduce(val_loss)  # agg loss across workers

            val_loss = val_loss.numpy()

            # metrics on rank 0
            if self.rank == 0:
                self.model.history["epochs"].append(epoch)
                self.model.history["train_loss"].append(float(avg_train_loss))
                self.model.history["val_loss"].append(float(val_loss))

                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch}/{epochs} - {epoch_time:.2f}s - "
                    f"train_loss: {avg_train_loss:.6f} - val_loss: {val_loss:.6f}"
                )

                if epoch % save_interval == 0:
                    checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch:06d}.h5"
                    self.model.save_weights(checkpoint_path)
                    logger.info(f"Model saved to {checkpoint_path}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = f"{checkpoint_dir}/model_best.h5"
                    self.model.save_weights(best_model_path)
                    logger.info(f"New best model saved with val_loss={val_loss:.6f}")

                if self.early_stopping(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    stop_training = True

            if self.distributed:
                stop_training = hvd.broadcast_object(stop_training, root_rank=0)

            if stop_training:
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s after {epoch} epochs")

        if self.rank == 0:
            history_path = f"{checkpoint_dir}/training_history.json"
            self.model.save_history(history_path)

        return self.model.history


class EarlyStopping:
    """Early stopping callback to end training when validation loss stops improving.

    Attributes:
        patience: Number of epochs to wait after no improvement
        min_delta: Minimum change to qualify as improvement
        best_loss: Best loss value observed so far
        counter: Counter for epochs without improvement
        stopped: Whether early stopping has been triggered
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        """Set up early stopping.

        Args:
            patience: Number of epochs to wait after no improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stopped = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should be stopped.

        Args:
            val_loss: Current validation loss

        Returns:
            bool: True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:  # improved, dont stop
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:  # didnt improve
            self.counter += 1
            if self.counter >= self.patience:  # stalling, time to stop
                self.stopped = True
                return True
            return False  # keep going

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_loss = float("inf")
        self.counter = 0
        self.stopped = False
