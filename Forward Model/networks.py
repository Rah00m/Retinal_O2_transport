
"""
Four independent neural networks, one per retinal layer, each mapping
its OWN LOCAL coordinate z in [0, LAYER_LENGTH_UM] to the predicted
oxygen partial pressure u(z) in that layer.

Plain tf.keras networks are used (rather than dde.maps.FNN wrapped in
a dde.Model) because DeepXDE's Model class assumes a single network;
coupling four networks through shared interface losses requires a
custom training loop with direct access to each network's forward
pass under our own tf.GradientTape.
"""
import os
import tensorflow as tf
import config


def build_subnetwork(name):
    """Simple tanh MLP: 1 input (local z) -> 1 output (u, mmHg)."""
    inputs = tf.keras.Input(shape=(1,), name=f"z_{name}")
    x = inputs
    for width in config.HIDDEN_LAYERS:
        x = tf.keras.layers.Dense(
            width,
            activation=config.ACTIVATION,
            kernel_initializer="glorot_normal",
        )(x)
    outputs = tf.keras.layers.Dense(1, activation=None, name=f"u_{name}")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"pinn_{name}")


def build_all_networks():
    """Returns dict layer_name -> tf.keras.Model, one per layer in
    config.LAYERS ("IR", "OR", "FL", "CC")."""
    return {layer: build_subnetwork(layer) for layer in config.LAYERS}


def get_trainable_variables(nets):
    """Flat list of trainable variables across all 4 networks. Shared
    helper so train_model.py and evaluate_model.py don't each
    reimplement this loop."""
    variables = []
    for net in nets.values():
        variables += net.trainable_variables
    return variables


def save_networks(nets, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    for name, net in nets.items():
        net.save_weights(os.path.join(checkpoint_dir, f"net_{name}.weights.h5"))


def load_networks(checkpoint_dir):
    """Builds fresh networks with the same architecture as training and
    restores their weights. Architecture MUST match config.py exactly,
    which it will as long as config.py hasn't changed since training."""
    nets = build_all_networks()
    for name, net in nets.items():
        path = os.path.join(checkpoint_dir, f"net_{name}.weights.h5")
        net.load_weights(path)
    return nets