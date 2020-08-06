import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from parameters import HyperParameters


class DDQLModel:
    def __init__(self, actions=4):
        self.actions = actions

    def compile(self, show_summary=True):
        input_layer = layers.Input(
            shape=(84, 84, self.actions),
            name='input_layer')
        scaling_layer = layers.Lambda(lambda layer: layer / 255,
                                      name="scale")(input_layer)

        # Use the same model as the paper
        # "The first convolutional layer has 32 8x8 filters with stride 4" and applies a rectifier nonlinearity
        hidden_layer1 = layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation=tf.nn.relu,
                                      kernel_initializer=keras.initializers.VarianceScaling(scale=2),
                                      padding="valid", use_bias=False, name='hidden_layer1')(scaling_layer)
        # "The second convolves 64 4×4 filters with stride 2" followed by a rectifier nonlinearity
        hidden_layer2 = layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation=tf.nn.relu,
                                      kernel_initializer=keras.initializers.VarianceScaling(scale=2),
                                      padding="valid", use_bias=False, name='hidden_layer2')(hidden_layer1)
        # "The third and final convolution layer consists 64 3x3 filters with stride 1" followed by a rectifier
        hidden_layer3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation=tf.nn.relu,
                                      kernel_initializer=keras.initializers.VarianceScaling(scale=2),
                                      padding="valid", use_bias=False, name='hidden_layer3')(hidden_layer2)

        # It is recommended to use another convolution layer instead of 2 separate fully connected layers with 512 units
        hidden_layer4 = layers.Conv2D(filters=1024, kernel_size=(7, 7), strides=1, activation=tf.nn.relu,
                                      kernel_initializer=keras.initializers.VarianceScaling(scale=2),
                                      padding="valid", use_bias=False, name='hidden_layer4')(hidden_layer3)

        # Split into value and advantage streams
        value_stream, advantage_stream = layers.Lambda(lambda layer: tf.split(layer, 2, 3),
                                                       name="split")(hidden_layer4)

        # Flatten each stream
        flattened_value_stream = layers.Flatten(name="value_flattened")(value_stream)
        flattened_advantage_stream = layers.Flatten(name="advantage_flattened")(advantage_stream)

        # Create value stream output layer: 1 output
        value_output_layer = layers.Dense(1, kernel_initializer=keras.initializers.VarianceScaling(scale=2),
                                          name="value_output")(flattened_value_stream)

        # Create advantage stream output layer: output per action
        advantage_output_layer = layers.Dense(self.actions, name="advantage_output")(flattened_advantage_stream)

        # Q value given an action and state =
        # value output of state +
        # (advantage output given an action and state - reduction mean of advantage output over all actions)

        # reduction mean of advantage output over all actions
        reduce_mean_layer = layers.Lambda(lambda layer: tf.reduce_mean(layer, axis=1, keepdims=True),
                                          name="reduce_mean")

        # (advantage output given an action and state - reduction mean of advantage output over all actions)
        subtract_layer = layers.Subtract(name="q_sub")(
            [advantage_output_layer, reduce_mean_layer(advantage_output_layer)])

        # Q values given action and state
        q_values = layers.Add(name="q_add")([value_output_layer, subtract_layer])

        # Build model using Huber loss
        model = tf.keras.models.Model(input_layer, q_values)
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=HyperParameters.LEARNING_RATE),
                      loss=tf.keras.losses.Huber())

        if show_summary:
            # Show Summary
            model.summary()

        return model
