from pathlib import Path

import numpy as np
import tensorflow as tf

from ddqlmodel import DDQLModel
from parameters import HyperParams, Constants


class DDQLAgent:
    def __init__(self, actions: int):
        self.file_path = Path(Constants.MODEL_PATH) / "networks"

        base_model = DDQLModel(actions)
        self.main_network = base_model.build("main_network")
        self.target_network = base_model.build("target_network")
        self.actions = actions

        # Ensure target network has same weights
        self.update_target_network()

    def save(self):
        self.file_path.mkdir(parents=True, exist_ok=True)

        self.main_network.save(str(self.file_path / "main_network.h5"))
        self.target_network.save(str(self.file_path / "target_network.h5"))

    def load(self) -> bool:
        if not self.validate_load():
            return False

        self.main_network = tf.keras.models.load_model(str(self.file_path / "main_network.h5"))
        self.target_network = tf.keras.models.load_model(str(self.file_path / "main_network.h5"))

        return True

    def validate_load(self) -> bool:
        required_files = ["main_network.h5", "target_network.h5"]
        return all([(self.file_path / x).is_file() for x in required_files])

    @staticmethod
    def _epsilon(frame_num):
        if frame_num < HyperParams.E_EXPLORE_START_FRAME:
            return HyperParams.E_INITIAL
        elif frame_num < HyperParams.E_EXPLORE_END_FRAME:
            # Calculate point on slope

            ramp_frames_tot = HyperParams.E_EXPLORE_END_FRAME - HyperParams.E_EXPLORE_START_FRAME
            ramp_rate = (HyperParams.E_INITIAL - HyperParams.E_FINAL) / ramp_frames_tot

            ramp_frames_current = frame_num - HyperParams.E_EXPLORE_START_FRAME
            return HyperParams.E_INITIAL - ramp_rate * ramp_frames_current
        else:
            return HyperParams.E_FINAL

    def predict_action(self, network_input, frame_num=-1):
        # When training, set frame_num.
        # Evaluation will never pick a random value
        if frame_num != -1:
            # Determine epsilon
            e = self._epsilon(frame_num)

            # E-greedy random chance to pick random action
            if np.random.rand(1) < e:
                return np.random.randint(0, self.actions)

        # Otherwise predict
        q_values = self.main_network.predict(network_input)[0]
        return q_values.argmax()

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def _error(self, target_q, batch_actions, batch_states):
        # Update main network loss gradients and return error
        with tf.GradientTape() as tape:
            q_values = self.main_network(batch_states)

            actions = tf.keras.utils.to_categorical(batch_actions, self.actions, dtype=np.float32)
            q = tf.reduce_sum(tf.multiply(q_values, actions), axis=1)

            error = q - target_q
            loss = tf.keras.losses.Huber()(target_q, q)

        model_gradients = tape.gradient(loss, self.main_network.trainable_variables)
        self.main_network.optimizer.apply_gradients(zip(model_gradients, self.main_network.trainable_variables))

        return float(loss.numpy()), error

    def learn(self, batch_next_states, batch_rewards, batch_game_ends, batch_actions, batch_states):
        # Main network estimates best action for each next state
        q_values = self.main_network.predict(batch_next_states)
        action_max_q = q_values.argmax(axis=1)

        # Target network estimates the q-values for each next state based on the best action from the main network
        next_q_values = self.target_network.predict(batch_next_states)
        # Get the q values for the action with the highest q value for each state
        double_q = next_q_values[range(HyperParams.REPLAY_BATCH_SIZE), action_max_q]

        # Bellman Equation
        target_q = batch_rewards + (HyperParams.DISCOUNT * double_q * (1 - batch_game_ends))

        # Update loss and return error
        return self._error(target_q, batch_actions, batch_states)
