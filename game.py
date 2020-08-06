from typing import List, Union, Tuple

import numpy as np
import gym
import cv2

from metadata import Metadata
from parameters import HyperParams, Constants


class Game:
    def __init__(self, metadata: Metadata, initial_action: int = 1, game_str: str = 'BreakoutDeterministic-v4'):
        # Run breakout by default
        self.game_str = game_str
        # Initial action by default is "fire" to start the game in breakout
        self.initial_action = initial_action

        # Initialize state
        self.metadata = metadata
        self.frame_history: List[np.ndarray] = []
        self.network_input: Union[None, np.ndarray] = None
        self.lives: int = -1

    def __enter__(self):
        # Create Environment
        self.env = gym.make(self.game_str)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.close()

    def reset(self, evaluation=False):
        # Reinitialize state
        self.frame_history: List[np.ndarray] = []
        self.network_input: Union[None, np.ndarray] = None

        # Reset environment
        self.env.reset()

        original_frame, reward, game_end, env_info = self._next_frame(self.initial_action)
        if evaluation:
            # Start at a random point when evaluating
            # This prevents machine from learning a fixed sequence of actions
            for _ in range(np.random.randint(0, HyperParams.MAX_INIT_WAIT_FRAMES - 1)):
                original_frame, reward, game_end, env_info = self._next_frame(self.initial_action)

        # Set initial number of lives
        self.lives = env_info["ale.lives"]

        # Duplicate initial frame by PROCESS_HISTORY_LEN
        processed_frame = self._preprocess(original_frame)
        for _ in range(HyperParams.PROCESS_HISTORY_LEN):
            self.frame_history.append(processed_frame)

        # Update Network Input
        self._set_network_input()

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        # Reduce inputs by converting from RGB to Grayscale. Colors in game does not affect score.
        # Crop to a 160x160 for faster processing
        gray_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)[34:34+160, :160]
        # Resize image to size required for model (84x84)
        return cv2.resize(gray_img, Constants.FRAME_SHAPE, interpolation=cv2.INTER_NEAREST)

    def _next_frame(self, action: Union[int, None]) -> Tuple[np.ndarray, float, bool, dict]:
        # Perform random action if action is None
        if not action:
            actual_action = self.random_action()
        else:
            actual_action = action

        # Tell game to move forward one frame
        self.metadata.frame_num += 1
        return self.env.step(actual_action)

    def random_action(self) -> object:
        return self.env.action_space.sample()

    def _set_network_input(self):
        # Update network input representation of last 4 frames in frame history
        # Transpose as model assumes shape (width, height, frame_idx). Currently (frame_idx, width, height).
        self.network_input = np.transpose(np.asarray(self.frame_history[-4:]), axes=(1, 2, 0))
        # Model assumes an array of states, hence shape of (1, width, height, frame_idx).
        self.network_input = self.network_input.reshape((-1, *self.network_input.shape))

    def next_state(self, action: Union[int, None] = None, render=False):
        # Run next frame
        original_frame, reward, game_end, env_info = self._next_frame(action)

        # Add to frame memory
        processed_frame = self._preprocess(original_frame)
        self.frame_history.append(processed_frame)

        # Check if lost a life (either fell in pit or game ended)
        lost_life = (env_info["ale.lives"] < self.lives) or game_end

        # Update Network Input
        self._set_network_input()

        # Update lives
        self.lives = env_info["ale.lives"]

        if render:
            self.env.render()

        return processed_frame, reward, game_end, lost_life, original_frame
