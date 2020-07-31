import time
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import gym
import cv2


class Game:
    def __init__(self):
        self.state_frame_len = 4

        self.frame_history: List[np.ndarray] = []
        self.current_frame: np.ndarray = np.array([])
        self.current_reward: float = 0.0
        self.env_done: bool = False
        self.env_info: dict = {}

    def __enter__(self):
        self.env = gym.make('BreakoutDeterministic-v4')
        self.env.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.close()

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        shape = img.shape
        # Reduce inputs by converting from RGB to Grayscale. Colors in game does not affect score.
        new_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # Resize image to reduce amount of processing needed
        return cv2.resize(new_img, (shape[1]//2, shape[0]//2))

    def random_action(self) -> object:
        return self.env.action_space.sample()

    def next_state(self, action: Union[object, None] = None, render=False):

        # Perform random action if action is None
        if not action:
            actual_action = self.random_action()
        else:
            actual_action = action

        # Tell game to move forward one frame
        self.current_frame, self.current_reward, self.env_done, self.env_info = self.env.step(actual_action)

        # Add to frame memory
        if len(self.frame_history) == 0:
            # Init frame history with copy of first frame state_frame_len times
            preprocessed = Game._preprocess(self.current_frame)
            for _ in range(0, self.state_frame_len):
                self.frame_history.append(preprocessed)
        else:
            self.frame_history.append(Game._preprocess(self.current_frame))

        if render:
            self.env.render()

        # return the last state_frame_len frames
        return np.array(self.frame_history[-self.state_frame_len:])


def main():
    max_fps = 120

    with Game() as game:
        print("Running game...")
        for _ in range(200):
            time.sleep(1/max_fps)
            game.next_state(render=False)

        print("Showing frame history...")
        fig = plt.figure()
        images = []
        for preprocessed_frame in game.frame_history:
            img = plt.imshow(preprocessed_frame, animated=True)
            images.append([img])
        # noinspection PyUnusedLocal
        ani = animation.ArtistAnimation(fig, images, interval=50, blit=True)  # animation must be saved to variable
        plt.show()


if __name__ == "__main__":
    main()
