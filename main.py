import time

import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2


def preprocess(img: np.ndarray) -> np.ndarray:
    shape = img.shape
    # Reduce inputs by converting from RGB to Grayscale. Colors in game does not affect score.
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize image to reduce amount of processing needed
    return cv2.resize(new_img, (shape[1]//2, shape[0]//2))


def init():
    # Play Atari Breakout
    env = gym.make('BreakoutDeterministic-v4')
    env.reset()
    return env


def main():
    env = init()
    for _ in range(1000):
        # env.render()
        time.sleep(1/30)
        frame, reward, is_end, _ = env.step(env.action_space.sample())
        plt.imshow(preprocess(frame))
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
