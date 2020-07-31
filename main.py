import time

import matplotlib.pyplot as plt
from matplotlib import animation

from game import Game


def main():
    max_fps = 30

    with Game() as game:
        print("Running game...")
        while not game.env_done:
            time.sleep(1/max_fps)
            game.next_state(render=True)

        print("Showing frame history...")
        fig = plt.figure()
        images = []
        for preprocessed_frame in game.frame_history:
            img = plt.imshow(preprocessed_frame, animated=True)
            images.append([img])
        # noinspection PyUnusedLocal
        # animation must be saved to variable due to python garbage collection
        ani = animation.ArtistAnimation(fig, images, interval=1000/max_fps, blit=True)
        plt.show()


if __name__ == "__main__":
    main()
