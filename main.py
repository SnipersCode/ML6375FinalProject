import time

import matplotlib.pyplot as plt
from matplotlib import animation

from game import Game
from ddqlmodel import DDQLModel


def main():
    max_fps = 30

    with Game() as game:
        # Create Model
        model = DDQLModel()
        model.compile()

        print("Running game...")
        game.reset(evaluation=True)
        env_done = False
        while not env_done:
            time.sleep(1/max_fps)
            processed_frame, reward, env_done, lost_life, original_frame = game.next_state(render=True)

        print("Showing frame history...")
        fig = plt.figure()
        images = []
        for preprocessed_frame in game.frame_history:
            img = plt.imshow(preprocessed_frame, animated=True, cmap='gray', vmin=0, vmax=255)
            images.append([img])
        # noinspection PyUnusedLocal
        # animation must be saved to variable due to python garbage collection
        ani = animation.ArtistAnimation(fig, images, interval=1000/max_fps, blit=True)
        plt.show()


if __name__ == "__main__":
    main()
