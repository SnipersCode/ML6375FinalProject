import time

import gym


def main():
    env = gym.make('BreakoutDeterministic-v4')
    env.reset()

    for _ in range(1000):
        env.render()
        time.sleep(1/30)
        env.step(env.action_space.sample())

    env.close()


if __name__ == "__main__":
    main()
