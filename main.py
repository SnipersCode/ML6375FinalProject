import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from game import Game
from replaymem import ReplayMem
from ddqlagent import DDQLAgent
from parameters import Constants, HyperParams
from metadata import Metadata


def train(metadata: Metadata, agent: DDQLAgent, game: Game, buffer: ReplayMem):
    # Predict next action
    next_action = agent.predict_action(game.network_input, metadata.frame_num)

    # Execute game and observe next state
    processed_frame, reward, game_end, lost_life, original_frame = game.next_state(next_action, True)

    # Save transition to replay memory
    buffer.add_exp(next_action, processed_frame, reward, lost_life)

    # Perform updates if current frame number is greater than the explore start frame
    if metadata.frame_num > HyperParams.E_EXPLORE_START_FRAME:
        if metadata.frame_num % HyperParams.UPDATE_FRAME_FREQ == 0:
            batch_info = buffer.get_batch()  # Sample replay memory
            loss = agent.learn(*batch_info)  # Determine loss and calculate gradients
            metadata.loss_list.append(loss)
        if metadata.frame_num % HyperParams.TARGET_NET_UPDATE_FRAME_FREQ == 0:
            agent.update_target_network()  # Update target network

    return reward, game_end


def main():
    alerted = False

    # Create Metadata
    metadata = Metadata()
    # Create Replay Buffer
    buffer = ReplayMem(metadata)

    with Game(metadata) as game:
        # Create Agent
        agent = DDQLAgent(game.env.action_space.n)

        print("Running game...")
        # Run until max frames
        while metadata.frame_num < HyperParams.MAX_FRAMES:
            # Periodically evaluate network
            while (metadata.frame_num % Constants.EVAL_FRAME_FREQUENCY != 0) or (metadata.frame_num == 0):
                game.reset(evaluation=False)  # Initializes first frame
                game_end = False
                episode_reward = 0
                while not game_end:
                    reward, game_end = train(metadata, agent, game, buffer)
                    episode_reward += reward
                metadata.rewards.append(episode_reward)

                num_episodes = len(metadata.rewards)
                print("Episode {}".format(num_episodes))
                if num_episodes % Constants.PRINT_GAME_FREQ == 0:
                    print("@Frame={} Rewards for games {}-{}: {}".format(
                        metadata.frame_num,
                        num_episodes - Constants.PRINT_GAME_FREQ + 1,
                        num_episodes,
                        np.mean(metadata.rewards[-10:])
                    ))
                if (metadata.frame_num > HyperParams.E_EXPLORE_START_FRAME) and not alerted:
                    print("===> Start updating networks")
                    alerted = True

        # print("Showing frame history...")
        # fig = plt.figure()
        # images = []
        # for preprocessed_frame in game.frame_history:
        #     img = plt.imshow(preprocessed_frame, animated=True, cmap='gray', vmin=0, vmax=255)
        #     images.append([img])
        # # noinspection PyUnusedLocal
        # # animation must be saved to variable due to python garbage collection
        # ani = animation.ArtistAnimation(fig, images, interval=1000/Constants.MAX_FPS, blit=True)
        # plt.show()


if __name__ == "__main__":
    main()
