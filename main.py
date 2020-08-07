import pickle
import signal
import sys
from pathlib import Path

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
    processed_frame, reward, game_end, lost_life, original_frame = game.next_state(next_action,
                                                                                   Constants.RENDER)

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


def evaluate(game: Game, metadata: Metadata, agent: DDQLAgent, num_epoch: int):
    eval_frame_num = 0
    eval_episode_rewards = []
    while eval_frame_num < Constants.EVAL_STEPS:
        eval_episode_reward = 0
        eval_episode_steps = 0
        game_end = False
        game.reset(evaluation=True)
        next_action = 1
        while not game_end and (eval_episode_steps < game.env.spec.max_episode_steps):
            processed_frame, reward, game_end, lost_life, original_frame = game.next_state(next_action,
                                                                                           Constants.RENDER)

            # Predict next action. Always start new life with a fire command
            next_action = agent.predict_action(game.network_input) if not lost_life else Constants.ACTION_FIRE

            eval_frame_num += 1
            eval_episode_steps += 1
            eval_episode_reward += reward
        eval_episode_rewards.append(eval_episode_reward)
        if len(eval_episode_rewards) % 10 == 0:
            print("Eval #{}> Frames:{}, Avg Episode Score:{}".format(num_epoch,
                                                                     eval_frame_num,
                                                                     np.mean(eval_episode_rewards)))

    # Summarize all eval episodes in current epoch
    metadata.eval_rewards.append(np.mean(eval_episode_rewards))
    print("Eval #{}> Episodes:{}, Final Score:{}".format(num_epoch,
                                                         len(eval_episode_rewards),
                                                         metadata.eval_rewards[-1]))


def execute(game: Game, metadata: Metadata, buffer: ReplayMem, agent: DDQLAgent):
    print("Running game...")
    num_epoch = 1
    # Run until max frames
    while metadata.frame_num < HyperParams.MAX_FRAMES:
        # Train network
        while metadata.frame_num // Constants.EVAL_FRAME_FREQUENCY < num_epoch:
            game.reset(evaluation=False)  # Initializes first frame
            game_end = False
            episode_reward = 0
            episode_steps = 0
            while not game_end and (episode_steps < game.env.spec.max_episode_steps):
                reward, game_end = train(metadata, agent, game, buffer)
                episode_reward += reward
                episode_steps += 1
            metadata.rewards.append(episode_reward)

            num_episodes = len(metadata.rewards)
            if num_episodes % Constants.PRINT_GAME_FREQ == 0:
                print("@Frame={} Rewards for games {}-{}: {}".format(
                    metadata.frame_num,
                    num_episodes - Constants.PRINT_GAME_FREQ + 1,
                    num_episodes,
                    np.mean(metadata.rewards[-10:])
                ))
        print("Running eval #{}...".format(num_epoch))
        # Periodically evaluate network
        evaluate(game, metadata, agent, num_epoch)
        num_epoch += 1


def save(file_path: Path, metadata: Metadata, buffer: ReplayMem, agent: DDQLAgent):
    # Save state before quitting (even interrupt)
    if Constants.DO_SAVE:
        print("Saving state...")
        file_path.mkdir(parents=True, exist_ok=True)

        with (file_path / "metadata.p").open("wb") as metadata_file:
            pickle.dump(metadata, metadata_file)  # Metadata
        buffer.save()
        agent.save()


# noinspection PyUnusedLocal
def _save_handler(sig, frame):
    print("\n>> Interrupt Detected")
    save(g_file_path, g_metadata, g_buffer, g_agent)
    print("exiting...")
    sys.exit(0)


if __name__ == "__main__":
    g_file_path = Path(Constants.MODEL_PATH)

    # Create Metadata
    g_metadata = Metadata()
    # Create Replay Buffer
    g_buffer = ReplayMem(g_metadata)

    with Game(g_metadata) as g_game:
        # Create Agent
        g_agent = DDQLAgent(g_game.env.action_space.n)
        signal.signal(signal.SIGINT, _save_handler)

        try:
            # Try Load
            g_metadata_file = (g_file_path / "metadata.p")
            if Constants.DO_LOAD and (all([x.validate_load() for x in [g_buffer, g_agent]])
                                      and g_metadata_file.is_file()):
                print("Loading state...")
                with g_metadata_file.open("rb") as g_metadata_file:
                    g_metadata = pickle.load(g_metadata_file)  # Metadata
                g_game.load_metadata(g_metadata)
                g_buffer.load(g_metadata)
                g_agent.load()

            execute(g_game, g_metadata, g_buffer, g_agent)

            save(g_file_path, g_metadata, g_buffer, g_agent)
        except SystemExit as e:
            raise e
