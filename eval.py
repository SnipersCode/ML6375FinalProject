from pathlib import Path
import pickle

# noinspection PyUnresolvedReferences
import tensorflow as tf

from main import evaluate
from parameters import Constants
from game import Game
from metadata import Metadata
from replaymem import ReplayMem
from ddqlagent import DDQLAgent


def main():
    # Create Metadata
    metadata = Metadata()
    # Create Replay Buffer
    g_buffer = ReplayMem(metadata)

    with Game(metadata) as game:
        # Create Agent
        agent = DDQLAgent(game.env.action_space.n)

        # Load files
        file_path = Path(Constants.MODEL_PATH)

        metadata_file = (file_path / "metadata.p")
        if all([x.validate_load() for x in [g_buffer, agent]]) and metadata_file.is_file():
            print("Loading state...")
            with metadata_file.open("rb") as metadata_file:
                metadata = pickle.load(metadata_file)  # Metadata
            game.load_metadata(metadata)
            g_buffer.load(metadata)
            agent.load()

        print("Running Eval...")
        evaluate(game, metadata, agent, 0)


if __name__ == "__main__":
    main()
