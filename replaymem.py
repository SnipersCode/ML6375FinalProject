from pathlib import Path

import numpy as np

from parameters import HyperParams, Constants
from metadata import Metadata


class ReplayMem:
    def __init__(self, metadata: Metadata):
        self.metadata = metadata
        self.file_path = Path(Constants.MODEL_PATH) / "replay"

        # Allocate memory
        self.actions = np.empty(HyperParams.REPLAY_MAX_SIZE, dtype=np.uint8)
        self.frames = np.empty((HyperParams.REPLAY_MAX_SIZE, *Constants.FRAME_SHAPE), dtype=np.uint8)
        self.rewards = np.empty(HyperParams.REPLAY_MAX_SIZE, dtype=np.uint32)
        self.game_ends = np.empty(HyperParams.REPLAY_MAX_SIZE, dtype=np.bool)

    def __len__(self):
        return HyperParams.REPLAY_MAX_SIZE if self.metadata.replay_buf_filled else self.metadata.replay_buf_tail

    def save(self):
        self.file_path.mkdir(parents=True, exist_ok=True)

        np.save(str(self.file_path / "actions.npy"), self.actions)
        np.save(str(self.file_path / "frames.npy"), self.frames)
        np.save(str(self.file_path / "rewards.npy"), self.rewards)
        np.save(str(self.file_path / "game_ends.npy"), self.game_ends)

    def load(self, metadata: Metadata) -> bool:
        if not self.validate_load():
            return False

        self.actions = np.load(str(self.file_path / "actions.npy"))
        self.frames = np.load(str(self.file_path / "frames.npy"))
        self.rewards = np.load(str(self.file_path / "rewards.npy"))
        self.game_ends = np.load(str(self.file_path / "game_ends.npy"))

        self.metadata = metadata

        return True

    def validate_load(self) -> bool:
        required_files = ["actions.npy", "frames.npy", "rewards.npy", "game_ends.npy"]
        return all([(self.file_path / x).is_file() for x in required_files])

    def add_exp(self, action, frame, reward, game_end):
        self.actions[self.metadata.replay_buf_tail] = action
        self.frames[self.metadata.replay_buf_tail] = frame
        self.rewards[self.metadata.replay_buf_tail] = reward  # Clip reward for faster processing
        self.game_ends[self.metadata.replay_buf_tail] = game_end

        if self.metadata.replay_buf_tail < HyperParams.REPLAY_MAX_SIZE - 1:
            self.metadata.replay_buf_tail += 1
        else:
            self.metadata.replay_buf_tail = 0  # Overwrite old frames
            self.metadata.replay_buf_filled = True

    def get_batch(self):
        # Find groups of PROCESS_HISTORY_LEN and next state that are from the same episode
        indices = []
        while len(indices) < HyperParams.REPLAY_BATCH_SIZE:
            index = np.random.randint(0, len(self) - HyperParams.PROCESS_HISTORY_LEN - 1)
            # Ensure index to index + PROCESS_HISTORY_LEN are from the same episode by checking if any is the end
            # If none are the end, next state (which can be the end) will definitely be in the same episode
            if not self.game_ends[index:index + HyperParams.PROCESS_HISTORY_LEN].any():
                indices.append(index)

        # Lookup frames related to indices
        states = []
        next_states = []
        for idx in indices:
            states.append(self.frames[idx:idx + HyperParams.PROCESS_HISTORY_LEN, ...])
            next_states.append(self.frames[idx + 1:idx + 1 + HyperParams.PROCESS_HISTORY_LEN, ...])
        # Transpose as model assumes shape (width, height, frame_idx). Currently (batch_num, frame_idx, width, height).
        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        next_states = np.transpose(np.asarray(next_states), axes=(0, 2, 3, 1))

        return next_states, self.rewards[indices], self.game_ends[indices], self.actions[indices], states
