class Metadata:
    def __init__(self):
        self.frame_num: int = 0
        self.rewards = []
        self.loss_list = []
        self.eval_rewards = []

        self.replay_buf_tail: int = 0
        self.replay_buf_filled: bool = False  # Denotes if entire memory has been filed at least once
