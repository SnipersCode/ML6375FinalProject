class HyperParams:
    PROCESS_HISTORY_LEN = 4
    LEARNING_RATE = 0.00001
    REPLAY_BATCH_SIZE = 32
    REPLAY_MAX_SIZE = 1000000
    MAX_FRAMES = 25000000
    TARGET_NET_UPDATE_FRAME_FREQ = 10000
    DISCOUNT = 0.99
    MAX_INIT_WAIT_FRAMES = 10
    UPDATE_FRAME_FREQ = 4

    # E-Greedy
    E_INITIAL = 1.0
    E_FINAL = 0.1
    E_EXPLORE_START_FRAME = 50000
    E_EXPLORE_END_FRAME = 1050000


class Constants:
    FRAME_SHAPE = (84, 84)
    MAX_FPS = 30
    EVAL_FRAME_FREQUENCY = 200000
    EVAL_STEPS = 10000
    PRINT_GAME_FREQ = 10
    ACTION_FIRE = 1
    RENDER = True
    MODEL_PATH = "./model"

    DO_LOAD = True
    DO_SAVE = True
