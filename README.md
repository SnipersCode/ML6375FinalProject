# ML6375FinalProject
Deep Q Learning

## Requirements
* Python 3.8
* Linux or OS X
  * atari-py is not compatible with Windows.
  * If using windows, see windows requirements below...
  
### Windows
If running on windows, you will need the following in WSL Ubuntu:
* cmake
* python-opengl
* python3-tk
* add ```export DISPLAY=:0``` and ```export LIBGL_ALWAYS_INDIRECT=1``` to ~/.bashrc

You will need the following on the windows side if using WSL 1
* VcXsrv
  * Ensure VcXsrv is open before running code


## Installation

1. Install pipenv
    ```shell script
    pip install --user pipenv
    ```
1. Install dependencies
    ```shell script
    pipenv install --skip-lock
    ```
    You must use the `--skip-lock` flag as the normal hash calculations take too long for tensorflow

## References
* [DeepMind - Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
* [DeepMind - Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
* https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
* https://medium.com/analytics-vidhya/building-a-powerful-dqn-in-tensorflow-2-0-explanation-tutorial-d48ea8f3177a
* https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26?gi=1440299df424
* https://towardsdatascience.com/atari-reinforcement-learning-in-depth-part-1-ddqn-ceaa762a546f
