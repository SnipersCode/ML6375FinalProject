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
    pipenv install
    ```
