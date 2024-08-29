# Probabilistic Language-Aware Speech Recognition

## Environments
- python version: `3.9.18`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1+cu117`

# Installation
Basically follow the installation process of espnet following `https://espnet.github.io/espnet/installation.html`

Based on the ESPnet library and Whisper library, modify the code to apply the proposed method.

Step by step installation
1. Add deadsnake repo `add-apt-repository -y 'ppa:deadsnakes/ppa'`
2. Install python3.9 `apt install python3.9 python3.9-venv python3.9-dev`
3. Create python3.9 environment `python3.9 -m venv env39`
4. Activate the environment `source env39/bin/activate`
5. Go to tools directory and run `rm -f activate_python.sh && touch activate_python.sh`
5. Go to tools directory and install the espnet by `make TH_VERSION=1.13.1 CUDA_VERSION=11.7`
6. Install transformers tools by run `installers/install_transformers.sh`
6. Go to whisper directory `cd ../whisper` and then install the whisper library by `pip install -e .`


# Inference Model
We can utilize the code `whisper_check.py` in the `espnet/tools`. The weight is available in MLLAB's NAS `willianto_sulaiman/seame`.

Make sure to modify the path for the model, config, and the audio file path.

# Language Head Config
Language-Head structure, and also  can be seen in `espnet/whisper/whisper/model.py`

# Training Process
Example of training process utilizing SEAME Recipe

First make sure to put the dataset in the correct folder path, for SEAME, put the data under the `seame` folder

1. Run the `run.sh` to do the data preprocessing
2. Run the `run_whisper_language_rescore_.sh` to run the training process