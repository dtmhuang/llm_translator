## Introduction

just a demo for me to learn more about LLMs with a friend

our end goal is potentially to put it on the app store? might be a little ambitious

current goal is to train an LLM from scratch to learn more about neural networks

## Instructions

First, you need to install the dependencies

You can install them in a conda environment or by using pip with a native python environment
* `conda create -n [env_name] python=3.11 anaconda::transformers huggingface::datasets evaluate sacrebleu`: replace [env_name] with any name you want
    * `conda activate [env_name]`: you will have to run this to activate the environment
    * `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
    * There may be some trouble with the transformers, so you may also have to install transformers using pip
    * `pip install git+https://github.com/huggingface/transformers`
* `pip install transformers datasets evaluate sacrebleu`

To run the training script, type `python train.py`
