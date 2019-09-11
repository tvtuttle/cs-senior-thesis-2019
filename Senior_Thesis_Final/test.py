# for testing packages
# to correctly install Gym and Ms. Pac-Man emulator on Windows, do the following:
# pip install gym (this can be done without difficulty)
# use command line, locate python and pip install locations
# use command "pip install git+https://github.com/Kojoley/atari-py.git" in command line
# then, pip install gym[atari]

# other installations: tensorflow or tensorflow-gpu depending on system
# (pip install tensorflow)
# numpy should already be installed as part of gym or tf
# pip install matplotlib

# so all packages needed: pip (if not already present), gym, tensorflow (or tensorflow-gpu, pick one), matplotlib
# all other necessary packages will be pip installed with these automatically

import gym
import numpy
import matplotlib.pyplot
from tensorflow import keras

import pickle
import os
import sys

model = keras.models.Sequential
env = gym.make("MsPacman-v0")

env.reset()
for i in range(200):
    env.render()
    env.step(2)
