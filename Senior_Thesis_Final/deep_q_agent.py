# A class containing all the code needed to build a Deep Q agent, with specific alterable values
# set up with default values if necessary
# This is in a separate file for readability
from tensorflow import keras
import pickle
from collections import deque
import numpy as np
import random
class DeepQAgent:
    def __init__(self, env, in_model=None, in_scores=None, in_mems=None,
                 out_model="default_model.h5", out_scores="default_scores.m", out_mems="default_memories.m",
                 eps_start=1.0, eps_min=0.01, eps_rate=0.99999, trials=500,
                 actions=9, lr=0.0001, model="dq3", memory_len=10000, batch=8, gamma=0.95,
                 learning_start=1000, target_freq=1000):
        self.env = env
        self.in_model = in_model
        self.in_scores = in_scores
        self.in_mems = in_mems
        if self.in_model is not None:
            self.in_model = "models/" + self.in_model
        if self.in_scores is not None:
            self.in_scores = "scores/" + self.in_scores
        if self.in_mems is not None:
            self.in_mems = "memories/" + self.in_mems
        # the out models are not none by default, and should never be set to none
        self.out_model = "models/" + out_model
        self.out_scores = "scores/" + out_scores
        self.out_mems = "memories/" + out_mems
        self.epsilon_start = eps_start
        self.epsilon_min = eps_min
        self.epsilon_rate = eps_rate
        self.trials = trials
        self.input_size = (86, 80, 1)
        self.action_size = actions
        self.learning_rate = lr

        self.model = self.init_model(model)
        self.target_model = self.init_model(model)

        self.memory_len = memory_len
        self.batch_size = batch
        self.gamma = gamma
        self.learning_start = learning_start
        self.step = 0
        self.target_freq = target_freq
        self.epsilon = self.epsilon_start
        self.scores = list()
        if self.in_scores is not None:
            sf = open(self.in_scores, "rb")
            self.scores = pickle.load(sf)
            sf.close()

        self.memory = deque(maxlen=self.memory_len)
        if self.in_mems is not None:
            mf = open(self.in_mems, "rb")
            self.memory = pickle.load(mf)
            mf.close()


    def init_model(self, model_name):
        if self.in_model is None:
            model = keras.models.Sequential()
            if model_name == "dq3":
                model.add(keras.layers.Conv2D(32, input_shape=self.input_size,
                                              kernel_size=[8, 8], strides=4,
                                              activation=keras.activations.relu))
                model.add(keras.layers.Conv2D(64, kernel_size=[4, 4], strides=2,
                                              activation=keras.activations.relu))
                model.add(keras.layers.Conv2D(64, kernel_size=[3, 3], strides=1,
                                              activation=keras.activations.relu))

                model.add(keras.layers.Flatten())
                model.add(keras.layers.Dense(512, activation=keras.activations.relu))
            elif model_name == "mnih":
                model.add(keras.layers.Conv2D(16, input_shape=self.input_size,
                                              kernel_size=[8, 8], strides=4,
                                              activation=keras.activations.relu))
                model.add(keras.layers.Conv2D(32, kernel_size=[4, 4], strides=2,
                                              activation=keras.activations.relu))
                model.add(keras.layers.Flatten())
                model.add(keras.layers.Dense(256, activation=keras.activations.relu))

            model.add(keras.layers.Dense(self.action_size))
            model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss=keras.losses.mean_squared_error)
        else:
            model = keras.models.load_model(self.in_model)
        return model

    def act(self, state):
        self.step += 1
        self.epsilon *= self.epsilon_rate  # reduce greedy epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon)  # not too low
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # random move
        return np.argmax(self.model.predict(state)[0])  # move w highest predicted score

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size):
        # choose a random set of samples from replay_memory
        # ideal return type is state/action/etc. tuples from gym
        samples = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in samples:
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            self.model.fit(state, target, epochs=1, verbose=0)  # this is where the mse loss function is used
            # as described in mnih paper, to compare the model's output based on state to the target given and update accordingly

    def target_train(self):
        # copy weights from model to target model
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)


    # preprocesses one frame, currently implemented
    def preprocess(self, img):
        # img is a 210x160x3 numpy ndarray
        # flatten to 2d grayscale image array and downscale (86x80x1)
        img = img[:172, :, :]
        img = img[::2, ::2, 0]

        # expand dimensions for model compatibility
        img = np.expand_dims(img, axis=3)
        img = np.expand_dims(img, axis=0)
        return img

        # saves model in hd5 file, scores and memories in a pickled file
    def save_model(self):
        self.target_model.save(self.out_model)
        name_scores = self.out_scores
        sf = open(name_scores, "wb")
        pickle.dump(self.scores, sf)
        sf.close()

        name_mem = self.out_mems
        mf = open(name_mem, "wb")
        pickle.dump(self.memory, mf)
        mf.close()
