import tensorflow as tf
import numpy as np
import random
import pathlib
import path
from neural_network import DeepQNetwork
from memory import Memory
from game import Game
from car import CarActions
import tensorflow as tf

class QLearning:
    def __init__(self, game: Game, memory_size, training_batch_size, learning_rate, gamma = 0.6, epsilon = 0.1) -> None:
        # Initialize atributes
        self.game = game
        self.state_size = self.game.get_state_size()
        self.action_size = self.game.get_action_size()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.memory = Memory(memory_size)
        self.training_batch_size = training_batch_size

        # Initialize discount and exploration rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Build networks
        self.q_network = self.build_network(self.state_size, self.action_size)
        self.target_network = self.build_network(self.state_size, self.action_size)
        self.align_target_model()

        # Populate memory
        self.pretrain()

    def build_network(self, state_size, action_size):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(10, input_shape=(state_size, ), activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(action_size, activation='linear'))        
        model.compile(loss='mse', optimizer="Adam")
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def store(self, state, action, reward, next_state, terminated):
        self.memory.add((state, action, reward, next_state, terminated))

    def pretrain(self):
        self.game.reset()
        state = self.game.get_state()
        for i in range(self.training_batch_size):
            # pick random movement
            action = random.randint(0, self.action_size - 1)
            # exercise the action
            self.game.make_action(action)
            # find out what happened
            reward = -1 # initialize if nothing particulary interesting happened
            if self.game.gate_collision():
                reward = 10
            if self.game.wall_collision():
                reward = -100
            next_state = self.game.get_state()
            # store the experience in memory
            self.memory.add(state, action, reward, next_state, self.game.is_episode_finished())

            # get ready for next move
            if self.game.is_episode_finished():
                self.game.reset()
                state = self.game.get_state()
            else:
                state = next_state

    def get_action(self, state):            
        self.memory.count()
        if np.random.rand() <= self.epsilon:
                return self.memory.sample(1)
        
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])