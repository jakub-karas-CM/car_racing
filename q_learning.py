import tensorflow as tf
import numpy as np
import pygame as pg
import random
import path
from memory import Memory
from game import Game

class QLearning:
    def __init__(self, game: Game, memory_size, training_batch_size, steps_to_align_target, learning_rate, gamma = 0.6, min_epsilon = 0.1, max_epsilon = 1) -> None:
        # Initialize atributes
        self.game = game
        self.state_size = self.game.get_state_size()
        self.action_size = self.game.get_action_size()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.memory = Memory(memory_size)
        self.training_batch_size = training_batch_size

        # Initialize discount and exploration rate
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay = 0
        
        # Build networks
        self.step = 0
        self.steps_to_align_target = steps_to_align_target
        self.q_network = self.build_network(self.state_size, self.action_size, self.optimizer)
        self.target_network = self.build_network(self.state_size, self.action_size, self.optimizer)
        self.align_target_model()

        # Populate memory
        self.pretrain()

    def build_network(self, state_size, action_size, optimizer):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(10, input_shape=(state_size, ), activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(action_size, activation='linear'))        
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def store(self, state, action, reward, next_state, terminated):
        self.memory.add((state, action, reward, next_state, terminated))

    def pretrain(self):
        state = self.game.get_state()
        position = self.game.player.get_current_position()
        for _ in range(self.training_batch_size):
            # pick random movement
            action = random.randint(0, self.action_size - 1)
            # exercise the action
            self.game.make_action(action)
            # find out what happened
            next_state = self.game.get_state()
            next_position = self.game.player.get_current_position()
            reward = self.get_reward(position, next_position)
            # store the experience in memory
            self.memory.add(state, action, reward, next_state, self.game.is_episode_finished())

            # get ready for next move
            if self.game.is_episode_finished():
                self.game.reset()
                state = self.game.get_state()
                position = self.game.player.get_current_position()
            else:
                state = next_state
                position = next_position

    def retrain_q_network(self):
        # get training batch
        states, actions, rewards, next_states, terminated = self.memory.sample(self.training_batch_size)
        # retrain network for each experience (this might be cause of instabilities in training, if so, consider training on whole minibatch)
        targets = self.q_network.predict(states, verbose=0)
        correct_targets = self.target_network.predict(next_states, verbose=0)
        correct_targets = np.amax(correct_targets, axis = 1) * (1 - terminated)
        for i in range(len(terminated)):
            targets[i][actions[i]] = rewards[i] + self.gamma * correct_targets[i]
        self.q_network.fit(states, targets, epochs=1, verbose=0)

    def train(self, training_episodes, max_steps, fps = 60):
        run = True
        clock = pg.time.Clock()

        for episode in range(training_episodes):
            self.game.reset()
            state = self.game.get_state()
            position = self.game.player.get_current_position()
            for _ in range(max_steps):
                self.step += 1
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        run = False
                        break
                clock.tick(fps)

                self.game.draw(self.game.MAP.gates[self.game.next_gate], True)
                # generate best action
                action = self.get_action(state)
                # exercise the action
                self.game.make_action(action)
                # find out what happened
                next_state = self.game.get_state()
                next_position = self.game.player.get_current_position()
                reward = self.get_reward(position, next_position)
                # store the experience in memory                
                self.memory.add(state, action, reward, next_state, self.game.is_episode_finished())
                # retrain Q Network
                self.retrain_q_network()
                # get ready for next move
                if self.game.is_episode_finished():
                    self.align_target_model()
                    break
                else:
                    if self.step >= self.steps_to_align_target:
                        self.step = 0
                        self.align_target_model()
                    state = next_state
                    position = next_position
            if not run:
                break

            if episode % 10 == 0 and episode % 15 != 0:
                self.save(f"episode_{episode}", memory=False)
            
            if episode % 10 != 0 and episode % 15 == 0:
                self.save(f"episode_{episode}", memory=True)

            print(f"Episode {episode} is done.")

    def get_action(self, state):            
        self.memory.count()
        if np.random.rand() <= self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-0.0001 * self.decay):
                return random.randint(0, self.action_size - 1)
        
        self.decay += 1
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def get_reward(self, position, next_position):
        reward = -1 # initialize for the case when nothing particulary interesting happened
        if self.game.gate_collision():
            reward = 10
        if self.game.wall_collision():
            reward += -100
        
        return reward

    def save(self, dir = None, memory = True):
        if not dir:
            dir = 'episode_{}'.format(self.game.game_number)
        # check if directory exists and remove all files if it does
        if not (path.MODELS / dir).exists():
            (path.MODELS / dir).mkdir()        
        # save the model
        self.q_network.save(path.MODELS / dir)
        # save the memory
        if memory:
            self.memory.save(path.MEMORY / (dir + '.csv'))

    def load(self, dir):
        # check directories
        if not (path.MODELS / dir).exists():
            raise FileExistsError("Directory for model doesn't exist.")
        if not (path.MEMORY / (dir + '.csv')).exists():
            raise FileNotFoundError("The memory storage file doesn't exist.")
        
        self.q_network = tf.keras.models.load_model(path.MODELS / dir)
        self.target_network = tf.keras.models.load_model(path.MODELS / dir)
        self.memory.load(path.MEMORY / (dir + '.csv'))