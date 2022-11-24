import tensorflow as tf
import numpy as np
import pygame as pg
import random
import path
from memory import Memory
from game import Game

class QLearning:
    def __init__(self, game: Game, memory_size, training_batch_size, steps_to_align_target, learning_rate, gamma, min_epsilon = 0.1, max_epsilon = 1) -> None:
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
        self.q_network = QLearning.build_network(self.state_size, self.action_size, self.optimizer)
        self.target_network = QLearning.build_network(self.state_size, self.action_size, self.optimizer)
        self.align_target_model()

        # Populate memory
        self.pretrain()

    @staticmethod
    def build_network(state_size, action_size, optimizer):
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
        for _ in range(self.training_batch_size):
            # pick random movement
            action = random.randint(0, self.action_size - 1)
            # exercise the action
            self.game.make_action(action)
            # find out what happened
            next_state = self.game.get_state()
            reward = self.get_reward(state[0][-1], state[0][-4])
            # store the experience in memory
            self.memory.add(state, action, reward, next_state, self.game.is_episode_finished())

            # get ready for next move
            if self.game.is_episode_finished():
                self.game.reset()
                state = self.game.get_state()
            else:
                state = next_state

    def retrain_q_network(self):
        # get training batch
        states, actions, rewards, next_states, terminated = self.memory.sample(self.training_batch_size)
        # retrain network
        q_eval = self.q_network.predict(states, verbose=0)
        q_next = self.target_network.predict(next_states, verbose=0)
        
        targets = q_eval.copy()
        batch_indices = np.arange(self.training_batch_size)
        targets[batch_indices, actions] = np.array(rewards) + self.gamma * np.amax(q_next, axis=1) * (1 - np.array(terminated))
        self.q_network.fit(states, targets, epochs=1, verbose=0)

    def train(self, training_episodes, max_steps, policy_steps, fps = 60):
        run = True
        clock = pg.time.Clock()
        scores = []
        epsilones = []
        for episode in range(training_episodes):
            self.game.reset()
            state = self.game.get_state()
            score = 0
            for _ in range(max_steps):
                self.step += 1
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        run = False
                        break

                # generate best action
                action = self.get_action(state)
                reward = 0
                for _ in range(policy_steps):                    
                    clock.tick(fps)
                    # exercise the action n times
                    self.game.make_action(action)
                    reward += self.get_reward(state[0][-1], state[0][-4])
                    # render window
                    self.game.draw(self.game.MAP.gates[self.game.next_gate], True, reward, False)
                    # check for boundary crossing
                    if self.game.is_episode_finished():
                        break
                # find out what happened
                next_state = self.game.get_state()
                score += reward
                # store the experience in memory                
                self.memory.add(state, action, reward, next_state, self.game.is_episode_finished())
                # retrain Q Network
                self.retrain_q_network()
                # get ready for next move
                if self.game.is_episode_finished():
                    break
                else:
                    if self.step >= self.steps_to_align_target:
                        self.step = 0
                        self.align_target_model()
                        print("Target network aligned.")
                    state = next_state
            if not run:
                break

            epsilones.append(self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay))
            scores.append(score)

            if episode % 20 == 0 and episode % 1500 != 0:
                self.save(f"episode_{episode}", memory=False)
            
            if episode % 20 != 0 and episode % 1500 == 0:
                self.save(f"episode_{episode}", memory=True)

            print(f"Episode {episode} is done. Score: {score}, avg score: {np.mean(scores[max(0, episode - 100):(episode+1)])}")

    def get_action(self, state):            
        self.memory.count()
        if np.random.rand() <= self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay):
                return random.randint(0, self.action_size - 1)
        
        self.decay += 1
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def get_reward(self, relative_distance_to_reward, relative_speed):
        collistion_punishment = -100
        gate_reward = -collistion_punishment / 10

        reward = -gate_reward *  (relative_distance_to_reward) / 10
        if self.game.gate_collision():
            reward = gate_reward
        if self.game.wall_collision():
            reward = collistion_punishment
        
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

    def load(self, dir, model_only = False):
        # check directories
        if not (path.MODELS / dir).exists():
            raise FileExistsError("Directory for model doesn't exist.")
        if not (path.MEMORY / (dir + '.csv')).exists() and not model_only:
            raise FileNotFoundError("The memory storage file doesn't exist.")
        
        self.q_network = tf.keras.models.load_model(path.MODELS / dir)
        self.target_network = tf.keras.models.load_model(path.MODELS / dir)
        if not model_only:
            self.memory.load(path.MEMORY / (dir + '.csv'))