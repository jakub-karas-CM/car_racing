import numpy as np
import path
import pandas as pd

class Memory:
    '''The experience memory of the AI.'''
    def __init__(self, max_size, state_size):
        self.max_size = max_size
        self.state_size = state_size
        self.counter = 0 # a counter of written experiences
        # initialize the vector of number of times the network has learned from an experience
        self.seen = np.zeros(self.max_size, dtype=np.int32)
        # setup buffers
        self.states = np.zeros((self.max_size, self.state_size))
        self.actions = np.zeros(self.max_size, dtype=np.int32)
        self.rewards = np.zeros(self.max_size)
        self.next_states = np.zeros((self.max_size, self.state_size))
        self.terminated = np.zeros(self.max_size, dtype=bool)

    def add(self, state, action, reward, next_state, is_episode_finished):
        '''Add an experience to memory'''
        # find index to overwrite
        index = self.counter % self.max_size
        # reset seen times
        self.seen[index] = 0
        # overwrite data
        self.states[index, :] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index, :] = next_state
        self.terminated[index] = is_episode_finished
        # increase the counter
        self.counter += 1

    def sample(self, batch_size):
        '''Retrieve batch of experience data of specified size'''
        # find the random indeces of experiences
        buffer_size = self.count()
        rnd_indeces = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        # increment the number of times the experience was seen
        self.seen[rnd_indeces] = self.seen[rnd_indeces] + 1
        # retrieve the data
        states = self.states[rnd_indeces, :]
        actions = self.actions[rnd_indeces]
        rewards = self.rewards[rnd_indeces]
        next_states = self.next_states[rnd_indeces, :]
        terminated = self.terminated[rnd_indeces]
        seen = self.seen[rnd_indeces]
        return states, actions, rewards, next_states, terminated, seen

    def count(self):
        return self.counter if self.counter < self.max_size else self.max_size

    def save(self, file = path.MEMORY / 'memory.csv'):
        '''Save the memory contents to a csv file'''
        count = self.count()
        if count == 0:
            raise Exception("Nothing to save.")
        # the process of saving is done using the pandas dataframe
        dataframe = pd.DataFrame()
        # create a column in the dataframe for each variable
        for i in range(self.state_size):
            dataframe['state_{}'.format(i)] = self.states[:, i]
        dataframe['action'] = self.actions
        dataframe['reward'] = self.rewards
        for i in range(self.state_size):
            dataframe['next_state_{}'.format(i)] = self.next_states[:, i]
        dataframe['is_episode_finished'] = self.terminated
        dataframe['seen'] = self.seen
        # save it
        dataframe.to_csv(file, index=False)

    def load(self, file = path.MEMORY / 'memory.csv'):
        '''Load memory from a csv file'''
        # the process of loading is done using the pandas dataframe
        dataframe = pd.read_csv(file)
        if dataframe.shape[1] == 2 * self.state_size + 4:
            for i in range(dataframe.shape[0]):
                row = dataframe.iloc[i]
                # add an experience to memory, faster way would probably be to insert the whole columns,
                # but since this function is run in O(const.) times, it is not that important
                self.add(
                    np.array(row[0:self.state_size]).reshape((1, -1)),
                    row[self.state_size],
                    row[self.state_size + 1],
                    np.array(row[self.state_size + 2:-2]).reshape((1, -1)),
                    row[-2],
                    row[-1]
                )
        else:
            for i in range(dataframe.shape[0]):
                row = dataframe.iloc[i]
                # add an experience to memory, faster way would probably be to insert the whole columns,
                # but since this function is run in O(const.) times, it is not that important
                self.add(
                    np.array(row[0:self.state_size]).reshape((1, -1)),
                    row[self.state_size],
                    row[self.state_size + 1],
                    np.array(row[self.state_size + 2:-1]).reshape((1, -1)),
                    row[-1]
                )