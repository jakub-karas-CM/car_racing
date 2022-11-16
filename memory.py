from collections import deque
import numpy as np
import path
import pandas as pd

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)

    def add(self, state, action, reward, next_state, is_episode_finished):
        self.buffer.append((state, action, reward, next_state, is_episode_finished))

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        rnd_indeces = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in rnd_indeces]

    def count(self):
        return len(self.buffer)

    def save(self, file = path.MEMORY / 'memory.csv'):
        count = self.count()
        if count == 0:
            raise Exception("Nothing to save.")
        state_size = self.buffer[0][0].shape[1]
        
        dataframe = pd.DataFrame()
        for i in range(state_size):
            dataframe['state_{}'.format(i)] = [self.buffer[r][0][0, i] for r in range(count)]
        dataframe['action'] = [self.buffer[r][1] for r in range(count)]
        dataframe['reward'] = [self.buffer[r][2] for r in range(count)]
        for i in range(state_size):
            dataframe['next_state_{}'.format(i)] = [self.buffer[r][3][0, i] for r in range(count)]
        dataframe['is_episode_finished'] = [self.buffer[r][4] for r in range(count)]
        dataframe.to_csv(file, index=False)

    def load(self, file = path.MEMORY / 'memory.csv'):
        dataframe = pd.read_csv(file)
        state_size = int((dataframe.shape[1] - 3) / 2)
        for i in range(dataframe.shape[0]):
            row = dataframe.iloc[i]
            self.add(np.array(row[0:state_size]), row[state_size], row[state_size + 1], np.array(row[state_size + 2:-2]), row[-1])