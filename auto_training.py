from game import Game
from q_learning import QLearning
import path

def auto_training(training_episodes, load_dir = None):
    game = Game(path.MAPS / 'simpler')
    agent = QLearning(game, 10000000, 1000, 10000, 0.9, max_epsilon=1, gamma=0.95)
    if load_dir:
        agent.load(load_dir, True)
    agent.train(training_episodes, 3600, 4)
    agent.save('end')