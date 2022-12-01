from game import Game
from q_learning import QLearning
import path

def auto_training(training_episodes, load_dir = None, model_only = False, map = 'simpler', max_steps = 500,
                  policy_steps = 4, memory_size = 5000, training_batch_size = 100, steps_to_align = 1000,
                  learning_rate = 0.4, gamma = 0.5, min_epsilon = 0.025, max_epsilon = 1, file = 'end'):
    '''This function serves to make hyperparameter setup easier. See the `QLearning` class for explanations.'''
    game = Game(path.MAPS / map)
    agent = QLearning(game, memory_size, training_batch_size, 1 if load_dir and not model_only else memory_size,
                      steps_to_align, learning_rate, gamma=gamma, min_epsilon=min_epsilon, max_epsilon=max_epsilon)
    if load_dir:
        agent.load(load_dir, model_only)
    agent.train(training_episodes, max_steps, policy_steps)
    agent.save(file)