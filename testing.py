import pygame as pg
from game import Game
from q_learning import QLearning
import path

def testing(dir, map = 'simpler'):
    '''Testing of the trained AI.'''
    game = Game(path.MAPS / map)
    agent = QLearning(game, 1, 1, 1, 2, 0.6, 1) # those parameters are not important now
    agent.load(dir, True)
    agent.min_epsilon = 0 # we have to prevent the random actions, otherwise the test will not go well
    agent.max_epsilon = 0
    clock = pg.time.Clock()
    FPS = 60

    run = True
    while run:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                break
        
        clock.tick(FPS)

        # these two lines are the core of the testing
        state = game.get_state() # get state
        game.make_action(agent.get_action(state)) # pass it to the NN and execute the best action

        game.draw(game.MAP.gates[game.next_gate], True, None, True)
        
        if game.wall_collision():
            game.reset()
        game.gate_collision()
    pg.quit()