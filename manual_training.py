import pygame as pg
import numpy as np
import logging
from game import Game
from q_learning import QLearning
from memory import Memory
import path

def manual_training(direction = None, map = 'simpler', memory_size = 100000, file = 'manual_driving'):
    '''Train the network by driving yourself. Or in constant `direction`.'''
    game = Game(path.MAPS / map)
    clock = pg.time.Clock()
    FPS = 60

    run = True
    data = Memory(memory_size, game.get_state_size())
    i = 0
    while run and i < memory_size + 100: # run for a little bit more than the mem size to overwrite the data from the beginning for the case of slow start
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                break        
        clock.tick(FPS)

        state = game.get_state()
        # execute the wanted function or listen to the user
        if direction is not None:
            action = game.move_player(direction)
        else:
            action = game.move_player()
        # add the experience to the memory
        data.add(state, action, 0, 0, False)
        # draw the game
        game.draw(game.MAP.gates[game.next_gate], True, None, True)
        # check collisions
        if game.wall_collision():
            game.reset()
        game.gate_collision()
        i += 1
    # log the result
    logging.getLogger().info(f"{data.count()}x{data.states.shape[1]}")
    # learn from the experience
    states = data.states
    actions = data.actions
    model = QLearning.build_network(state_size=game.get_state_size(), action_size=game.get_action_size(), optimizer='adam')
    model.fit(states, actions, verbose=0)
    # save it
    model.save(path.MODELS / file)
    pg.quit()