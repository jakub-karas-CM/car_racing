import pygame as pg
import numpy as np
from game import Game
from q_learning import QLearning
from memory import Memory
import path

def manual_training(direction):
    game = Game(path.MAPS / 'simpler')
    clock = pg.time.Clock()
    FPS = 60

    run = True
    data = Memory(100000)
    i = 0
    while run and i < 100100:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                break
        
        clock.tick(FPS)

        state = game.get_state()
        action = game.move_player(direction)
        data.add(state, action, 0, 0, False)

        game.draw(game.MAP.gates[game.next_gate], True, True)
                
        if game.wall_collision():
            game.reset()
        game.gate_collision()
        i += 1
    print(f"{data.count()}x{data.buffer[0][0].shape[1]}")
    states = np.zeros((data.count(), game.get_state_size()))
    actions = np.zeros((data.count(), game.get_action_size()))
    for idx, d in enumerate(data.buffer):
        states[idx, :] = d[0]
        actions[idx, d[1]] = 1
    model = QLearning.build_network(state_size=game.get_state_size(), action_size=game.get_action_size(), optimizer='adam')
    model.fit(states, actions, verbose=0)
    model.save(path.MODELS / 'go_forward')
    pg.quit()