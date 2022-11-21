import pygame as pg
from game import Game
from q_learning import QLearning
import path

def testing():
    game = Game(path.MAPS / 'simpler')
    agent = QLearning(game, 1, 1, 2, 0.6)
    agent.load('go_forward', True)
    clock = pg.time.Clock()
    FPS = 60

    run = True
    while run:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                break
        
        clock.tick(FPS)

        state = game.get_state()
        game.make_action(agent.get_action(state))

        game.draw(game.MAP.gates[game.next_gate], True, True)
        
        if game.wall_collision():
            game.reset()
        game.gate_collision()
    pg.quit()