import pygame as pg
from game import Game
import path
from q_learning import QLearning

game = Game(path.MAPS / 'simpler')
agent = QLearning(game, 100000, 64, 0.1)

clock = pg.time.Clock()
FPS = 60

run = True
i = 0
while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
            break
    
    clock.tick(FPS)

    # state = game.get_state()
    # action = agent.get_action(state)

    game.draw(game.MAP.gates[game.next_gate], True)
    
    if i < agent.memory.count():
        game.make_action(agent.memory.buffer[i][1])
    else:
        game.move_player()

    print(game.get_state())
    if game.wall_collision():
        game.reset()
    game.gate_collision()
    i += 1

pg.quit()