from game import Game
import pygame as pg
import path

def manual_movement(map):
    game = Game(path.MAPS / map)
    clock = pg.time.Clock()
    FPS = 60

    run = True
    while run:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                break
        
        clock.tick(FPS)

        game.get_state()
        game.move_player()

        game.draw(game.MAP.gates[game.next_gate], True, None, True)
                
        # print((state, action))
        if game.wall_collision():
            game.reset()
        game.gate_collision()
    pg.quit()