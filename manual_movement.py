from game import Game
import pygame as pg
import path

def manual_movement(map):
    '''This function enables user to take control over the vehicle.'''
    game = Game(path.MAPS / map)
    clock = pg.time.Clock()
    FPS = 60

    run = True
    while run:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
                break
        
        clock.tick(FPS) # syncing up program and display clock

        game.get_state()
        game.move_player()

        game.draw(game.MAP.gates[game.next_gate], True, None, True)
                
        if game.wall_collision():
            game.reset()
        game.gate_collision()
    pg.quit()