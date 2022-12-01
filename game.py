import pygame as pg
import numpy as np
from pathlib import Path
import json
from utils import scale_image, get_collision_point, distance, distance_to_line
from window_params import WIN_WIDTH, WIN_HEIGHT
from car import Car, CarActions
from map import Map

class Game():
    '''The overlord game class.'''
    def __init__(self, path_to_map):
        '''path_to_map -- path to the map files, the `maps` directory is recommended to you.'''
        pg.init()
        # set up backgroud images
        self.GRASS = scale_image(pg.image.load("imgs/grass.jpg"), 3)
        self.FINISH = pg.image.load("imgs/finish.png")
        # set up the game window
        self.WIDTH, self.HEIGHT = WIN_WIDTH, WIN_HEIGHT
        self.WIN = pg.display.set_mode((self.WIDTH, self.HEIGHT))
        # set up the player
        with (path_to_map / 'car.json').open() as fs:
            car_config = json.load(fs)
        self.CAR = scale_image(pg.image.load(str(Path('imgs') / car_config['model'])), car_config['model_scale'])
        self.player = Car(
            car_config['max_velocity'],
            car_config['acceleration'],
            car_config['max_angular_velocity'],
            car_config['starting_position'],
            car_config['starting_angle'],
            min(self.WIDTH, self.HEIGHT) / 2,
            self.CAR
        )
        # set up map
        walls = [x for x in (path_to_map / 'walls').glob('**/*') if x.is_file()]
        gates = [x for x in (path_to_map / 'gates').glob('**/*') if x.is_file()]
        self.TRACK = pg.image.load(str(path_to_map / "track.png"))
        self.MAP = Map(walls, gates)
        self.next_gate = 0
        # less gooo
        self.images = [(self.GRASS, (0, 0)), (self.TRACK, (0, 0))]
        pg.display.set_caption("Let's race!")

    def move_player(self, manual = None):
        '''Move player based on the prescribed action in `manual` or on the user's decision.'''
        keys = pg.key.get_pressed()
        action = -1
        moved = False
        if manual is None:
            if keys[pg.K_a] or keys[pg.K_LEFT]:
                action = CarActions.ROTATE_LEFT.value
                self.player.move_with_action(CarActions.ROTATE_LEFT.value)

            if keys[pg.K_d] or keys[pg.K_RIGHT]:
                action = CarActions.ROTATE_RIGHT.value
                self.player.move_with_action(CarActions.ROTATE_RIGHT.value)
            
            if keys[pg.K_w] or keys[pg.K_UP]:
                moved = True
                action = CarActions.FORWARD.value
                self.player.move_with_action(CarActions.FORWARD.value)
            
            if keys[pg.K_s] or keys[pg.K_DOWN]:
                moved = True
                action = CarActions.BACKWARD.value
                self.player.move_with_action(CarActions.BACKWARD.value)

            if not moved:
                if action == -1:
                    action = CarActions.DO_NOTHING.value
                self.player.move_with_action(CarActions.DO_NOTHING.value)
        else:
            action = manual
            self.player.move_with_action(manual)

        self.update_closest_seen_points()
        return action

    def distance_to_reward(self):
        '''Distance from the front of the car to the next reward gate.'''
        car_position = self.player.get_current_front()
        return distance_to_line(car_position[0], car_position[1], self.MAP.gates[self.next_gate])

    def wall_collision(self):
        '''Look for collision with a wall segment.'''
        not_hit = True
        for wall in self.MAP.walls:
            not_hit *= not self.player.check_wall_hit(wall)
        return not not_hit

    def update_closest_seen_points(self):
        car_x, car_y = self.player.get_current_position()
        # for every vision line check every wall segment, find the one that is closest to the car
        for i in range(len(self.player.vision_lines)):
            min_dist = 2 * max(self.HEIGHT, self.WIDTH)
            closest_point = None
            for wall in self.MAP.walls:
                collision_point = get_collision_point(wall.x1, wall.y1, wall.x2, wall.y2, self.player.vision_lines[i].x1, self.player.vision_lines[i].y1, self.player.vision_lines[i].x2, self.player.vision_lines[i].y2)
                if collision_point is not None:
                    dist = distance(car_x, car_y, collision_point[0], collision_point[1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = collision_point
            self.player.closest_seen_points[i] = closest_point
            self.player.closest_seen_points_distances[i] = min(min_dist, self.player.vision_line_length)

    def get_state(self):
        '''Get the full state vector.'''
        car_state = self.player.get_normalized_state()
        normalized_distance_to_reward = self.distance_to_reward() / self.MAP.distances_between_gates[self.next_gate]
        return np.array([[*car_state, normalized_distance_to_reward]])

    def get_state_size(self):
        return self.get_state().shape[1]

    def get_action_size(self):
        return self.player.action_size()

    def gate_collision(self):
        '''Check collision with the next reward gate.'''
        if self.player.check_gate_hit(self.MAP.gates[self.next_gate]):
            self.next_gate = (self.next_gate + 1) % len(self.MAP.gates)
            return True
        return False

    def make_action(self, action):
        '''Move player with action.'''
        self.player.move_with_action(action)

    def is_episode_finished(self):
        '''Check whether the player has died.'''
        return self.player.dead

    def reset(self):
        '''Reset the game.'''
        self.next_gate = 0
        self.player.reset()

    def draw(self, next, gates = False, text = None, vision = False):
        '''Draw the game. Mirrors functionality of map, car, line `draw` functions.'''
        for img, pos in self.images:
            self.WIN.blit(img, pos)
        self.MAP.draw(self.WIN, next, gates=gates)
        self.player.draw(self.WIN, vision)
        if text:            
            font = pg.font.SysFont('Arial', 20)
            text = font.render(f"Reward: {text[0]}\nAction: {text[1]}", True, (255, 255, 255), (0, 255, 0))
            rect = text.get_rect()
            rect.topleft = (10, 10)
            self.WIN.blit(text, rect)
        pg.display.update()    