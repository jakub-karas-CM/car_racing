import math
import numpy as np
import pygame as pg
import enum
from map import Line
from utils import rotate_image, lines_collided

class CarActions(enum.Enum):
    FORWARD = 0
    BACKWARD = 1
    LEFT_FORWARD = 2
    RIGHT_FORWARD = 3
    LEFT_BACKWARD = 4
    RIGHT_BACKWARD = 5
    # DO_NOTHING = 6
    # ROTATE_LEFT = 7
    # ROTATE_RIGHT = 8

class Car():
    def __init__(self, max_velocity, acceleration, rotation_velocity, start_position, starting_angle, vision_line_length, img):
        self.img = img
        self.width, self.height = self.img.get_width(), self.img.get_height()
        self.start_position = start_position
        self.starting_angle = starting_angle
        self.max_velocity = max_velocity
        self.max_reverse_velocity = -self.max_velocity / 2
        self.rotation_velocity = rotation_velocity
        self.acceleration = acceleration
        self.vision_angles = [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi, 5 * math.pi / 4, 3 * math.pi / 2, 7 * math.pi / 4]
        self.vision_line_length = vision_line_length
        self.vision_lines = [0] * len(self.vision_angles)
        self.closest_seen_points = [None] * len(self.vision_angles)
        self.closest_seen_points_distances = np.array([self.vision_line_length] * len(self.vision_angles))
        self.reset()

    def action_size(self):
        return max([x.value for x in CarActions]) + 1
        
    def rotate(self, left = False, right = False):
        if left:
            self.angle += self.rotation_velocity
        elif right:
            self.angle -= self.rotation_velocity

    def move_forward(self):
        self.velocity = min(self.velocity + self.acceleration, self.max_velocity)
        self.move()
    
    def move_backward(self):
        self.velocity = max(self.velocity - self.acceleration, self.max_reverse_velocity)
        self.move()
    
    def reduce_speed(self):
        if self.velocity < 0:
            self.velocity = min(self.velocity + self.acceleration / 2, 0)
        else:
            self.velocity = max(self.velocity - self.acceleration / 2, 0)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.velocity
        horizontal = math.sin(radians) * self.velocity

        self.y -= vertical
        self.x -= horizontal
        self.update_vision_lines()

    def move_with_action(self, action):
        if action == CarActions.FORWARD.value:
            self.move_forward()
        elif action == CarActions.BACKWARD.value:
            self.move_backward()
        elif action == CarActions.LEFT_FORWARD.value:
            self.rotate(left=True)
            self.move_forward()
        elif action == CarActions.RIGHT_FORWARD.value:
            self.rotate(right=True)
            self.move_forward()
        elif action == CarActions.LEFT_BACKWARD.value:
            self.rotate(left=True)
            self.move_backward()
        elif action == CarActions.RIGHT_BACKWARD.value:
            self.rotate(right=True)
            self.move_backward()
        # elif action == CarActions.ROTATE_LEFT.value:
        #     self.rotate(left=True)
        # elif action == CarActions.ROTATE_RIGHT.value:
        #     self.rotate(right=True)
        else:
            self.reduce_speed()
        
    def get_current_position(self):
        return np.array([self.x + self.width / 2, self.y + self.height / 2])
        
    def get_current_front(self):
        length = self.height / 2
        radians = math.radians(self.angle + 90)
        x, y = self.get_current_position()
        x += length * np.cos(radians)
        y -= length * np.sin(radians)
        return np.array([x, y])

    def update_vision_lines(self):
        x, y = self.get_current_position()
        rad = math.radians(self.angle)
        for i in range(len(self.vision_angles)):
            x2 = x + math.cos(rad + self.vision_angles[i])*self.vision_line_length
            y2 = y - math.sin(rad + self.vision_angles[i])*self.vision_line_length
            self.vision_lines[i] = Line(x, y, x2, y2)

    def get_normalized_state(self):
        '''Return array of inputs of neural network.'''
        normlized_vision_distances = self.closest_seen_points_distances / self.vision_line_length
        normalized_forward_velocity = max(0, self.velocity / self.max_velocity)
        normalized_backward_velocity = max(0, self.velocity / self.max_reverse_velocity) if self.max_reverse_velocity != 0 else 0
        normalized_angle = self.angle / 360
        normalized_state = [* normlized_vision_distances, normalized_forward_velocity, normalized_backward_velocity, normalized_angle]
        return np.array(normalized_state)

    def check_line_hit(self, line):
        radians = math.radians(self.angle)
        right_vector = np.array([math.cos(radians), -math.sin(radians)])
        up_vector = np.array([math.sin(radians), math.cos(radians)])
        car_corners = [0]*4
        corner_multipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        car_position = self.get_current_position()
        for i in range(4):
            car_corners[i] = car_position + (right_vector * self.width / 2 * corner_multipliers[i][0]) +(up_vector * self.height / 2 * corner_multipliers[i][1])

        for i in range(3):
            if lines_collided(line.x1, line.y1, line.x2, line.y2, car_corners[i][0], car_corners[i][1], car_corners[i + 1][0], car_corners[i + 1][1]):
                return True
        return False

    def check_wall_hit(self, wall):
        if self.check_line_hit(wall):
            self.dead = True
        return self.dead

    def check_gate_hit(self, gate):
        return self.check_line_hit(gate)
    
    def reset(self):
        self.dead = False
        self.x, self.y = self.start_position
        self.angle = self.starting_angle
        self.velocity = 0
        self.update_vision_lines()

    def draw(self, window, vision = False, center = False):
        rotate_image(window, self.img, (self.x, self.y), self.angle)
        if vision:
            for p in self.closest_seen_points:
                for l in self.vision_lines:
                    pg.draw.line(window, (0,255, 255), l.pixel1, l.pixel2, width = 2)
                if p is not None:
                    pg.draw.circle(window, (0, 255, 255), p, 4)
        if center:
            pg.draw.circle(window, (255, 255, 255), self.get_current_position(), 4)