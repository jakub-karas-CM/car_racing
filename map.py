from utils import distance_to_line
import pygame as pg
import pandas as pd
import numpy as np

class Line():
    def __init__(self, x1, y1, x2, y2):
        self.pixel1 = x1, y1
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.pixel2 = x2, y2
        self.y2 = y2

class Wall(Line):
    def draw(self, window):
        pg.draw.line(window, (0, 0, 0), self.pixel1, self.pixel2, width = 7)

class RewardGate(Line):
    def draw(self, window, colour = (190, 250, 0)):
        pg.draw.line(window, colour, self.pixel1, self.pixel2, width = 5)

class Map():
    def __init__(self, walls, gates):
        self.walls = []
        self.gates = []
        self.set_walls(walls)
        self.set_gates(gates)
        self.distances_between_gates = self.gate_distances()

    def set_walls(self, walls):
        for f in walls:
            points = np.array(pd.read_csv(f, sep=',', header=None))
            points = points.reshape((-1, 2))
            for i in range(points.shape[0] - 1):
                self.walls.append(Wall(points[i, 0], points[i, 1], points[i + 1, 0], points[i + 1, 1]))
    
    def set_gates(self, gates):
        for f in gates:
            points = np.array(pd.read_csv(f, sep=',', header=None))
            points = points.reshape((-1, 2))
            for i in range(0, points.shape[0] - 1, 2):
                self.gates.append(RewardGate(points[i, 0], points[i, 1], points[i + 1, 0], points[i + 1, 1]))

    def gate_distances(self):
        distances = np.arange(len(self.gates))
        for i in range(len(self.gates)):
            j = (i + 1) % len(self.gates)
            distances[i] = max(distance_to_line(self.gates[i].x1, self.gates[i].y1, self.gates[j]), distance_to_line(self.gates[i].x2, self.gates[i].y2, self.gates[j]))
        return distances

    def draw(self, window, next, gates = False):
        for wall in self.walls:
            wall.draw(window)
        if gates:
            for gate in self.gates:
                gate.draw(window)
            next.draw(window, (255, 0, 0))