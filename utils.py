import pygame as pg
import math

def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pg.transform.scale(img, size)

def rotate_image(win, image, top_left, angle):
    rotated_image = pg.transform.rotate(image, angle)
    new_rectangle = rotated_image.get_rect(
        center = image.get_rect(topleft = top_left).center
    )
    win.blit(rotated_image, new_rectangle.topleft)

def lines_collided(x1, y1, x2, y2, x3, y3, x4, y4):
    uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    if 0 <= uA <= 1 and 0 <= uB <= 1:
        return True
    return False

def distance_to_line(x, y, line):
    return abs((line.x2 - line.x1) * (line.y1 - y) - (line.y2 - line.y1) * (line.x1 - x)) / math.sqrt((line.x2 - line.x1) ** 2 + (line.y2 - line.y1) ** 2)

def get_collision_point(x1, y1, x2, y2, x3, y3, x4, y4):
    uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    if 0 <= uA <= 1 and 0 <= uB <= 1:
        intersectionX = x1 + (uA * (x2 - x1))
        intersectionY = y1 + (uA * (y2 - y1))
        return (intersectionX, intersectionY)
    return None

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)