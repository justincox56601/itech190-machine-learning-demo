import pygame
import random
from Demo4Config import *

class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 60
        self.top_height = random.randint(80, HEIGHT - PIPE_GAP - 120)
        self.bottom_y = self.top_height + PIPE_GAP
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, surface):
        # Top Pipe
        pygame.draw.rect(surface, (70, 180, 80), (self.x, 0, self.width, self.top_height))
        pygame.draw.rect(surface, (40, 120, 50), (self.x, 0, self.width, self.top_height), 3)
        # Bottom Pipe
        pygame.draw.rect(surface, (70, 180, 80), (self.x, self.bottom_y, self.width, HEIGHT - self.bottom_y))
        pygame.draw.rect(surface, (40, 120, 50), (self.x, self.bottom_y, self.width, HEIGHT - self.bottom_y), 3)

    def check_collision(self, bird):
        if bird.x + 12 > self.x and bird.x - 12 < self.x + self.width:
            if bird.y - 12 < self.top_height or bird.y + 12 > self.bottom_y:
                return True
        return False
