import numpy as np
from BirdNeuralNetwork import BirdNeuralNetwork
import pygame
from Demo4Config import *

class Bird:
    def __init__(self, brain=None):
        self.x = 80
        self.y = HEIGHT // 2
        self.velocity = 0
        self.alive = True
        self.fitness = 0
        self.score = 0
        self.brain = brain if brain else BirdNeuralNetwork()

    def step(self, next_pipe):
        if not self.alive:
            return

        self.fitness += 1
        self.velocity += GRAVITY
        self.y += self.velocity

        # Prepare 3 normalized inputs for the neural network
        dist_x = (next_pipe.x - self.x) / WIDTH
        gap_center = next_pipe.top_height + (PIPE_GAP / 2)
        dist_y = (gap_center - self.y) / HEIGHT
        norm_vel = self.velocity / 15.0

        inputs = np.array([[norm_vel, dist_x, dist_y]])

        # Jump if network output exceeds 0.5 confidence threshold
        if self.brain.forward(inputs) > 0.5:
            self.velocity = JUMP_STRENGTH

        # Boundary checks (floor or ceiling crash)
        if self.y <= 0 or self.y >= HEIGHT - 20:
            self.alive = False

    def draw(self, surface, is_best=False):
        if not self.alive:
            return
        color = (255, 220, 0) if is_best else (200, 100, 100)
        alpha_color = color if is_best else (180, 180, 180)
        pygame.draw.circle(surface, alpha_color, (int(self.x), int(self.y)), 12)
        pygame.draw.circle(surface, (0, 0, 0), (int(self.x), int(self.y)), 12, 2)
