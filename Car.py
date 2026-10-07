from CarNeuralNetwork import CarNeuralNetwork
import numpy as np
import math
import random
from Track import * 

class Car:
    def __init__(self, brain=None):
        self.x, self.y = 160, 160
        self.angle = 0  # degrees
        self.speed = 0
        self.max_speed = 7.0
        self.alive = True
        self.distance_traveled = 0
        self.time_alive = 0
        self.brain = brain if brain else CarNeuralNetwork()
        self.ray_distances = [1.0] * 5
        self.ray_points = []

    def update_raycasts(self):
        # 5 Rays: -60°, -30°, 0° (center), 30°, 60°
        angles = [-60, -30, 0, 30, 60]
        max_dist = 250
        self.ray_distances = []
        self.ray_points = []

        for deg in angles:
            rad = math.radians(self.angle + deg)
            rx = self.x + math.cos(rad) * max_dist
            ry = self.y + math.sin(rad) * max_dist
            
            closest_dist = max_dist
            closest_pt = (rx, ry)

            for p1, p2 in TRACK_SEGMENTS:
                hit = line_intersection((self.x, self.y), (rx, ry), p1, p2)
                if hit:
                    d = math.hypot(hit[0] - self.x, hit[1] - self.y)
                    if d < closest_dist:
                        closest_dist = d
                        closest_pt = hit

            # Normalize distance inputs between 0.0 and 1.0
            self.ray_distances.append(closest_dist / max_dist)
            self.ray_points.append(closest_pt)

    def check_collision(self):
        # Check if car position is too close to any wall segment
        for p1, p2 in TRACK_SEGMENTS:
            # Simple point-to-segment proximity check
            rx = self.x + math.cos(math.radians(self.angle)) * 12
            ry = self.y + math.sin(math.radians(self.angle)) * 12
            if line_intersection((self.x, self.y), (rx, ry), p1, p2):
                return True
        return False

    def step(self):
        if not self.alive:
            return

        self.time_alive += 1
        self.update_raycasts()

        # Feed 5 sensor readings to Neural Net
        inputs = np.array(self.ray_distances).reshape(1, -1)
        steer, accel = self.brain.forward(inputs)

        # Apply controls
        self.angle += steer * 5.0
        self.speed = np.clip(self.speed + accel * 0.3, 1.0, self.max_speed)
        
        # Move
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        self.distance_traveled += self.speed

        # Check collision or timeout
        if self.check_collision() or min(self.ray_distances) < 0.04 or self.time_alive > 1200:
            self.alive = False

    def draw(self, surface, is_best=False):
        if not self.alive:
            return

        # Draw Raycasts for the best car
        if is_best:
            for pt in self.ray_points:
                pygame.draw.line(surface, (255, 255, 0), (self.x, self.y), pt, 1)
                pygame.draw.circle(surface, (255, 0, 0), (int(pt[0]), int(pt[1])), 3)

        # Draw Car Body
        color = (0, 255, 255) if is_best else (200, 100, 100)
        size = 10
        rad = math.radians(self.angle)
        front = (self.x + math.cos(rad) * size * 1.5, self.y + math.sin(rad) * size * 1.5)
        left = (self.x + math.cos(rad + 2.4) * size, self.y + math.sin(rad + 2.4) * size)
        right = (self.x + math.cos(rad - 2.4) * size, self.y + math.sin(rad - 2.4) * size)
        pygame.draw.polygon(surface, color, [front, left, right])