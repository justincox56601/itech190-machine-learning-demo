import sys
import math
import random
import numpy as np
import pygame
from Car import Car
from Track import *

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1000, 700
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Demo 3: Self-Driving Car (Raycast Sensors + Neural Nets)")
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("Consolas", 16)

# =====================================================================
# STEP 1: Track Construction (Inner & Outer Boundaries)
# Track.py
# =====================================================================

# =====================================================================
# STEP 2: Neural Network Architecture
# CarNueralNetwork.py
# =====================================================================

# =====================================================================
# STEP 3: Car Entity & Physics
# Car.py
# =====================================================================

# =====================================================================
# STEP 4: Population Simulation Loop
# =====================================================================
POPULATION_SIZE = 25
generation = 1
population = [Car() for _ in range(POPULATION_SIZE)]

running = True
while running:
    CLOCK.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

    # Step all alive cars
    alive_cars = [c for c in population if c.alive]
    for car in alive_cars:
        car.step()

    # Find leading car
    best_car = max(population, key=lambda c: c.distance_traveled)

    # Next Generation Evolution trigger
    if len(alive_cars) == 0:
        generation += 1
        # Sort by distance traveled
        population.sort(key=lambda c: c.distance_traveled, reverse=True)
        top_parents = population[:4]  # Keep top 4 best performers
        
        new_pop = []
        # Elitism: keep best brain untouched
        new_pop.append(Car(brain=top_parents[0].brain))

        # Create mutated offspring
        while len(new_pop) < POPULATION_SIZE:
            parent = random.choice(top_parents)
            child_brain = parent.brain.mutate(rate=0.15, magnitude=0.3)
            new_pop.append(Car(brain=child_brain))
            
        population = new_pop
        continue

    # Render Frame
    draw_track(SCREEN)
    
    # Draw all cars (draw non-best cars first, then draw best on top)
    for car in population:
        if car != best_car:
            car.draw(SCREEN, is_best=False)
    best_car.draw(SCREEN, is_best=True)

    # HUD Stats Overlay
    info_lines = [
        f"Generation: {generation}",
        f"Cars Alive: {len(alive_cars)} / {POPULATION_SIZE}",
        f"Best Distance: {int(best_car.distance_traveled)} px",
        f"Sensors (L->R): {['%.2f'%d for d in best_car.ray_distances]}"
    ]
    for i, line in enumerate(info_lines):
        txt = FONT.render(line, True, (255, 255, 255))
        SCREEN.blit(txt, (20, 20 + i * 22))

    pygame.display.flip()