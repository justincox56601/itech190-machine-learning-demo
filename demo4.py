import sys
import random
import numpy as np
import pygame
from Bird import Bird
from Pipe import Pipe
from Demo4Config import *

# Initialize Pygame
pygame.init()
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Demo 4: Flappy Bird AI (Neural Nets + Evolution)")
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("Consolas", 16)



# =====================================================================
# STEP 1: Neural Network Architecture
# BirdNeuralNetwork.py
# =====================================================================

# =====================================================================
# STEP 2: Bird & Pipe Entities
# Bird.py
# Pipe.py
# =====================================================================

# =====================================================================
# STEP 3: Evolutionary Simulation Loop
# =====================================================================
POPULATION_SIZE = 20
generation = 1
high_score = 0

population = [Bird() for _ in range(POPULATION_SIZE)]
pipes = [Pipe(WIDTH + 100)]
frame_count = 0

running = True
while running:
    CLOCK.tick(60)
    frame_count += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

    # Spawn new pipes periodically
    if frame_count % PIPE_FREQUENCY == 0:
        pipes.append(Pipe(WIDTH))

    # Move pipes and prune off-screen ones
    for pipe in pipes:
        pipe.update()
    pipes = [p for p in pipes if p.x + p.width > 0]

    # Find the target pipe ahead of the population
    next_pipe = next((p for p in pipes if p.x + p.width > population[0].x - 12), pipes[0])

    # Update birds and check pipe collisions
    alive_birds = [b for b in population if b.alive]
    for bird in alive_birds:
        bird.step(next_pipe)
        if next_pipe.check_collision(bird):
            bird.alive = False

        # Award points for passing pipes
        if not next_pipe.passed and next_pipe.x + next_pipe.width < bird.x:
            bird.score += 1

    if any(p.x + p.width < population[0].x for p in pipes if not p.passed):
        next_pipe.passed = True

    # Track best bird performance
    if len(alive_birds) > 0:
        best_bird = max(alive_birds, key=lambda b: b.fitness)
        if best_bird.score > high_score:
            high_score = best_bird.score
    else:
        # ALL BIRDS DEAD: Evolve to Generation N+1
        generation += 1
        population.sort(key=lambda b: b.fitness, reverse=True)
        top_parents = population[:4]  # Select top 4 performers

        new_pop = []
        # Elitism: preserve top performer intact
        new_pop.append(Bird(brain=top_parents[0].brain))

        # Fill remaining population with mutated offspring
        while len(new_pop) < POPULATION_SIZE:
            parent = random.choice(top_parents)
            child_brain = parent.brain.mutate(rate=0.2, magnitude=0.35)
            new_pop.append(Bird(brain=child_brain))

        population = new_pop
        pipes = [Pipe(WIDTH + 100)]
        frame_count = 0
        continue

    # Render Scene
    SCREEN.fill((110, 190, 230))  # Sky blue

    for pipe in pipes:
        pipe.draw(SCREEN)

    # Draw regular birds first, best bird on top
    for bird in population:
        if bird != best_bird:
            bird.draw(SCREEN, is_best=False)
    if best_bird in alive_birds:
        best_bird.draw(SCREEN, is_best=True)

        # Draw decision ray to target gap
        gap_center = next_pipe.top_height + (PIPE_GAP / 2)
        pygame.draw.line(SCREEN, (255, 0, 0), (best_bird.x, best_bird.y), (next_pipe.x, gap_center), 1)

    # Ground
    pygame.draw.rect(SCREEN, (220, 200, 130), (0, HEIGHT - 20, WIDTH, 20))

    # HUD Stats Overlay
    stats = [
        f"Generation: {generation}",
        f"Alive: {len(alive_birds)} / {POPULATION_SIZE}",
        f"Current Score: {best_bird.score}",
        f"High Score: {high_score}"
    ]
    for i, line in enumerate(stats):
        txt = FONT.render(line, True, (0, 0, 0))
        SCREEN.blit(txt, (15, 15 + i * 22))

    pygame.display.flip()