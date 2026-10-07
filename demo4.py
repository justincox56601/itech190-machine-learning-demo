import sys
import random
import numpy as np
import pygame

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 500, 700
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Class 3: Flappy Bird AI (Neural Nets + Evolution)")
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("Consolas", 16)

# Game Constants
GRAVITY = 0.4
JUMP_STRENGTH = -7.0
PIPE_SPEED = 3.5
PIPE_GAP = 160
PIPE_FREQUENCY = 90  # frames between pipes

# =====================================================================
# STEP 1: Neural Network Architecture
# =====================================================================
class BirdBrain:
    def __init__(self, input_size=3, hidden_size=6, output_size=1):
        # Small weight initialization so initial actions aren't maxed out
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def forward(self, inputs):
        # inputs shape: (1, 3) -> [vel, dist_x, dist_y]
        z1 = np.dot(inputs, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        # Sigmoid activation for binary jump decision (0.0 to 1.0)
        output = 1.0 / (1.0 + np.exp(-z2))
        return output[0][0]

    def mutate(self, rate=0.15, magnitude=0.3):
        child = BirdBrain()
        child.W1 = self.W1.copy()
        child.b1 = self.b1.copy()
        child.W2 = self.W2.copy()
        child.b2 = self.b2.copy()

        for w in [child.W1, child.b1, child.W2, child.b2]:
            mask = np.random.rand(*w.shape) < rate
            w += mask * np.random.randn(*w.shape) * magnitude
        return child

# =====================================================================
# STEP 2: Bird & Pipe Entities
# =====================================================================
class Bird:
    def __init__(self, brain=None):
        self.x = 80
        self.y = HEIGHT // 2
        self.velocity = 0
        self.alive = True
        self.fitness = 0
        self.score = 0
        self.brain = brain if brain else BirdBrain()

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