import pygame

OUTER_TRACK = [
    (100, 100), (900, 100), (950, 200), (950, 550),
    (900, 620), (500, 620), (450, 450), (350, 450),
    (300, 620), (100, 620), (50, 500), (50, 200)
]

INNER_TRACK = [
    (220, 220), (780, 220), (830, 270), (830, 470),
    (780, 500), (580, 500), (530, 330), (270, 330),
    (220, 500), (220, 500), (170, 450), (170, 270)
]

def draw_track(surface):
    surface.fill((30, 30, 35))
    # Draw track fill
    pygame.draw.polygon(surface, (60, 60, 65), OUTER_TRACK)
    pygame.draw.polygon(surface, (30, 30, 35), INNER_TRACK)
    # Draw walls
    pygame.draw.lines(surface, (255, 255, 255), True, OUTER_TRACK, 4)
    pygame.draw.lines(surface, (255, 255, 255), True, INNER_TRACK, 4)
    # Finish line
    pygame.draw.line(surface, (0, 255, 0), (100, 100), (220, 220), 4)

# Line-intersection math for collision and raycasts
def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))
    return None

def get_track_segments():
    segments = []
    for track in [OUTER_TRACK, INNER_TRACK]:
        for i in range(len(track)):
            p1 = track[i]
            p2 = track[(i + 1) % len(track)]
            segments.append((p1, p2))
    return segments

TRACK_SEGMENTS = get_track_segments()