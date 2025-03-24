import numpy as np
import pygame
import random
import math
from pygame import gfxdraw

pygame.init()

WIDTH, HEIGHT = 800, 800
SCALE = 8
ROBOT_RADIUS = 3
NUM_ROBOTS = 100
K1 = 2
K2 = 20
N_NEIGHBORS = 5
MASTER_INDEX = 0
FORMATION_FORCE = 1.2
FORMATION_ANGLE = 60
ROW_SPACING = 4.0
WING_SPACING = 3.0
MASTER_SPEED = 2.5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)


class Robot:
    def __init__(self, x, y):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.neighbors = []
        self.formation_pos = np.array([x, y], dtype=float)

    def update(self, dt, mouse_pos=None):
        if mouse_pos is not None:
            direction = mouse_pos - self.position
            distance = np.linalg.norm(direction)
            if distance > 0:
                self.velocity = direction / distance * MASTER_SPEED
            else:
                self.velocity = np.array([0.0, 0.0])

        self.position += self.velocity * dt
        self.position[0] = np.clip(self.position[0], 10, 40)
        self.position[1] = np.clip(self.position[1], 10, 40)


def calculate_formation_positions(master_pos, master_velocity, num_robots):
    formation = []
    half_angle = math.radians(FORMATION_ANGLE / 2)
    if np.linalg.norm(master_velocity) > 0:
        forward = master_velocity / np.linalg.norm(master_velocity)
    else:
        forward = np.array([1.0, 0.0])
        
    left_wing = np.array([
        forward[0] * math.cos(half_angle) - forward[1] * math.sin(half_angle),
        forward[0] * math.sin(half_angle) + forward[1] * math.cos(half_angle)
    ])

    right_wing = np.array([
        forward[0] * math.cos(-half_angle) - forward[1] * math.sin(-half_angle),
        forward[0] * math.sin(-half_angle) + forward[1] * math.cos(-half_angle)
    ])

    row = 1
    positions_generated = 0
    while positions_generated < num_robots - 1:
        for wing in [left_wing, right_wing]:
            back_offset = forward * row * ROW_SPACING
            wing_offset = wing * (row - 1) * WING_SPACING

            pos = master_pos - back_offset + wing_offset
            formation.append(pos)
            positions_generated += 1

            if positions_generated >= num_robots - 1:
                break

        row += 1

    return formation


def calculate_forces(robots, formation_mode):
    master = robots[MASTER_INDEX]

    if formation_mode:
        formation_positions = calculate_formation_positions(
            master.position,
            master.velocity,
            len(robots))

        for i, robot in enumerate(robots):
            if i == MASTER_INDEX:
                continue
            if (i - 1) < len(formation_positions):
                robot.formation_pos = formation_positions[i - 1]

    for i, robot in enumerate(robots):
        if i == MASTER_INDEX:
            continue

        total_force = np.array([0.0, 0.0])

        if formation_mode and not np.isnan(robot.formation_pos).any():
            direction = robot.formation_pos - robot.position
            distance = np.linalg.norm(direction)
            if distance > 0.5:
                total_force += FORMATION_FORCE * direction / distance * min(distance, 5.0)

        distances = []
        for j, other in enumerate(robots):
            if i != j:
                dist = np.linalg.norm(robot.position - other.position)
                distances.append((j, dist))

        distances.sort(key=lambda x: x[1])
        robot.neighbors = distances[:N_NEIGHBORS]

        for neighbor_idx, dist in robot.neighbors:
            neighbor = robots[neighbor_idx]
            direction = neighbor.position - robot.position
            unit_vector = direction / (np.linalg.norm(direction) + 1e-6)

            force = (K1 * dist - K2 / (dist ** 2 + 1e-6)) * unit_vector
            total_force += force

        robot.acceleration = total_force


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("V-образный клин БПЛА")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    robots = [Robot(random.uniform(10, 40), random.uniform(10, 40)) for _ in range(NUM_ROBOTS)]
    master = robots[MASTER_INDEX]

    running = True
    paused = False
    dragging = False
    formation_mode = False
    t = 0
    dt = 0.1
    
    while running:
        mouse_pos = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                dragging = True
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = pygame.mouse.get_pos()
                    mouse_pos = ((mx - 100) / SCALE, (my - 100) / SCALE)
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    robots = [Robot(random.uniform(10, 40), random.uniform(10, 40)) for _ in range(NUM_ROBOTS)]
                    master = robots[MASTER_INDEX]
                elif event.key == pygame.K_f:
                    formation_mode = not formation_mode
                elif event.key == pygame.K_UP:
                    global FORMATION_ANGLE
                    FORMATION_ANGLE = min(FORMATION_ANGLE + 5, 120)
                elif event.key == pygame.K_DOWN:
                    FORMATION_ANGLE = max(FORMATION_ANGLE - 5, 30)

        if not paused:
            master.update(dt, mouse_pos=mouse_pos if dragging else None)

            calculate_forces(robots, formation_mode)

            for i, robot in enumerate(robots):
                if i != MASTER_INDEX:
                    robot.velocity += robot.acceleration * dt
                    robot.position += robot.velocity * dt
                    speed = np.linalg.norm(robot.velocity)
                    if speed > 2.0:
                        robot.velocity = robot.velocity / speed * 2.0

        screen.fill(BLACK)
        pygame.draw.rect(screen, WHITE, (100, 100, 50 * SCALE, 50 * SCALE), 1)

        for i, robot in enumerate(robots):
            x = int(100 + robot.position[0] * SCALE)
            y = int(100 + robot.position[1] * SCALE)

            color = RED if i == MASTER_INDEX else BLUE
            pygame.draw.circle(screen, color, (x, y), ROBOT_RADIUS + (2 if i == MASTER_INDEX else 0))

            if formation_mode and i != MASTER_INDEX and not np.isnan(robot.formation_pos).any():
                tx = int(100 + robot.formation_pos[0] * SCALE)
                ty = int(100 + robot.formation_pos[1] * SCALE)

        info = [
            f"Режим: {'V-КЛИН' if formation_mode else 'РОЙ'} (F)",
            f"Угол: {FORMATION_ANGLE}° (Вверх/Вниз)",
            f"Роботов: {NUM_ROBOTS}",
            f"Время: {t:.1f} с",
            "ЛКМ: тянуть, Пробел: пауза, R: сброс"
        ]
        for i, text in enumerate(info):
            surf = font.render(text, True, WHITE)
            screen.blit(surf, (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(60)
        t += dt

    pygame.quit()


if __name__ == "__main__":
    main()
