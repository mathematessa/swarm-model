import numpy as np
import pygame
import csv
import random
import math
from pygame import gfxdraw

pygame.init()

WIDTH, HEIGHT = 800, 800
SCALE = 16
ROBOT_RADIUS = 5
NUM_ROBOTS = 20
MAX_SPEED = 1.2
BEHAVIOR_MODES = ['РОЙ', 'ЛИНИЯ']
CURRENT_MODE = 1

SEPARATION_RADIUS = 4.0
ALIGNMENT_RADIUS = 8.0
COHESION_RADIUS = 12.0
SEPARATION_FORCE = 0.6
COHESION_FORCE = 0.4
ALIGNMENT_FORCE = 0.5

LINE_SPACING = 2.0
LINE_FOLLOW_FORCE = 1.2
LINE_ALIGNMENT_FORCE = 0.8
MAX_LINE_DEVIATION = 2.0

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

TRAJECTORIES = [[] for _ in range(NUM_ROBOTS)]


class Robot:
    def __init__(self, x, y):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.formation_pos = np.array([x, y], dtype=float)

    def update(self, dt, mouse_pos=None):
        if mouse_pos is not None:
            direction = mouse_pos - self.position
            distance = np.linalg.norm(direction)
            if distance > 0:
                self.velocity = direction / distance * MAX_SPEED

        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = self.velocity / speed * MAX_SPEED

        self.position += self.velocity * dt
        self.position = np.clip(self.position, 0, 50)

        self.acceleration = np.zeros(2)


def calculate_swarm_forces(robots):
    for i, robot in enumerate(robots):
        separation = np.zeros(2)
        alignment = np.zeros(2)
        cohesion = np.zeros(2)
        neighbors_count = 0

        for j, other in enumerate(robots):
            if i == j: continue

            dist = np.linalg.norm(robot.position - other.position)

            if dist < SEPARATION_RADIUS and dist > 0:
                separation += (robot.position - other.position) / dist ** 2

            if dist < COHESION_RADIUS and dist > 0:
                alignment += other.velocity
                cohesion += other.position
                neighbors_count += 1

        if neighbors_count > 0:
            alignment = (alignment / neighbors_count - robot.velocity) * ALIGNMENT_FORCE
            cohesion = (cohesion / neighbors_count - robot.position) * COHESION_FORCE

        separation *= SEPARATION_FORCE

        robot.acceleration = separation + alignment + cohesion


def calculate_line_formation(robots):
    master = robots[0]

    if np.linalg.norm(master.velocity) > 0.1:
        line_direction = master.velocity / np.linalg.norm(master.velocity)
    else:
        line_direction = np.array([1.0, 0.0])

    for i in range(1, len(robots)):
        ideal_pos = robots[i - 1].position - line_direction * LINE_SPACING

        to_ideal = ideal_pos - robots[i].position
        dist_to_ideal = np.linalg.norm(to_ideal)
        if dist_to_ideal > 0:
            robots[i].acceleration += to_ideal / dist_to_ideal * LINE_FOLLOW_FORCE

        if i > 1:
            line_deviation = robots[i].position - ideal_pos
            dev_magnitude = np.linalg.norm(line_deviation)
            if dev_magnitude > MAX_LINE_DEVIATION:
                robots[i].acceleration -= line_deviation / dev_magnitude * LINE_ALIGNMENT_FORCE

        for j in [i - 1, i + 1]:
            if 0 <= j < len(robots) and j != i:
                dist = np.linalg.norm(robots[i].position - robots[j].position)
                if dist < SEPARATION_RADIUS and dist > 0:
                    avoidance = (robots[i].position - robots[j].position) / dist ** 2
                    robots[i].acceleration += avoidance * SEPARATION_FORCE


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Управление роем БПЛА - Линейная формация")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18)

    robots = []
    for i in range(NUM_ROBOTS):
        x = 25 - i * 0.5
        y = 25
        robots.append(Robot(x, y))
    master = robots[0]

    running = True
    paused = False
    dragging = False
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
                    mouse_pos = (mx / SCALE, my / SCALE)
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    robots = []
                    for i in range(NUM_ROBOTS):
                        x = 25 - i * 0.5
                        y = 25
                        robots.append(Robot(x, y))
                    master = robots[0]
                elif event.key == pygame.K_f:
                    global CURRENT_MODE
                    CURRENT_MODE = (CURRENT_MODE + 1) % len(BEHAVIOR_MODES)
                elif event.key == pygame.K_UP:
                    global LINE_SPACING
                    LINE_SPACING = min(LINE_SPACING + 0.5, 8.0)
                elif event.key == pygame.K_DOWN:
                    LINE_SPACING = max(LINE_SPACING - 0.5, 2.0)

        if not paused:
            master.update(dt, mouse_pos=mouse_pos if dragging else None)

            if CURRENT_MODE == 0:
                calculate_swarm_forces(robots)
            else:
                calculate_line_formation(robots)

            for robot in robots[1:]:
                robot.velocity += robot.acceleration * dt
                robot.update(dt)
                robot.acceleration = np.zeros(2)
            for i, robot in enumerate(robots):
                TRAJECTORIES[i].append(robot.position.copy())
            t += dt

        screen.fill(BLACK)

        pygame.draw.rect(screen, WHITE, (0, 0, 50 * SCALE, 50 * SCALE), 2)

        if CURRENT_MODE == 1:
            for i in range(len(robots) - 1):
                start = robots[i].position * SCALE
                end = robots[i + 1].position * SCALE
                # pygame.draw.line(screen, GREEN, start, end, 2)

        for i, robot in enumerate(robots):
            pos = robot.position * SCALE
            color = RED if i == 0 else BLUE
            pygame.draw.circle(screen, color, pos, ROBOT_RADIUS)
            pygame.draw.circle(screen, WHITE, pos, ROBOT_RADIUS, 1)

        info = [
            f"Режим: {BEHAVIOR_MODES[CURRENT_MODE]} (F)",
            f"Роботов: {NUM_ROBOTS}",
            f"Дистанция: {LINE_SPACING:.1f} м (▲/▼)" if CURRENT_MODE == 1 else "",
            f"Время: {t:.1f} с",
            "ЛКМ: управление, Пробел: пауза, R: сброс"
        ]
        y_offset = 10
        for text in info:
            if text:
                surf = font.render(text, True, WHITE, BLACK)
                screen.blit(surf, (10, y_offset))
                y_offset += 25

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()

with open('robot_trajectory.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for traj in TRAJECTORIES:
        row = []
        for pos in traj:
            row.extend([pos[0], pos[1]])
        writer.writerow(row)
print("Траектории роботов сохранены в robot_trajectory.csv")
