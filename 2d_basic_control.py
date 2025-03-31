import numpy as np
import pygame
import random
import math
import csv
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
MASTER_FORCE_SCALE = 0.5
MOUSE_CONTROL_FORCE = 0.8

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

    def update(self, dt, master=False, t=0, mouse_pos=None):
        if master and mouse_pos is not None:
            direction = mouse_pos - self.position
            distance = np.linalg.norm(direction)
            if distance > 0:
                force = MOUSE_CONTROL_FORCE * direction / distance
                self.acceleration += force

        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

        max_speed = 2.0
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed

        self.acceleration = np.array([0.0, 0.0])


def calculate_forces(robots):
    for i, robot in enumerate(robots):
        if i == MASTER_INDEX:
            robot.neighbors = []
            continue

        distances = []
        for j, other in enumerate(robots):
            if i != j:
                dist = np.linalg.norm(robot.position - other.position)
                distances.append((j, dist))
        distances.sort(key=lambda x: x[1])
        robot.neighbors = distances[:N_NEIGHBORS]

        total_force = np.array([0.0, 0.0])
        for neighbor_idx, dist in robot.neighbors:
            neighbor = robots[neighbor_idx]
            direction = neighbor.position - robot.position
            unit_vector = direction / np.linalg.norm(direction)
            force_magnitude = K1 * dist - K2 / (dist ** 2 + 1e-6)
            force = force_magnitude * unit_vector
            total_force += force

        robot.acceleration += total_force


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Управление роем БПЛА с помощью мыши")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    dragging_master = False
    mouse_influence = False

    robots = [Robot(random.uniform(10, 40), random.uniform(10, 40)) for _ in range(NUM_ROBOTS)]
    trajectories = [[] for _ in range(NUM_ROBOTS)]

    running = True
    paused = False
    t = 0
    dt = 0.1

    while running:
        mouse_sim_pos = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                sim_x = (mx - 100) / SCALE
                sim_y = (my - 100) / SCALE
                master_pos = robots[MASTER_INDEX].position
                distance = math.hypot(sim_x - master_pos[0], sim_y - master_pos[1])
                if distance < ROBOT_RADIUS * 2:
                    dragging_master = True
                    mouse_influence = True
            elif event.type == pygame.MOUSEMOTION:
                if dragging_master:
                    mouse_pos = pygame.mouse.get_pos()
                    sim_x = (mouse_pos[0] - 100) / SCALE
                    sim_y = (mouse_pos[1] - 100) / SCALE
                    robots[MASTER_INDEX].position = np.array([sim_x, sim_y])
                    mouse_sim_pos = np.array([sim_x, sim_y])
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_master = False
                mouse_influence = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    robots = []
                    for _ in range(NUM_ROBOTS):
                        x = random.uniform(10, 40)
                        y = random.uniform(10, 40)
                        robots.append(Robot(x, y))
                    t = 0
                    trajectories = [[] for _ in range(NUM_ROBOTS)]
                elif event.key == pygame.K_UP:
                    global N_NEIGHBORS
                    N_NEIGHBORS = min(N_NEIGHBORS + 1, 10)
                elif event.key == pygame.K_DOWN:
                    N_NEIGHBORS = max(N_NEIGHBORS - 1, 1)

        if not paused:
            calculate_forces(robots)
            current_mouse = pygame.mouse.get_pos()
            if mouse_influence:
                sim_mouse = ((current_mouse[0] - 100) / SCALE,
                             (current_mouse[1] - 100) / SCALE)
            else:
                sim_mouse = None

            for i, robot in enumerate(robots):
                robot.update(dt, master=(i == MASTER_INDEX),
                             t=t, mouse_pos=sim_mouse if (i == MASTER_INDEX and mouse_influence) else None)

            t += dt

            for i, robot in enumerate(robots):
                trajectories[i].append(robot.position.copy())

        screen.fill(BLACK)
        pygame.draw.rect(screen, WHITE, (100, 100, 50 * SCALE, 50 * SCALE), 1)

        if mouse_influence:
            mx, my = pygame.mouse.get_pos()
            pygame.draw.circle(screen, YELLOW, (mx, my), 5, 1)

        for i, robot in enumerate(robots):
            x = int(100 + robot.position[0] * SCALE)
            y = int(100 + robot.position[1] * SCALE)

            if i == MASTER_INDEX:
                pygame.draw.circle(screen, RED, (x, y), ROBOT_RADIUS + 2)
            else:
                pygame.draw.circle(screen, BLUE, (x, y), ROBOT_RADIUS)

        for neighbor_idx, dist in robots[-1].neighbors:
            neighbor = robots[neighbor_idx]
            nx = int(100 + neighbor.position[0] * SCALE)
            ny = int(100 + neighbor.position[1] * SCALE)
            pygame.draw.line(screen, GREEN, (x, y), (nx, ny), 1)

        # Отображение информации
        info_text = [
            f"Время: {t:.1f} с",
            f"Количество роботов: {NUM_ROBOTS}",
            f"Ближайших соседей: {N_NEIGHBORS} (Вверх/Вниз для изменения)",
            f"Параметры: K1={K1}, K2={K2}",
            "Пробел: пауза, R: сброс"
        ]

        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, WHITE)
            screen.blit(text_surface, (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    with open('trajectory.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for robot_traj in trajectories:
            row = []
            for pos in robot_traj:
                row.extend([pos[0], pos[1]])
            writer.writerow(row)
    print("Траектория сохранена в trajectory.csv")


if __name__ == "__main__":
    main()
