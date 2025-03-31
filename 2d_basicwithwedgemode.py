import numpy as np
import pygame
import csv
import math

pygame.init()

WIDTH, HEIGHT = 800, 800
SCALE = 16
ROBOT_RADIUS = 5
NUM_ROBOTS = 20
MAX_SPEED = 1.2
BEHAVIOR_MODES = ['РОЙ', 'КЛИН']
CURRENT_MODE = 1

SEPARATION_RADIUS = 4.0
ALIGNMENT_RADIUS = 8.0
COHESION_RADIUS = 12.0
SEPARATION_FORCE = 0.6
COHESION_FORCE = 0.4
ALIGNMENT_FORCE = 0.5

LINE_SPACING = 2.0
LINE_ANGLE = 30
LINE_FOLLOW_FORCE = 1.2

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

TRAJECTORIES = [[] for _ in range(NUM_ROBOTS)]


class Robot:
    def __init__(self, x, y, line_id=0, line_order=0):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.line_id = line_id
        self.line_order = line_order
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
            if i == j:
                continue

            dist = np.linalg.norm(robot.position - other.position)

            if dist < SEPARATION_RADIUS and dist > 0:
                separation += (robot.position - other.position) / (dist ** 2)

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
        forward = master.velocity / np.linalg.norm(master.velocity)
    else:
        forward = np.array([1.0, 0.0])
    backward = -forward
    angle_rad = math.radians(LINE_ANGLE)
    left_dir = np.array([
        backward[0] * math.cos(angle_rad) - backward[1] * math.sin(angle_rad),
        backward[0] * math.sin(angle_rad) + backward[1] * math.cos(angle_rad)
    ])
    right_dir = np.array([
        backward[0] * math.cos(-angle_rad) - backward[1] * math.sin(-angle_rad),
        backward[0] * math.sin(-angle_rad) + backward[1] * math.cos(-angle_rad)
    ])

    left_line = sorted([r for r in robots if r.line_id == 0 and r != master], key=lambda r: r.line_order)
    right_line = sorted([r for r in robots if r.line_id == 1], key=lambda r: r.line_order)

    for robot in left_line:
        robot.formation_pos = master.position + left_dir * LINE_SPACING * (robot.line_order + 1)
    for robot in right_line:
        robot.formation_pos = master.position + right_dir * LINE_SPACING * (robot.line_order + 1)

    for r in left_line + right_line:
        to_ideal = r.formation_pos - r.position
        dist = np.linalg.norm(to_ideal)
        if dist > 0:
            r.acceleration += to_ideal / dist * LINE_FOLLOW_FORCE


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Рой и клиновая формация")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18)
    global LINE_SPACING

    robots = []

    robots.append(Robot(25, 25, line_id=0, line_order=0))
    for i in range(1, NUM_ROBOTS):
        line_id = 0 if i % 2 == 0 else 1
        line_order = i // 2
        x = 25 + (line_id * 2 - 1) * LINE_SPACING
        y = 25
        robots.append(Robot(x, y, line_id, line_order))
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
                    robots.append(Robot(25, 25, line_id=0, line_order=0))
                    for i in range(1, NUM_ROBOTS):
                        line_id = 0 if i % 2 == 0 else 1
                        line_order = i // 2
                        x = 25 + (line_id * 2 - 1) * LINE_SPACING
                        y = 25
                        robots.append(Robot(x, y, line_id, line_order))
                    master = robots[0]
                elif event.key == pygame.K_f:
                    global CURRENT_MODE
                    CURRENT_MODE = (CURRENT_MODE + 1) % len(BEHAVIOR_MODES)
                elif event.key == pygame.K_UP:
                    LINE_SPACING = min(LINE_SPACING + 0.5, 8.0)
                elif event.key == pygame.K_DOWN:
                    LINE_SPACING = max(LINE_SPACING - 0.5, 1.0)

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

        for i, robot in enumerate(robots):
            pos = robot.position * SCALE
            if i == 0:
                color = RED
            else:
                color = BLUE if robot.line_id == 0 else BLUE
            pygame.draw.circle(screen, color, pos, ROBOT_RADIUS)
            pygame.draw.circle(screen, BLACK, pos, ROBOT_RADIUS, 1)

        info = [
            f"Режим: {BEHAVIOR_MODES[CURRENT_MODE]} (F)",
            f"Роботов: {NUM_ROBOTS}",
            f"Дистанция: {LINE_SPACING:.1f} (▲/▼)",
            f"Угол: {LINE_ANGLE}° (между линиями 60°)",
            f"Время: {t:.1f} с",
            "ЛКМ: управление, Пробел: пауза, R: сброс"
        ]
        y_offset = 10
        for text in info:
            surf = font.render(text, True, WHITE)
            screen.blit(surf, (10, y_offset))
            y_offset += 25

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    with open('robot_trajectory.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for traj in TRAJECTORIES:
            row = []
            for pos in traj:
                row.extend([pos[0], pos[1]])
            writer.writerow(row)
    print("Траектории сохранены в robot_trajectory.csv")


if __name__ == "__main__":
    main()
