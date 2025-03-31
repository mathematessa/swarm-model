import numpy as np
import csv

K1, K2 = 2, 20
M = 100
N = 5
dt = 1
Vmax = 5 

np.random.seed(0)
positions = np.random.uniform(0, 50, (M, 2))
velocities = np.zeros((M, 2))


def master_accel(t):
    """Ускорение мастер-бота (по эллипсу)"""
    return np.array([1.0 * np.sin(np.pi * t / 100), 1.0 * np.cos(np.pi * t / 100)])


def get_nearest_neighbors(k, positions, n=5):
    """Находит n ближайших соседей для робота k"""
    dists = np.linalg.norm(positions - positions[k], axis=1)
    nearest = np.argsort(dists)[1:n + 1]
    return nearest


def calculate_forces(positions, velocities, master_accel, t):
    """Обновление позиций и скоростей роботов"""
    new_positions = np.copy(positions)
    new_velocities = np.copy(velocities)

    new_velocities[0] += master_accel(t) * dt
    new_velocities[0] = np.clip(new_velocities[0], -Vmax, Vmax)
    new_positions[0] += new_velocities[0] * dt

    for i in range(1, M):
        total_force = np.array([0.0, 0.0])
        neighbors = get_nearest_neighbors(i, positions)
        for j in neighbors:
            diff = positions[j] - positions[i]
            distance = np.linalg.norm(diff)
            distance = max(distance, 0.1)
            unit_vector = diff / distance

            attraction = K1 * distance * unit_vector
            repulsion = -K2 / (distance ** 2) * unit_vector

            total_force += attraction + repulsion

        new_velocities[i] += total_force * dt
        new_velocities[i] = np.clip(new_velocities[i], -Vmax, Vmax)
        new_positions[i] += new_velocities[i] * dt

    return np.round(new_positions * 2) / 2, new_velocities


def simulate_swarm(N):
    """Запускаем симуляцию на N секунд"""
    trajectory = []
    global positions, velocities
    for t in range(N):
        positions, velocities = calculate_forces(positions, velocities, master_accel, t)
        trajectory.append(positions.copy())
    return trajectory


trajectory = simulate_swarm(N)

with open('trajectory.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for robot_index in range(M):
        row = []
        for pos in trajectory:
            x, y = pos[robot_index]
            row.extend([x, y])
        writer.writerow(row)
