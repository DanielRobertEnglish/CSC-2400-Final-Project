#"""
#------------------------------------------------------------
#Project: Telescopic Scheduling
#Course: CSC 2400-002
#Date: 11/11/2025
#Team Members: Remek Botwright, Soren Richards, Danny English, Luke Sapp
#Professor: Cristina Radian
#------------------------------------------------------------
#
#Overview:
#This project explores telescope scheduling — a process used to efficiently
#allocate limited resources (like telescope time) to high-priority tasks.
#We’ll implement and compare three algorithms: Ant Colony, Greedy, and Dynamic
#Scheduling. The goal is to evaluate efficiency, adaptability, and performance
#across each method.
#
#------------------------------------------------------------
#"""

# ------------------------------------
# Ant Colony Optimization
# ------------------------------------

from Shared_Components import *
import random
import math

SAFE_EPSILON = 0.000001

class AntColony:
    def __init__(self, tasks, n_ants=8, n_iterations=18,
                 decay=0.1, alpha=1, beta=2):

        self.tasks = tasks
        self.n = len(tasks)

        self.distances = self._build_distance_matrix()
        self.pheromone = [[1.0 for _ in range(self.n)] for _ in range(self.n)]

        self.n_ants = n_ants
        self.n_iterations = n_iterations

        self.decay = decay      # pheromone evaporation
        self.alpha = alpha      # pheromone importance
        self.beta = beta        # heuristic importance

    # -----------------------------------------------------------
    # Distance Matrix
    # - overlap gives huge penalty
    # - time gap is small penalty
    # - NO negative distances (bad for heuristic)
    #
    # interpretation: LOWER = better transition
    # -----------------------------------------------------------
    def _build_distance_matrix(self):
        n = len(self.tasks)
        M = [[0]*n for _ in range(n)]

        for i in range(n):
            A = self.tasks[i]

            for j in range(n):
                if i == j:
                    M[i][j] = float("inf")
                    continue

                B = self.tasks[j]

                # Overlap penalty (huge)
                overlap = max(0, min(A.end, B.end) - max(A.start, B.start))
                if overlap > 0:
                    M[i][j] = 1e8  # don't allow transitions into overlaps
                    continue

                # Legit transition → small time gap penalty
                gap = max(0, B.start - A.end)

                # Weight helps reduce distance (slightly)
                avg_weight = (A.priority + B.priority) / 2

                dist = gap + (1 / (avg_weight + 1))

                if dist <= 0:
                    dist = SAFE_EPSILON

                M[i][j] = dist

        return M

    # -----------------------------------------------------------
    # Run ACO
    # -----------------------------------------------------------
    def run(self):
        best_weight = -1
        best_path = None

        for iteration in range(self.n_iterations):
            paths = self._construct_all_paths()
            self._evaporate_pheromones()
            self._reinforce_pheromones(paths)

            for path_weight, path in paths:
                if path_weight > best_weight:
                    best_weight = path_weight
                    best_path = path

        final_schedule = self._build_final_schedule(best_path)
        return final_schedule, best_weight

    # -----------------------------------------------------------
    # Build paths for all ants
    # -----------------------------------------------------------
    def _construct_all_paths(self):
        results = []
        for _ in range(self.n_ants):
            path = self._generate_path()
            weight = sum(self.tasks[i].priority for i in path)
            results.append((weight, path))
        return results

    # -----------------------------------------------------------
    # Generate one ant's path
    # -----------------------------------------------------------
    def _generate_path(self):
        start = random.randint(0, self.n - 1)
        used = {start}
        path = [start]

        while True:
            nxt = self._select_next(path[-1], used)
            if nxt is None:
                break
            used.add(nxt)
            path.append(nxt)

        return path

    # -----------------------------------------------------------
    # Select next job via probability:
    #   (pheromone^α) * (heuristic^β)
    #
    # heuristic = (priority / (distance + duration))
    #
    # -----------------------------------------------------------
    def _select_next(self, current, used):
        candidates = []

        for j in range(self.n):
            if j in used:
                continue

            dist = self.distances[current][j]
            if dist >= 1e7:   # impossible overlap transition
                continue

            B = self.tasks[j]
            duration = max(B.end - B.start, SAFE_EPSILON)

            pher = self.pheromone[current][j] ** self.alpha
            heuristic = (B.priority / duration) / dist

            if heuristic <= 0:
                heuristic = SAFE_EPSILON

            score = pher * (heuristic ** self.beta)
            candidates.append((j, score))

        if not candidates:
            return None

        # Roulette Wheel Selection
        total = sum(s for _, s in candidates)
        r = random.random() * total
        running = 0

        for j, score in candidates:
            running += score
            if running >= r:
                return j

        return candidates[-1][0]

    # -----------------------------------------------------------
    # Pheromone Update
    # Ants with HIGH TOTAL WEIGHT deposit more pheromone
    # -----------------------------------------------------------
    def _reinforce_pheromones(self, paths):
        for weight, path in paths:
            if weight <= 0:
                continue

            deposit = weight ** 1.3  # reward exponential growth

            for i in range(len(path) - 1):
                a = path[i]
                b = path[i+1]
                self.pheromone[a][b] += deposit

    # -----------------------------------------------------------
    def _evaporate_pheromones(self):
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - self.decay)
                if self.pheromone[i][j] < SAFE_EPSILON:
                    self.pheromone[i][j] = SAFE_EPSILON

    # -----------------------------------------------------------
    # Build final non-overlapping schedule from the best path
    # -----------------------------------------------------------
    def _build_final_schedule(self, path):
        tasks_sorted = sorted([self.tasks[i] for i in path],
                              key=lambda t: t.start)

        schedule = []
        current_end = -1

        for t in tasks_sorted:
            if t.start >= current_end:
                schedule.append(t)
                current_end = t.end

        return schedule


def run_ant_colony(tasks):
    print("\n🐜 Running Improved Ant Colony Optimization...")
    colony = AntColony(tasks)
    schedule, score = colony.run()
    print("ACO Schedule Weight:", score)
    print("ACO Count:", len(schedule))
    return schedule
