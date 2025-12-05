# ant_colony_scheduler.py
import random
import math
from bisect import bisect_left

def ant_colony_schedule(
    observations,
    slew_matrix,
    num_ants=40,
    alpha=1.0,         # pheromone influence
    beta=2.0,          # heuristic influence
    evap=0.3,
    iterations=40,
    top_k=50           # number of candidate successors per node (sparsity)
):
    """
    Optimized Ant Colony for interval scheduling using sparse successor lists.
    observations: list of Observation objects (assumed sorted arbitrarily)
    slew_matrix: 2D list-like of slew times (seconds)
    Returns a schedule list of Observation objects.
    """

    n = len(observations)
    if n == 0:
        return []

    # 1) Build index lists sorted by start time to allow fast successor search
    starts = [obs.start for obs in observations]
    ends   = [obs.end for obs in observations]
    order_by_start = sorted(range(n), key=lambda i: starts[i])
    order_by_end   = sorted(range(n), key=lambda i: ends[i])

    # 2) For each job i, precompute feasible successors: indices j with start_j >= end_i
    # We'll keep only top_k successors per i by heuristic (weight / slew)
    successors = [[] for _ in range(n)]
    for i in range(n):
        # find first index in order_by_start with start >= ends[i]
        pos = bisect_left([starts[k] for k in order_by_start], ends[i])
        # collect candidates starting from pos
        cand = []
        # iterate through order_by_start from pos, but limit scanning to avoid O(n^2) worst-case
        # we will scan up to top_k * 5 candidates to pick best top_k by heuristic
        scan_limit = min(len(order_by_start), pos + top_k * 5)
        for idx in order_by_start[pos:scan_limit]:
            # heuristic: weight / max(slew, 1)
            dist = max(slew_matrix[i][idx], 1)
            h = observations[idx].weight / dist
            cand.append((h, idx))
        # pick top_k by heuristic
        cand.sort(reverse=True)
        if len(cand) > top_k:
            cand = cand[:top_k]
        successors[i] = [idx for (_, idx) in cand]

    # 3) Sparse pheromone structure: list of dicts
    pher = [dict() for _ in range(n)]
    # initialize with small positive pheromone
    init_pher = 1.0
    for i in range(n):
        for j in successors[i]:
            pher[i][j] = init_pher

    # 4) Precompute heuristic attractiveness for candidate edges
    heuristic = [dict() for _ in range(n)]
    for i in range(n):
        for j in successors[i]:
            dist = max(slew_matrix[i][j], 1)
            heuristic[i][j] = observations[j].weight / dist

    # helper: build tour for one ant
    def build_ant_tour():
        selected = []
        used = [False] * n

        # start from a random feasible seed (choose among all nodes)
        start_idx = random.randrange(n)
        selected.append(start_idx)
        used[start_idx] = True

        # then greedily/probabilistically choose next among successors of last
        while True:
            last = selected[-1]
            cand = []
            for j in successors[last]:
                if used[j]:
                    continue
                # ensure time feasibility relative to last chosen
                if observations[j].start >= observations[last].end:
                    # compute pheromone * heuristic score
                    tau = pher[last].get(j, 0.0) ** alpha
                    eta = heuristic[last].get(j, 0.0) ** beta
                    if tau <= 0 and eta <= 0:
                        continue
                    cand.append((tau * eta, j))
            if not cand:
                break
            # normalize probabilities
            total = sum(score for score, _ in cand)
            if total <= 0:
                # fallback: choose highest score
                cand.sort(reverse=True)
                chosen = cand[0][1]
            else:
                r = random.random() * total
                acc = 0.0
                chosen = cand[-1][1]
                for score, j in cand:
                    acc += score
                    if acc >= r:
                        chosen = j
                        break
            selected.append(chosen)
            used[chosen] = True

        return selected

    best_schedule = []
    best_score = 0.0

    # main ACO loop
    for it in range(iterations):
        # produce ant tours
        local_best_score = 0.0
        local_best = None
        for _ in range(num_ants):
            tour = build_ant_tour()
            score = sum(observations[i].weight for i in tour)
            if score > local_best_score:
                local_best_score = score
                local_best = tour
            if score > best_score:
                best_score = score
                best_schedule = tour[:]

        # pheromone evaporation
        for i in range(n):
            row = pher[i]
            if not row:
                continue
            # evaporate in-place; also remove tiny entries to keep dicts small
            remove = []
            for j in list(row.keys()):
                row[j] *= (1 - evap)
                if row[j] < 1e-8:
                    remove.append(j)
            for j in remove:
                del row[j]

        # pheromone deposit from local_best (or global best)
        if local_best:
            deposit = local_best_score / (1.0 + len(local_best))
            # add pheromone along consecutive pairs in local_best
            for k in range(len(local_best) - 1):
                a = local_best[k]
                b = local_best[k + 1]
                if b in pher[a]:
                    pher[a][b] += deposit
                else:
                    # only add if b was a candidate successor; skip otherwise
                    if b in successors[a]:
                        pher[a][b] = init_pher + deposit

    # return best_schedule as Observation objects, sorted by start
    if not best_schedule:
        return []
    best_schedule_obs = [observations[i] for i in best_schedule]
    best_schedule_obs.sort(key=lambda o: o.start)
    return best_schedule_obs