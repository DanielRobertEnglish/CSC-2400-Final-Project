from models import angular_distance_deg

def greedy_schedule(observations, slew_matrix=None):
    """
    Greedy by descending weight.
    No advanced heuristics, no multi-factor score.
    """
    obs_sorted = sorted(observations, key=lambda x: x.weight, reverse=True)

    schedule = []
    current_end = None

    for obs in obs_sorted:
        if current_end is None or obs.start >= current_end:
            schedule.append(obs)
            current_end = obs.end

    return schedule