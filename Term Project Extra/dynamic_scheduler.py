import bisect
from array import array

def dynamic_weighted_interval_scheduling(obs_list):
    """
    Optimized Weighted Interval Scheduling for lists of Observation objects.

    Expects each Observation to have:
      - start: float (seconds since epoch)
      - end:   float (seconds since epoch)
      - duration: float (seconds)
      - weight: float (base weight)

    Returns:
        list of Observation objects in optimal non-overlapping schedule.
    """

    n = len(obs_list)
    if n == 0:
        return []

    # -------------------------------------------
    # 1. Extract primitive arrays for speed
    # -------------------------------------------
    starts = array('d', (o.start for o in obs_list))
    ends   = array('d', (o.end   for o in obs_list))
    weights = array('d', (o.weight * (max(o.duration, 1.0) ** 0.25) for o in obs_list))

    # -------------------------------------------
    # 2. Sort by end time
    # -------------------------------------------
    indices = list(range(n))
    indices.sort(key=lambda i: ends[i])

    ends_s    = [ends[i] for i in indices]
    starts_s  = [starts[i] for i in indices]
    weights_s = [weights[i] for i in indices]

    # -------------------------------------------
    # 3. Compute p(i) using bisect
    # -------------------------------------------
    bisect_right = bisect.bisect_right
    p = [-1] * n
    for i in range(n):
        j = bisect_right(ends_s, starts_s[i]) - 1
        p[i] = j

    # -------------------------------------------
    # 4. DP table M
    # -------------------------------------------
    M = [0.0] * (n + 1)

    for i in range(1, n + 1):
        incl = weights_s[i - 1]
        j = p[i - 1]
        if j >= 0:
            incl += M[j + 1]

        excl = M[i - 1]
        M[i] = incl if incl > excl else excl

    # -------------------------------------------
    # 5. Reconstruct optimal schedule
    # -------------------------------------------
    chosen_indices = []
    i = n
    while i > 0:
        incl = weights_s[i - 1]
        j = p[i - 1]
        if j >= 0:
            incl += M[j + 1]

        if incl >= M[i - 1]:
            chosen_indices.append(indices[i - 1])  # store original index
            i = j + 1
        else:
            i -= 1

    chosen_indices.reverse()

    # Return original Observation objects in optimal schedule
    return [obs_list[idx] for idx in chosen_indices]