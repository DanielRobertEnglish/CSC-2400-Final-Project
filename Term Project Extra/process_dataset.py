import csv
import time
import math
import random
from array import array
from datetime import datetime
from models import Observation, parse_time, angular_distance_deg
from greedy_scheduler import greedy_schedule
from dynamic_scheduler import dynamic_weighted_interval_scheduling
from ant_colony_scheduler import ant_colony_schedule

INPUT_CSV = "HST_2025-12-04T22_48_27-06_00.csv"

OUT_GREEDY = "schedule_greedy.csv"
OUT_DYNAMIC = "schedule_dynamic.csv"
OUT_ANT = "schedule_ant_colony.csv"

MAX_ITEMS = 1000           # sample 1000 to keep ACO reasonable
SLEW_RATE_DEG_PER_SEC = 0.1

def to_timestamp(t):
    """Accept either datetime or ISO string; return float seconds since epoch."""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, datetime):
        return t.timestamp()
    # otherwise treat as ISO string
    try:
        # Use fromisoformat (fast) — strip Z if present
        s = t.replace("Z", "") if isinstance(t, str) else str(t)
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        # fallback parse attempt
        return float(t)

def load_observations(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                # parse times (as floats)
                start_raw = r.get('sci_start_time') or r.get('start_time') or ""
                stop_raw  = r.get('sci_stop_time')  or r.get('stop_time')  or ""
                if not start_raw or not stop_raw:
                    continue
                # parse_time from models returns datetime; convert to timestamp float
                start_dt = parse_time(start_raw)
                stop_dt  = parse_time(stop_raw)
                if start_dt is None or stop_dt is None:
                    continue
                start_ts = start_dt.timestamp()
                stop_ts  = stop_dt.timestamp()
                if stop_ts <= start_ts:
                    # skip invalid intervals
                    continue

                ra = float(r.get('sci_ra') or r.get('ra') or 0.0)
                dec = float(r.get('sci_dec') or r.get('dec') or 0.0)

                pep_field = r.get('sci_pep_id') or r.get('sci_proposal_id') or ""
                try:
                    pep = float(pep_field) if pep_field.strip() else 99999.0
                except:
                    pep = 99999.0

                # Greedy weight = simple single-field: 1 / proposal_id
                simple_weight = 1.0 / max(1.0, pep)

                # duration in seconds (float) for cheap arithmetic
                duration_sec = float(stop_ts - start_ts)

                obs = Observation(
                    oid=r.get('sci_data_set_name') or r.get('dataset') or "",
                    start=start_ts,
                    end=stop_ts,
                    duration=duration_sec,   # float seconds
                    ra=ra,
                    dec=dec,
                    weight=simple_weight
                )
                rows.append(obs)
            except Exception:
                # skip malformed rows quietly
                continue

    print(f"Loaded {len(rows)} observations.")
    return rows

def sample_observations(rows):
    if len(rows) <= MAX_ITEMS:
        return rows
    return random.sample(rows, MAX_ITEMS)

def compute_slew_matrix(obs):
    n = len(obs)
    # small optimization: use list comprehensions and local references
    mat = [[0.0]*n for _ in range(n)]
    ang_dist = angular_distance_deg
    rate = SLEW_RATE_DEG_PER_SEC
    for i in range(n):
        rai = obs[i].ra; deci = obs[i].dec
        rowi = mat[i]
        for j in range(n):
            if i == j:
                continue
            ang = ang_dist(rai, deci, obs[j].ra, obs[j].dec)
            rowi[j] = ang / rate
    return mat

def export_schedule(path, schedule):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","start_iso","start_ts","end_iso","end_ts","weight","ra","dec"])
        for o in schedule:
            # convert timestamps back to ISO for readability
            try:
                start_iso = datetime.fromtimestamp(o.start).isoformat()
                end_iso   = datetime.fromtimestamp(o.end).isoformat()
            except Exception:
                start_iso = ""
                end_iso = ""
            w.writerow([o.id, start_iso, o.start, end_iso, o.end, o.weight, o.ra, o.dec])

def main():
    all_rows = load_observations(INPUT_CSV)
    rows = sample_observations(all_rows)

    print("Computing slew matrix...")
    t0 = time.time()
    slew_matrix = compute_slew_matrix(rows)
    print("Slew matrix done. Time:", time.time() - t0)

    # GREEDY
    t0 = time.time()
    greedy_res = greedy_schedule(rows, slew_matrix)
    greedy_t = time.time() - t0
    print("Greedy schedule:", len(greedy_res), "items in", greedy_t, "seconds")
    export_schedule(OUT_GREEDY, greedy_res)

    # DYNAMIC
    t0 = time.time()
    dynamic_res = dynamic_weighted_interval_scheduling(rows)
    dynamic_t = time.time() - t0
    print("Dynamic DP schedule:", len(dynamic_res), "items in", dynamic_t, "seconds")
    export_schedule(OUT_DYNAMIC, dynamic_res)

    # ANT COLONY
    t0 = time.time()
    ant_res = ant_colony_schedule(rows, slew_matrix)
    ant_t = time.time() - t0
    print("Ant Colony schedule:", len(ant_res), "items in", ant_t, "seconds")
    export_schedule(OUT_ANT, ant_res)

    print("\n=== SUMMARY ===")
    print("Greedy:        items =", len(greedy_res), " time =", greedy_t)
    print("Dynamic (DP):  items =", len(dynamic_res), " time =", dynamic_t)
    print("Ant Colony:    items =", len(ant_res), " time =", ant_t)
    print("All schedules exported.")

if __name__ == "__main__":
    main()