import os
import time
import csv

# Import shared loader
from Shared_Components import load_tasks_from_csv

# Import the 3 algorithms directly
from Greedy import run_greedy
from Ant_Colony import run_ant_colony
from Dynamic_Scheduling import run_dynamic


# ------------------------------------------------------------
# Time a function
# ------------------------------------------------------------
def measure(func, tasks):
    start = time.time()
    result = func(tasks)
    return time.time() - start, result


# ------------------------------------------------------------
# Write optimized schedule CSV
# With new first column = ORIGINAL CSV INDEX
# ------------------------------------------------------------
def write_schedule_csv(output_name, original_tasks, optimized_list):
    """
    original_tasks: list of Task objects from original CSV
    optimized_list: list of Task objects in final schedule order
    """

    # Map task name → original index in CSV
    index_map = {}
    for i, t in enumerate(original_tasks):
        index_map[t.name] = i

    with open(output_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_index", "start", "end", "weight"])

        for t in optimized_list:
            writer.writerow([
                index_map[t.name],   # new first column
                t.start,
                t.end,
                t.priority
            ])


# ------------------------------------------------------------
# Pretty-print schedule with newlines every 3 tasks
# ------------------------------------------------------------
def print_schedule(title, schedule):
    print(f"\n--- {title} Schedule (formatted) ---")
    for i, t in enumerate(schedule):
        print(t, end="  ")
        if (i + 1) % 3 == 0:
            print()
    print("\n")


# ------------------------------------------------------------
# Main execution: loop through ALL CSV files
# ------------------------------------------------------------
if __name__ == "__main__":
    folder = input("Enter folder containing CSV schedule files: ").strip()

    folder = os.path.expanduser(folder)
    folder = os.path.abspath(folder)

    print(f"Using folder: {folder}")

    if not os.path.isdir(folder):
        print("Error: Folder not found.")
        exit(1)

    results = {}
    first_file_processed = False

    csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    csv_files.sort()

    for filename in csv_files:
        full_path = os.path.join(folder, filename)
        print(f"\n=== Processing {filename} ===")

        tasks = load_tasks_from_csv(full_path)
        print(f"Loaded {len(tasks)} tasks")

        # ---- time algorithms AND retrieve their schedules ----
        t_g, out_g = measure(run_greedy, tasks)
        t_a, out_a = measure(run_ant_colony, tasks)
        t_d, out_d = measure(run_dynamic, tasks)

        # Save times
        results[filename] = {
            "greedy": t_g,
            "aco": t_a,
            "dynamic": t_d
        }

        print(f"Greedy:  {t_g:.6f}s")
        print(f"ACO:     {t_a:.6f}s")
        print(f"Dynamic: {t_d:.6f}s")

        # ---- For the FIRST CSV file only: output 3 optimized CSVs ----
        if not first_file_processed:
            print("\nCreating optimized schedule CSVs for FIRST file only...")

            write_schedule_csv("optimized_greedy.csv", tasks, out_g)
            write_schedule_csv("optimized_aco.csv", tasks, out_a)
            write_schedule_csv("optimized_dynamic.csv", tasks, out_d)

            # Print schedules nicely
            print_schedule("Greedy", out_g)
            print_schedule("ACO", out_a)
            print_schedule("Dynamic", out_d)

            first_file_processed = True

    # --------------------------------------------------------
    # Write algorithm timing CSV (3 columns, 1 row per CSV)
    # --------------------------------------------------------
    with open("algorithm_timings.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "greedy", "aco", "dynamic"])

        for fname, data in results.items():
            writer.writerow([
                fname,
                f"{data['greedy']:.6f}",
                f"{data['aco']:.6f}",
                f"{data['dynamic']:.6f}"
            ])

    print("\nTiming results written to algorithm_timings.csv")
    print("Optimized schedules written for the FIRST CSV only:")
    print("   optimized_greedy.csv")
    print("   optimized_aco.csv")
    print("   optimized_dynamic.csv")