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
# Greedy Scheduling
# ------------------------------------

from Shared_Components import *

def greedy_schedule(tasks):
    # Sort by: 1) highest weight (priority), 2) earliest end time
    tasks_sorted = sorted(tasks, key=lambda t: (-t.priority, t.end))

    schedule = []
    current_end = -1

    for task in tasks_sorted:
        if task.start >= current_end:
            schedule.append(task)
            current_end = task.end

    return schedule

def run_greedy(tasks):
    print("\n🪙 Running Greedy Scheduler...")
    result = greedy_schedule(tasks)
    print("Greedy Schedule:", result)
    return result

