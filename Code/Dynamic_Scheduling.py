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
# Dynamic Scheduling
# ------------------------------------

from Shared_Components import *

class DynamicScheduler:
    def __init__(self):
        self.active_tasks = []
        self.schedule = []
        self.current_time = 0

    def add_task(self, task):
        print(f"[Time {self.current_time}] Adding task {task.name}")
        self.active_tasks.append(task)
        self.reschedule()

    def remove_task(self, task_name):
        print(f"[Time {self.current_time}] Removing task {task_name}")
        self.active_tasks = [t for t in self.active_tasks if t.name != task_name]
        self.reschedule()

    def reschedule(self):
        # Sort by weight, then earliest finish time
        tasks_sorted = sorted(self.active_tasks,
                              key=lambda t: (-t.priority, t.end))

        new_schedule = []
        current_end = -1

        for task in tasks_sorted:
            if task.start >= current_end:
                new_schedule.append(task)
                current_end = task.end

        self.schedule = new_schedule
        print(f"[Time {self.current_time}] New Schedule: {new_schedule}\n")

    def simulate(self, steps=3, delay=0):
        for _ in range(steps):
            self.current_time += 1
            if delay:
                time.sleep(delay)
            print(f"Tick {self.current_time}: Active schedule: {self.schedule}")

def run_dynamic(tasks):
    print("\n🔁 Running Dynamic Scheduler...")
    scheduler = DynamicScheduler()
    for task in tasks:
        scheduler.add_task(task)

    scheduler.simulate(2)
    return scheduler.schedule
