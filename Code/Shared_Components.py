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
# Shared Data Structures + CSV Loader
# ------------------------------------

import csv
import random
import math
import time

class Task:
    def __init__(self, name, start_time, end_time, priority):
        self.name = name
        self.start = float(start_time)
        self.end = float(end_time)
        self.priority = int(priority)

    def __repr__(self):
        return f"{self.name}(Priority={self.priority}, {self.start}-{self.end})"

def load_tasks_from_csv(path):
    tasks = []

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # Validate required columns
        required = {"start", "end", "weight"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV {path} must contain the columns: {', '.join(required)}"
            )

        for i, row in enumerate(reader):
            name = f"Task{i+1}"

            start = float(row["start"])
            end = float(row["end"])

            # Convert weight → priority (still integer 1–9)
            priority = int(row["weight"])

            tasks.append(Task(name, start, end, priority))

    return tasks
