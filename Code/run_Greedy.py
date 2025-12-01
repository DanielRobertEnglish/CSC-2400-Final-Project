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

# Greedy

from Shared_Components import load_tasks_from_csv
from Greedy import run_greedy

if __name__ == "__main__":
    filename = input("Enter path to your tasks CSV file (e.g., tasks.csv): ")
    tasks = load_tasks_from_csv(filename)
    print(f"\nLoaded {len(tasks)} tasks: {tasks}")

    result = run_greedy(tasks)
    print("\n🪙 Final Greedy Schedule:", result)