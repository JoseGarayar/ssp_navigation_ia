"""Main file"""
import time
from memory_profiler import memory_usage

from utils import load_json
from iteration_value import value_iteration, plot_policy
from iteration_policy import policy_iteration
from iteration_policy_test import policy_iteration as policy_iteration_test

import psutil
import os
import re

def memory_usage_psutil():
    # Return the memory usage in KB
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024  # Convert bytes to kilobytes
    return mem

def extract_dimensions(filename):
    match = re.search(r'navigator(\d+)-(\d+)-\d+-\d+\.json', filename)
    if match:
        nx = int(match.group(1))
        ny = int(match.group(2))
        return nx, ny
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern.")

def print_values(V, policy, iterations, sort_values=True):
    """
    Print the values and policies.

    Args:
        V (dict): The value function for each state.
        policy (dict): The policy for each state.
        iterations (int): Number of iterations performed.
        sort_values (bool): Whether to sort the states before printing.
    """
    if sort_values:
        # Sort V values
        sorted_values = sorted(V.keys(), key=lambda x: int(x))
        V = {value: V[value] for value in sorted_values}
        # Sort policy states
        sorted_states = sorted(policy.keys(), key=lambda x: int(x))
        policy = {state: policy[state] for state in sorted_states}

    for state, value in zip(policy, V):
        print(f"State {state}: {policy[state]}, Value {value}: {V[value]}")
    print(f"Converged in {iterations} iterations")

def run_and_profile(file_path, algorithm,  gamma, epsilon, description, image_generation,  image_file):
    print(f"---Resultados {description}")
    initial_memory_usage  = memory_usage_psutil()
    initial_time = time.time()
    states = load_json(file_path)
    
    V, policy, iterations  = algorithm(states, gamma=gamma, epsilon=epsilon)
       
    final_time = time.time()
    current_memory_usage  = memory_usage_psutil()
    memory_used  = current_memory_usage  - initial_memory_usage 
    tiempo_ejecucion = (final_time - initial_time) * 1000  # Convert to milliseconds
    
    if image_generation==True:
        nx, ny = extract_dimensions(file_path)
        plot_policy(policy = policy, nx = nx, ny = ny, filename = image_file)

    print_values(V, policy, iterations)
    print(f"Convergence time: {tiempo_ejecucion:.2f} milisegundos")
    print(f"Memory used : {memory_used:.2f} KB")

def main():


    test_cases = [
        ("./json_files/test.json", policy_iteration_test, 0.9, 1e-3, "iteracion de política para test.json",False, ""),
        ("./json_files/navigator3-15-0-0.json", value_iteration, 1, 1e-6, "iteracion de valor para navigator3-15-0-0.json", True,  "value_iteration01.png"),
        ("./json_files/navigator3-15-0-0.json", policy_iteration, 0.9, 1e-3, "iteracion de política para navigator3-15-0-0.json",  True, "policy_ploy01.png"),
        ("./json_files/navigator4-10-0-0.json", value_iteration, 1, 1e-6, "iteracion de valor para navigator4-10-0-0.json",  True, "value_iteration02.png"),
        ("./json_files/navigator4-10-0-0.json", policy_iteration, 0.9, 1e-3, "iteracion de política para navigator4-10-0-0.json",  True, "policy_ploy02.png"),
    ]
    # test_cases = [        
    #     ("./json_files/test.json", policy_iteration_test, 0.9, 1e-3, "iteracion de política para test.json",False, ""),
    # ]
    
    for file_path, algorithm, gamma, epsilon, description,image_generation,  image_file in test_cases:    
        run_and_profile(file_path, algorithm, gamma, epsilon, description, image_generation, image_file)
        


if __name__ == "__main__":
    main()