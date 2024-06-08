"""Main file"""
from utils import load_json
from iteration_value import value_iteration, plot_policy
from iteration_policy import policy_iteration
from iteration_policy_test import policy_iteration as policy_iteration_test
from visualization import bcolors

import time
import re
import tracemalloc


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
        print(f"{bcolors.BLUE}State {state}:{bcolors.ENDC} {bcolors.OKGREEN}{policy[state]},{bcolors.ENDC} {bcolors.BLUE}Value {value}:{bcolors.ENDC} {bcolors.OKGREEN}{V[value]:.4f}{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}Converged in {iterations} iterations{bcolors.ENDC}")


def run_and_profile(file_path, algorithm,  gamma, epsilon, description, image_generation=False,  image_file=""):
    print(f"{bcolors.BOLD_WARNING}---Resultados {description}{bcolors.ENDC}")
    initial_time = time.time()
    states = load_json(file_path)

    # tracemalloc starts
    tracemalloc.start()
    # Run algorithms
    V, policy, iterations  = algorithm(states, gamma=gamma, epsilon=epsilon)
    # Get values and stop tracemalloc
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
       
    final_time = time.time()
    tiempo_ejecucion = (final_time - initial_time) * 1000  # Convert to milliseconds
    
    if image_generation==True:
        nx, ny = extract_dimensions(file_path)
        plot_policy(policy = policy, nx = nx, ny = ny, filename = image_file)

    print_values(V, policy, iterations, image_generation)
    print(f"{bcolors.RED}Convergence time: {tiempo_ejecucion:.2f} milisegundos{bcolors.ENDC}")
    print(f"{bcolors.CYAN}Current memory usage is {current / 10**3} KB; Peak was {peak / 10**3} KB{bcolors.ENDC}\n")


def main():
    test_cases = [
        ("./json_files/test.json", policy_iteration_test, 0.9, 1e-3, "iteracion de política para test.json",False, ""),
        ("./json_files/navigator3-15-0-0.json", value_iteration, 1, 1e-6, "iteracion de valor para navigator3-15-0-0.json", True,  "value_iteration01.png"),
        ("./json_files/navigator3-15-0-0.json", policy_iteration, 0.9, 1e-3, "iteracion de política para navigator3-15-0-0.json",  True, "policy_plot01.png"),
        ("./json_files/navigator4-10-0-0.json", value_iteration, 1, 1e-6, "iteracion de valor para navigator4-10-0-0.json",  True, "value_iteration02.png"),
        ("./json_files/navigator4-10-0-0.json", policy_iteration, 0.9, 1e-3, "iteracion de política para navigator4-10-0-0.json",  True, "policy_plot02.png"),
    ]
    for file_path, algorithm, gamma, epsilon, description,image_generation,  image_file in test_cases:    
        run_and_profile(file_path, algorithm, gamma, epsilon, description, image_generation, image_file)


if __name__ == "__main__":
    main()