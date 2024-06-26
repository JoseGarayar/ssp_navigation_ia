"""Main file"""
from utils import load_json
from iteration_value import value_iteration, plot_policy
from iteration_policy import policy_iteration
from iteration_policy_test import policy_iteration as policy_iteration_test
from visualization import bcolors

import time
import re
import tracemalloc
import csv

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


def run_and_profile(file_path, algorithm,  gamma, epsilon, description, image_generation=False, image_file="", random_values=False):
    print(f"{bcolors.BOLD_WARNING}---Resultados {description}{bcolors.ENDC}")
    initial_time = time.time()
    states = load_json(file_path)

    # tracemalloc starts
    tracemalloc.start()
    # Run algorithms
    V, policy, iterations  = algorithm(states, gamma=gamma, epsilon=epsilon, random_values=random_values)
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

    results = {"algorithm": algorithm.__name__,
               "file_path": file_path,
               "gamma": gamma,
               "epsilon" : epsilon,
               "random_values" : random_values,
               "iterations": iterations,
               "tiempo_ejecucion": tiempo_ejecucion,
               "memory_current_kb": current / 10**3,
               "memory_peak_kb": peak / 10**3}

    return results

def main():
    test_cases = [
        ("./json_files/test.json", policy_iteration_test, 0.9, 1e-3, "iteracion de política para test.json",False, "", False),
        ("./json_files/navigator3-15-0-0.json", value_iteration, 0.9, 1e-6, "Prueba 1 - iteracion de valor para navigator3-15-0-0.json", True, "value_iteration01_prueba1.png", False),
        ("./json_files/navigator3-15-0-0.json", value_iteration, 0.9, 1e-2, "Prueba 2 - iteracion de valor para navigator3-15-0-0.json", True, "value_iteration01_prueba2.png", False),
        ("./json_files/navigator3-15-0-0.json", value_iteration, 0.99, 1e-6, "Prueba 3 - iteracion de valor para navigator3-15-0-0.json", True,  "value_iteration01_prueba3.png", False),
        ("./json_files/navigator3-15-0-0.json", value_iteration, 0.1, 1e-6, "Prueba 4 - iteracion de valor para navigator3-15-0-0.json", True,  "value_iteration01_prueba4.png", False),
        ("./json_files/navigator3-15-0-0.json", value_iteration, 0.9, 1e-6, "Prueba 5 - iteracion de valor para navigator3-15-0-0.json", True, "value_iteration01_prueba5.png", True),
        ("./json_files/navigator3-15-0-0.json", policy_iteration, 0.9, 1e-6, "Prueba 1 - iteracion de política para navigator3-15-0-0.json", True, "policy_plot01_prueba1.png", False),
        ("./json_files/navigator3-15-0-0.json", policy_iteration, 0.9, 1e-2, "Prueba 2 - iteracion de política para navigator3-15-0-0.json", True, "policy_plot01_prueba2.png", False),
        ("./json_files/navigator3-15-0-0.json", policy_iteration, 0.99, 1e-6, "Prueba 3 - iteracion de política para navigator3-15-0-0.json", True, "policy_plot01_prueba3.png", False),
        ("./json_files/navigator3-15-0-0.json", policy_iteration, 0.1, 1e-6, "Prueba 4 - iteracion de política para navigator3-15-0-0.json", True, "policy_plot01_prueba4.png", False),
        ("./json_files/navigator3-15-0-0.json", policy_iteration, 0.9, 1e-6, "Prueba 5 - iteracion de política para navigator3-15-0-0.json", True, "policy_plot01_prueba5.png", True),
        ("./json_files/navigator4-10-0-0.json", value_iteration, 0.9, 1e-6, "Prueba 1 - iteracion de valor para navigator4-10-0-0.json", True, "value_iteration02_prueba1.png", False),
        ("./json_files/navigator4-10-0-0.json", value_iteration, 0.9, 1e-2, "Prueba 2 - iteracion de valor para navigator4-10-0-0.json", True, "value_iteration02_prueba2.png", False),
        ("./json_files/navigator4-10-0-0.json", value_iteration, 0.99, 1e-6, "Prueba 3 - iteracion de valor para navigator4-10-0-0.json", True, "value_iteration02_prueba3.png", False),
        ("./json_files/navigator4-10-0-0.json", value_iteration, 0.1, 1e-6, "Prueba 4 - iteracion de valor para navigator4-10-0-0.json", True, "value_iteration02_prueba4.png", False),
        ("./json_files/navigator4-10-0-0.json", value_iteration, 0.9, 1e-6, "Prueba 5 - iteracion de valor para navigator4-10-0-0.json", True, "value_iteration02_prueba5.png", True),
        ("./json_files/navigator4-10-0-0.json", policy_iteration, 0.9, 1e-6, "Prueba 1 - iteracion de política para navigator4-10-0-0.json", True, "policy_plot02_prueba1.png", False),
        ("./json_files/navigator4-10-0-0.json", policy_iteration, 0.9, 1e-2, "Prueba 2 - iteracion de política para navigator4-10-0-0.json", True, "policy_plot02_prueba2.png", False),
        ("./json_files/navigator4-10-0-0.json", policy_iteration, 0.99, 1e-6, "Prueba 3 - iteracion de política para navigator4-10-0-0.json", True, "policy_plot02_prueba3.png", False),
        ("./json_files/navigator4-10-0-0.json", policy_iteration, 0.1, 1e-6, "Prueba 4 - iteracion de política para navigator4-10-0-0.json", True, "policy_plot02_prueba4.png", False),
        ("./json_files/navigator4-10-0-0.json", policy_iteration, 0.9, 1e-6, "Prueba 5 - iteracion de política para navigator4-10-0-0.json", True, "policy_plot02_prueba5.png", True),
    ]
    # create records
    records = list()
    for file_path, algorithm, gamma, epsilon, description,image_generation, image_file, random_values in test_cases:    
        records.append(run_and_profile(file_path, algorithm, gamma, epsilon, description, image_generation, image_file, random_values))

    # refurbish records
    unique_keys = list(records[0].keys())

    # consolidate records
    data = dict()
    for key in unique_keys:
        data[key] = [record[key] for record in records]

    # print result
    # declare headers
    headers = {'algorithm': 'Algorithm',
               'file_path': 'File Path',
               'gamma': 'Gamma',
               'epsilon': 'Epsilon',
               'random_values': 'Valores iniciales random',
               'iterations': 'Iterations',
               'tiempo_ejecucion': 'Tiempo Ejecucion (ms)',
               'memory_current_kb': 'Memory Current (KB)',
               'memory_peak_kb': 'Memory Peak (KB)'}

    # find max length characters
    max_lengths = [max(len(header_title), max(len(str(data[header][i])) for i in range(len(data[header])))) for header, header_title in headers.items()]

    # function to plot each row
    def print_row(row, align='left'):
        formatted_row = [str(item).ljust(max_lengths[i]) if type(item) in [str, int] or i == 3 else str(round(float(item), 3)).ljust(max_lengths[i])
                        for i, item in enumerate(row)]
        
        print("| " + " | ".join(formatted_row) + " |")

    # print table
    print_row(list(headers.values()))
    print("-" * (sum(max_lengths) + len(headers) * 3 + 1))

    for i in range(len(data['algorithm'])):
        row = [
            data['algorithm'][i],
            data['file_path'][i],
            data['gamma'][i],
            data['epsilon'][i],
            data['random_values'][i],
            data['iterations'][i],
            data['tiempo_ejecucion'][i],
            data['memory_current_kb'][i],
            data['memory_peak_kb'][i]
        ]
        print_row(row)

    # store data into a pandas dataframe
   # Crear el diccionario de mapeo
    header_mapping = headers.copy()

    # Crear un archivo CSV
    with open('data_output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Escribir los encabezados amigables
        writer.writerow([header_mapping[header] for header in headers])

        # Escribir las filas de datos
        for i in range(len(data['algorithm'])):
            row = [
                data['algorithm'][i],
                data['file_path'][i],
                data['gamma'][i],
                data['epsilon'][i],
                data['random_values'][i],
                data['iterations'][i],
                data['tiempo_ejecucion'][i],
                data['memory_current_kb'][i],
                data['memory_peak_kb'][i]
            ]
            writer.writerow(row)

    print(f"{bcolors.BOLD_OKGREEN}Archivo CSV creado con éxito: data_output.csv{bcolors.ENDC}")


if __name__ == "__main__":
    main()