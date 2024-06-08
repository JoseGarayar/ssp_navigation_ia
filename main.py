"""Main file"""

from utils import load_json
from iteration_value import value_iteration, plot_policy
from iteration_policy import policy_iteration
from iteration_policy_test import policy_iteration as policy_iteration_test


def print_values(V, policy, iterations, sort_values=True):
    if sort_values:
        # sort V values
        sorted_values = sorted(V.keys(), key=lambda x: int(x))
        V = {value: V[value] for value in sorted_values}
        # sort policy states
        sorted_states = sorted(policy.keys(), key=lambda x: int(x))
        policy = {state: policy[state] for state in sorted_states}
    
    print(f"Converged in {iterations} iterations")
    for state, value in zip(policy, V):
        print(f"State {state}: {policy[state]}, Value {value}: {V[value]}")


def main():
    # file_path_test = "./json_files/test.json"
    # states_test = load_json(file_path_test)
    # print("---Resultados iteracion de política para test.json")
    # V, policy, iterations = policy_iteration_test(states_test, gamma=0.9, epsilon=1e-3)
    # print_values(V, policy, iterations, sort_values=False)
    # print(V)
    # print(policy)
    # print(iterations)

    file_path_1 = "./json_files/navigator3-15-0-0.json"
    states_1 = load_json(file_path_1)
    print(states_1)
    
    # print("---Resultados iteracion de valor para navigator3-15-0-0.json")
    # V, policy, iterations = value_iteration(states_1,gamma=1,epsilon=1e-6)
    # print_values(V, policy, iterations)
    print("---Resultados iteracion de política para navigator3-15-0-0.json")
    V, policy, iterations = policy_iteration(states_1, gamma=0.9, epsilon=1e-3)
    print_values(V, policy, iterations)
    plot_policy(policy = policy, nx = 3, ny = 15, filename = 'policy_plot.png')

    # file_path_2 = "./json_files/navigator4-10-0-0.json"
    # states_2 = load_json(file_path_2)
    # V, policy, iterations = value_iteration(states_2,gamma=1,epsilon=1e-6)
    # print("---Resultados iteracion de valor para navigator4-10-0-0.json")
    # print_values(V, policy, iterations)
    # V, policy, iterations = policy_iteration(states_2, gamma=0.9, epsilon=1e-3)
    # print("---Resultados iteracion de política para navigator4-10-0-0.json")
    # print_values(V, policy, iterations)


if __name__ == "__main__":
    main()