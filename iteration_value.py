import json
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def value_iteration(states, gamma=0.9, epsilon=1e-6):
    V = {state: 0 for state in states}
    policy = {state: None for state in states}

    def get_max_action_value(state):
        max_value = -float('inf')
        best_action = None
        for adj in states[state]['Adj']:
            for action, prob in adj['A'].items():
                value = prob * (1 + gamma * V[adj['name']])
                if value > max_value:
                    max_value = value
                    best_action = action
        return max_value, best_action

    iteration = 0
    while True:
        delta = 0
        for state in states:
            if states[state]['goal'] or states[state]['deadend']:
                continue
            max_value, best_action = get_max_action_value(state)
            delta = max(delta, abs(max_value - V[state]))
            V[state] = max_value
            policy[state] = best_action

        iteration += 1
        if delta < epsilon:
            break

    return V, policy, iteration

def main():
    file_path = "/mnt/data/navigator4-10-0-0.json"
    states = load_json(file_path)
    V, policy, iterations = value_iteration(states)
    
    print(f"Converged in {iterations} iterations")
    for state in policy:
        print(f"State {state}: {policy[state]}")

if __name__ == "__main__":
    main()
