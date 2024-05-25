import json
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def policy_iteration(states, gamma=0.9, epsilon=1e-6):
    V = {state: 0 for state in states}
    policy = {state: 'N' for state in states}  # Initial policy with arbitrary action

    def get_action_value(state, action):
        value = 0
        for adj in states[state]['Adj']:
            if action in adj['A']:
                prob = adj['A'][action]
                value += prob * (1 + gamma * V[adj['name']])
        return value

    def evaluate_policy():
        while True:
            delta = 0
            for state in states:
                if states[state]['goal'] or states[state]['deadend']:
                    continue
                v = V[state]
                V[state] = get_action_value(state, policy[state])
                delta = max(delta, abs(v - V[state]))
            if delta < epsilon:
                break

    def improve_policy():
        policy_stable = True
        for state in states:
            if states[state]['goal'] or states[state]['deadend']:
                continue
            old_action = policy[state]
            action_values = {action: get_action_value(state, action) for action in ['N', 'S', 'E', 'W']}
            best_action = max(action_values, key=action_values.get)
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = True
        return policy_stable

    iterations = 0
    while True:
        evaluate_policy()
        policy_stable = improve_policy()
        iterations += 1
        if policy_stable:
            break

    return V, policy, iterations

def main():
    file_path = "/mnt/data/navigator4-10-0-0.json"
    states = load_json(file_path)
    V, policy, iterations = policy_iteration(states)
    
    print(f"Converged in {iterations} iterations")
    for state in policy:
        print(f"State {state}: {policy[state]}")

if __name__ == "__main__":
    main()
