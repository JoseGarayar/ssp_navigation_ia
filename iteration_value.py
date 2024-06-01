import json
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def value_iteration(states, gamma=0.9, epsilon=1e-6):
    V = {state: 0 for state in states}
    policy = {state: None for state in states}

    def get_min_action_value(state):
        min_value = float('inf')
        best_action = None
        value_per_action = dict()
        for adj in states[state]['Adj']:
            for action, prob in adj['A'].items():
                value = prob * (1 + gamma * V[adj['name']])
                if not value_per_action.get(action):
                    value_per_action[action] = value
                else:
                    value_per_action[action] += value
        for action, value in value_per_action.items():
            if value < min_value:
                min_value = value
                best_action = action
        return min_value, best_action

    iteration = 0
    while True:
        delta = 0
        for state in states:
            if states[state]['goal'] or states[state]['deadend']:
                V[state] = 0
                continue
            min_value, best_action = get_min_action_value(state)
            delta = max(delta, abs(min_value - V[state]))
            V[state] = min_value
            policy[state] = best_action

        iteration += 1
        if delta < epsilon:
            break

    return V, policy, iteration

def plot_policy(states, policy, nx, ny, filename='policy_plot.png'):
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xticks(np.arange(0, nx, 1))
    ax.set_yticks(np.arange(0, ny, 1))
    ax.grid(True)

    for state in states:
        x = (int(state) - 1) % nx
        y = (int(state) - 1) // nx
        action = policy[state]
        if action == 'N':
            dx, dy = 0, 0.5
        elif action == 'S':
            dx, dy = 0, -0.5
        elif action == 'E':
            dx, dy = 0.5, 0
        elif action == 'W':
            dx, dy = -0.5, 0
        else:
            dx, dy = 0, 0
        ax.text(x,y,s=f"{x},{y}")
        ax.arrow(x + 0.5, y + 0.5, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    plt.gca().invert_yaxis()
    plt.savefig(filename)
    # plt.show(block=True)


def main():
    # file_path = "./json_files/navigator3-15-0-0.json"
    file_path = "./json_files/navigator4-10-0-0.json"
    states = load_json(file_path)
    V, policy, iterations = value_iteration(states,gamma=1,epsilon=1e-6)
    
    print(f"Converged in {iterations} iterations")
    for state in policy:
        print(f"State {state}: {policy[state]}")
    for value in V:
        print(f"Value {value}: {V[value]}")

    nx, ny = 3, 24  # Replace with actual dimensions of the grid
    plot_policy(states, policy, nx, ny)

if __name__ == "__main__":
    main()
