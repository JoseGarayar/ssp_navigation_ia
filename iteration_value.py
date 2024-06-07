"""Algorithm for Value Iteration"""

import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def value_iteration(states, gamma=0.9, epsilon=1e-6):
    """
    Perform value iteration to find the optimal policy and value function.

    Args:
        states (dict): A dictionary where keys are state names and values are dictionaries with state information.
        gamma (float): Discount factor for future costs.
        epsilon (float): Small value for determining convergence of the value function.

    Returns:
        V (dict): The value function for each state.
        policy (dict): The optimal policy for each state.
        iteration (int): Number of iterations performed.
    """
    # Initialize value function for each state to 0
    V = {state: 0 for state in states}

    # Initialize an empty policy
    policy = {state: None for state in states}

    def get_min_action_value(state):
        """
        Compute the minimum value of taking any action in a given state.

        Args:
            state (str): The state name.

        Returns:
            tuple: Minimum value and the best action.
        """
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


