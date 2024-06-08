
def policy_iteration(states, gamma=0.9, epsilon=1e-6):
    """
    Perform policy iteration to find the optimal policy and value function.

    Args:
        states (dict): A dictionary where keys are state names and values are dictionaries with state information.
        gamma (float): Discount factor for future costs.
        epsilon (float): Small value for determining convergence of the value function.

    Returns:
        V (dict): The value function for each state.
        policy (dict): The optimal policy for each state.
        iterations (int): Number of iterations performed.
    """
    # Initialize value function for each state to 0
    V = {state: 0 for state in states}

    # Initialize a policy with an arbitrary action 'N' for each state
    policy = {state: 'N' for state in states}

    # Map to store available actions for each state
    action_per_value = dict()
    for state in states:
        actions = []
        for adj in states[state]['Adj']:
            actions.extend(adj['A'].keys())
        action_per_value[state] = list(set(actions))

    def get_action_value(state, action):
        """
        Compute the value of taking a specific action in a given state.

        Args:
            state (str): The state name.
            action (str): The action to evaluate.

        Returns:
            float: The computed value of the action.
        """
        value = 0
        for adj in states[state]['Adj']:
            if action in adj['A']:
                prob = adj['A'][action]
                value += prob * (1 + gamma * V[adj['name']])
        return value

    def evaluate_policy():
        """
        Evaluate the current policy by updating the value function until convergence.
        """
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
        """
        Improve the current policy by making it greedy with respect to the current value function.

        Returns:
            bool: True if the policy is stable (no changes), False otherwise.
        """
        policy_stable = True
        for state in states:
            if states[state]['goal'] or states[state]['deadend']:
                policy[state] = 'goal' if states[state]['goal'] else 'deadend'
                continue
            old_action = policy[state]
            action_values = {action: get_action_value(state, action) for action in action_per_value[state]}
            best_action = min(action_values, key=action_values.get)
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable

    iterations = 0
    while True:
        evaluate_policy()
        policy_stable = improve_policy()
        iterations += 1
        if policy_stable:
            break

    return V, policy, iterations

