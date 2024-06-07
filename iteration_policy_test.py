"""Algorithm for Policy Iteration for a test case"""

def policy_iteration(states, gamma=0.9, epsilon=1e-6):
    V = {state: 0 for state in states}
    policy = {
        "S0": "a00", 
        "S1": "a10", 
        "S2": "a20", 
        "S3": "a30", 
    }
    action_per_value = dict()
    for state in states:
        actions = []
        for adj in states[state]['Adj']:
            actions.extend(adj['A'].keys())
        action_per_value[state] = list(set(actions))

    def get_action_value(state, action):
        value = 0
        for adj in states[state]['Adj']:
            if action in adj['A']:
                prob = adj['A'][action]
                value += prob * (adj['cost'] + gamma * V[adj['name']])
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
            action_values = {action: get_action_value(state, action) 
                             for action in action_per_value[state]}
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
        if iterations == 10:
            break
        if policy_stable:
            break

    return V, policy, iterations

