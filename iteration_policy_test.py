import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def policy_iteration(states, gamma=0.9, epsilon=1e-6):
    V = {state: 0 for state in states}
    # policy = {state: 'N' for state in states}  # Initial policy with arbitrary action
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
        # state = S0 , action = a00
        # state = S1 , action = a10
        # state = S2 , action = a20
        # state = S3 , action = a30
        value = 0
        for adj in states[state]['Adj']:
            if action in adj['A']:
                prob = adj['A'][action]
                value += prob * (adj['cost'] + gamma * V[adj['name']])
                # value = 0.4 * (2 + 0) = 0.8      0.4 * (2 + 5.8)
                # value = 0.6 * (2 + 0) = 1.2      0.6 * (2 + 6.8)      2 + 0.4*5.8 + 0.6*6.8 =
                # value = 2
                #
                # value = 1 * (1 + 0) = 1
                # value = 1
                #
                # value = 0.8 * (5 + 1 * 0) = 4
                # value = 0.2 * (5 + 1 * 0) = 1
                # value = 5
                #
                # value = 1 * (1 + 1 * 0) = 1

        return value

    def evaluate_policy():
        while True:
            delta = 0
            for state in states:
                if states[state]['goal'] or states[state]['deadend']:
                    continue
                v = V[state]
                # v = 0
                V[state] = get_action_value(state, policy[state])
                # V[S0] = 2
                # V[S1] = 1
                # V[S2] = 5
                # V[S3] = 1
                delta = max(delta, abs(v - V[state]))
                # delta = 2
                # delta = 1
                # delta = 5
                # delta = 1
            if delta < epsilon:
                break

    def improve_policy():
        policy_stable = True
        for state in states:
            if states[state]['goal'] or states[state]['deadend']:
                continue
            old_action = policy[state]
            # Para state S0 -> old_action = a00
            # Para state S1 -> old_action = a10
            action_values = {action: get_action_value(state, action) 
                             for action in action_per_value[state]}
            # action_values = {"a00": 8.4}
            # action_values = {"a10": 6.8, "a11": 3}
            best_action = min(action_values, key=action_values.get)
            # best_action = "a00"
            # best_action = "a11"
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable

    iterations = 0
    while True:
        evaluate_policy()
        print("V", V)
        print("policy", policy)
        policy_stable = improve_policy()
        iterations += 1
        if iterations == 10:
            break
        if policy_stable:
            break

    return V, policy, iterations

def main():
    # file_path = "./json_files/navigator3-15-0-0.json"
    # file_path = "./json_files/navigator4-10-0-0.json"
    file_path = "./json_files/test.json"
    states = load_json(file_path)
    V, policy, iterations = policy_iteration(states, gamma=1)
    
    print(f"Converged in {iterations} iterations")
    for state in policy:
        print(f"State {state}: {policy[state]}")
    for value in V:
        print(f"Value {value}: {V[value]}")

if __name__ == "__main__":
    main()
