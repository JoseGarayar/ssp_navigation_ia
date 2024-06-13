# Project Navigation - IA

This repository contains implementations of Policy Iteration and Value Iteration algorithms for solving Markov Decision Processes (MDPs). It includes test cases and JSON files with state information.

## File Structure

- `main.py`: The main file that runs the policy iteration and value iteration algorithms on given JSON files and prints the results.
- `utils.py`: Utility functions, including `load_json` to load state information from JSON files.
- `iteration_value.py`: Implementation of the Value Iteration algorithm.
- `iteration_policy.py`: Implementation of the Policy Iteration algorithm.
- `iteration_policy_test.py`: Test implementation of the Policy Iteration algorithm for a specific test case.
- `json_files/`: Directory containing JSON files with state information for running the algorithms.

## How to Run

1. Clone the repository:
   ```sh
   git clone https://github.com/JoseGarayar/ssp_navigation_ia.git
   cd ssp_navigation_ia

2. Ensure you have Python installed (preferably 3.11 or above).

3. Create a virtual environment 
   ```sh
   python3 -m venv .venv

4. Activate virtual environment (in Ubuntu)
   ```sh
   source .venv/bin/activate

5. Run the main script:
   ```sh
   python3 main.py

The script will load the JSON files, run the value iteration and policy iteration algorithms, and print the results.