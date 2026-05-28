import os
import numpy as np
import matplotlib.pyplot as plt
import wntr

# Import optimization engines from our clean backbone file
from models import run_dqn_training, run_ga_training

# ------------------------------------------------------
# Config & Dynamic Relative Path Management
# ------------------------------------------------------
EPISODES = 1000
SEEDS = 10

# Safely targets the project root folder regardless of execution scope
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORKS_DIR = os.path.join(BASE_DIR, "networks")

# Map your clean files matching GitHub precisely, including C-Town!
EXPERIMENT_CONFIGS = {
    "C-Town": os.path.join(NETWORKS_DIR, "CTOWN.inp"),
    "Hanoi": os.path.join(NETWORKS_DIR, "Hanoi.inp"),
    "Net3": os.path.join(NETWORKS_DIR, "Net3.inp")
}

# ------------------------------------------------------
# Core Experiment Driver
# ------------------------------------------------------
def run_experiment(inp_file, network_name):
    print(f"\n=== Running Audit Evaluation: {network_name} ===")
    if not os.path.exists(inp_file):
        print(f"❌ Error: Network file not found at {inp_file}")
        return

    dqn_results = []
    ga_results = []

    for seed in range(SEEDS):
        wn = wntr.network.WaterNetworkModel(inp_file)
        dqn_results.append(run_dqn_training(wn, seed, EPISODES))
        ga_results.append(run_ga_training(wn, seed, EPISODES))

    dqn_mean = np.mean(dqn_results, axis=0)
    ga_mean  = np.mean(ga_results, axis=0)

    # Plot results
    episodes_arr = np.arange(1, EPISODES+1)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_arr, dqn_mean, label="DQN Mean Execution", color='blue')
    plt.plot(episodes_arr, ga_mean, label="GA Mean Execution", color='green')
    plt.xlabel("Episode Framework")
    plt.ylabel("System Cumulative Reward")
    plt.title(f"10-Seed Benchmark Optimization Discrepancy: DQN vs GA ({network_name})")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------------------------------
# Execution Layer
# ------------------------------------------------------
if __name__ == "__main__":
    for name, path in EXPERIMENT_CONFIGS.items():
        run_experiment(path, name)
