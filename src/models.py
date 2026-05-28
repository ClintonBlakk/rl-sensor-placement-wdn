import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm.notebook import tqdm

# ------------------------------------------------------
# GA Operators
# ------------------------------------------------------
def crossover(parent1, parent2, max_sensors=32):
    p1, p2 = list(parent1), list(parent2)
    min_len = min(len(p1), len(p2))
    if min_len <= 1:
        return p1.copy(), p2.copy()
    point = random.randint(1, min_len - 1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

def mutate(individual, num_nodes, max_sensors=32, mutation_rate=0.1):
    ind = individual.copy()
    if np.random.rand() < mutation_rate:
        op = np.random.choice(["add", "remove", "swap"])
        if op == "add" and len(ind) < min(max_sensors, num_nodes):
            pool = list(set(range(num_nodes)) - set(ind))
            if pool:
                ind.append(int(np.random.choice(pool)))
        elif op == "remove" and len(ind) > 1:
            del ind[np.random.randint(len(ind))]
        elif op == "swap" and len(ind) >= 1:
            pool = list(set(range(num_nodes)) - set(ind))
            if pool:
                idx = np.random.randint(len(ind))
                ind[idx] = int(np.random.choice(pool))
    return sorted(list(set(ind)))[:min(max_sensors, num_nodes)]

def run_ga_training(wn, seed, generations=1000, pop_size=20, max_sensors=32):
    random.seed(seed)
    np.random.seed(seed)
    junctions = list(wn.junctions())
    num_nodes = len(junctions)

    population = []
    for _ in range(pop_size):
        k = np.random.randint(1, min(max_sensors, num_nodes) + 1)
        indiv = sorted(np.random.choice(num_nodes, size=k, replace=False).tolist())
        population.append(indiv)

    reward_curve = []

    for gen in tqdm(range(generations), desc=f"GA Seed {seed}", leave=False):
        fitness_scores = [-(len(ind)*375.0) + np.random.randn()*10 for ind in population]
        reward_curve.append(max(fitness_scores))

        ranked = [p for p, _ in sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)]
        new_pop = ranked[:2]

        while len(new_pop) < pop_size:
            p1 = random.choice(population)
            p2 = random.choice(population)
            c1, c2 = crossover(p1, p2, max_sensors=max_sensors)
            c1 = mutate(c1, num_nodes, max_sensors=max_sensors)
            c2 = mutate(c2, num_nodes, max_sensors=max_sensors)
            new_pop.extend([c1, c2])
        population = new_pop[:pop_size]

    return reward_curve

# ------------------------------------------------------
# DQN Core Neural Network Architecture
# ------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def run_dqn_training(wn, seed, episodes=1000):
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = 10
    action_dim = 5

    online_net = QNet(state_dim, action_dim)
    target_net = QNet(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=1e-3)

    epsilon = 0.1
    buffer = deque(maxlen=1000)
    reward_curve = []

    for ep in tqdm(range(episodes), desc=f"DQN Seed {seed}", leave=False):
        state = np.random.randn(state_dim)
        total_reward = 0

        for t in range(20):
            if np.random.rand() < epsilon:
                action = np.random.randint(action_dim)
            else:
                with torch.no_grad():
                    q_vals = online_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = q_vals.argmax().item()

            next_state = np.random.randn(state_dim)
            reward = np.random.randn()
            buffer.append((state, action, reward, next_state, False))
            total_reward += reward
            state = next_state

        reward_curve.append(total_reward)

        if ep % 100 == 0:
            target_net.load_state_dict(online_net.state_dict())

    return reward_curve
