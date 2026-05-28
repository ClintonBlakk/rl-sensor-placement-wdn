# Adaptive Sensor Placement in Water Distribution Networks Using Deep Reinforcement Learning: A Comparative Study with Genetic Algorithms
This repository contains the official algorithmic framework, benchmark dataset infrastructure, and execution pipelines for addressing the sensor placement optimization problem across complex Water Distribution Networks (WDNs). 

The project evaluates and validates the convergence, performance, and stability of a **Deep Q-Network (DQN)** agent compared against a standardized **Genetic Algorithm (GA)** across multi-seed computational horizons.

---

## 📂 Repository Structure

The workspace is organized using a decoupled, modular layout to ensure a clean, reproducible structural audit trail:

```text
rl-sensor-placement-wdn/
├── networks/               # Standardized Hydraulic Benchmark Dataset Layers
│   ├── CTOWN.inp           # Medium-sized complex looped network benchmark
│   ├── Hanoi.inp           # High-looping density verification network
│   └── Net3.inp            # Standard baseline validation benchmark
└── src/                    # Optimization Engines & Algorithmic Backbone
    ├── models.py           # Core DQN PyTorch architecture and GA operators
    └── main.py             # Master evaluation loop and comparative plotting driver
