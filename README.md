# Frontier-Based Path Planning RL
This repository implements an autonomous **exploration / scanning path planning system** that combines:

- **Frontier-based exploration** (detecting unknown boundary regions)
- **Voronoi-diagram-based path graph reduction** for large-map efficiency
- **Reinforcement learning (RL)** for optimal frontier selection

It aims to reduce unnecessary path search cost, improve exploration coverage, and learn intelligent navigation strategies in large 2D environments.

---

## 🚀 Key Features

### 🔍 Frontier-Based Exploration  
Frontiers are the boundary between **known** and **unknown** space.  
This project performs:

- Occupancy grid update  
- Frontier detection (grid-based, BFS/edge extraction)  
- Frontier clustering  
- Next-target selection via RL or baseline heuristic  
- Exploration loop with map expansion  

Frontier extraction provides interpretable, efficient exploration targets.

## 🗺️ Voronoi Diagram–Based Path Graph Reduction  
Large grid maps make graph search expensive.  
To solve this, the project constructs a **Voronoi diagram over the free-space** to generate a skeletonized road graph.

Advantages:

- ✔ Vastly fewer nodes than full grid  
- ✔ Much cleaner topological representation  
- ✔ Natural clearance from obstacles  
- ✔ Faster path planning (A*, Dijkstra)  
- ✔ More stable paths for RL training  

### 🤖 Reinforcement Learning (DQN/PPO)  
Instead of picking the "closest" frontier or using simple heuristics,  
the agent learns to:

- Maximize map coverage  
- Minimize travel distance  
- Avoid dead-ends and local traps  
- Choose globally advantageous frontier targets  
- Develop long-horizon scanning strategies  

RL Inputs typically include:  
- Partial observation map  
- Frontier positions  
- Distance / cost information  
- Visited-cell reward  
- Collision penalties  

Training scripts and evaluation toolkit included.

