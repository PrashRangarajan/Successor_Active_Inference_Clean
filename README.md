# Hierarchical Active Inference with Successor Representations

A modular Python framework for hierarchical active inference using successor representations. Supports both MDP and POMDP environments with automatic macro-state discovery via spectral clustering.

## Features

- **Hierarchical Planning**: Two-level planning with automatic macro-state discovery
- **Successor Representations**: Learn expected future state occupancy for efficient planning
- **POMDP Support**: Full belief-based learning and inference with Bayesian state estimation
- **Multiple Environments**: Gridworld, Key Gridworld, Mountain Car, Acrobot
- **Modular Architecture**: Easy to add new environments via adapter pattern
- **Visualization**: Built-in plotting for value functions, trajectories, and macro-state networks

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HierarchicalSRAgent                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  Macro Planning │  │  Micro Planning │  │   SR Learning   │         │
│  │  (Clustering)   │  │  (Value-based)  │  │   (TD / Exact)  │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           └──────────────┬─────┴─────────────────────┘                  │
│                          ▼                                               │
│                 BaseEnvironmentAdapter (ABC)                             │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Gridworld   │   │  POMDPGridworld │   │   MountainCar   │
│    Adapter    │   │     Adapter     │   │     Adapter     │
│               │   │                 │   │                 │
│ • MDP         │   │ • Belief track  │   │ • Continuous    │
│ • Discrete    │   │ • Bayesian inf  │   │ • Discretized   │
│ • 4 actions   │   │ • Obs. entropy  │   │ • 3 actions     │
└───────┬───────┘   └────────┬────────┘   └────────┬────────┘
        │                    │                     │
        ▼                    ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ StandardGrid  │   │  StandardGrid   │   │   Gymnasium     │
│   (unified)   │   │   + Noise A     │   │  MountainCar    │
└───────────────┘   └─────────────────┘   └─────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/PrashRangarajan/hierarchical-active-inference.git
cd hierarchical-active-inference

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### Basic MDP Gridworld

```python
from unified_env import StandardGridworld
from environments.gridworld import GridworldAdapter
from core import HierarchicalSRAgent

# Create environment and adapter
env = StandardGridworld(grid_size=9)
adapter = GridworldAdapter(env, grid_size=9)

# Create agent
agent = HierarchicalSRAgent(adapter, n_clusters=5, gamma=0.99)
agent.set_goal((8, 8), reward=100.0)

# Learn environment dynamics
agent.learn_environment(num_episodes=2000)

# Run episode
agent.reset_episode(init_state=0)
result = agent.run_episode_hierarchical(max_steps=200)
print(f"Reached goal: {result['reached_goal']} in {result['steps']} steps")
```

### POMDP with Noisy Observations

```python
from unified_env import StandardGridworld
from environments.pomdp_gridworld import POMDPGridworldAdapter
from core import HierarchicalSRAgent

# Create POMDP environment (20% observation noise)
env = StandardGridworld(grid_size=9)
adapter = POMDPGridworldAdapter(
    env,
    grid_size=9,
    noise_level=0.2,  # 20% chance of incorrect observation
    use_true_state_for_learning=False  # Full POMDP mode
)

# Agent learns and plans using beliefs, not true states
agent = HierarchicalSRAgent(adapter, n_clusters=5)
agent.set_goal((8, 8))
agent.learn_environment(num_episodes=2000)

# During execution, agent maintains belief distribution
agent.reset_episode(init_state=0)
result = agent.run_episode_flat(max_steps=200)

# Check belief vs reality
print(f"Believes at goal: {result['reached_goal']}")
print(f"Actually at goal: {adapter.get_true_state_index() in agent.goal_states}")
```

### Continuous Control (Mountain Car)

```python
import gymnasium as gym
from environments.mountain_car import MountainCarAdapter
from core import HierarchicalSRAgent

# Create continuous environment
env = gym.make('MountainCar-v0')
adapter = MountainCarAdapter(env, n_pos_bins=10, n_vel_bins=10)

# Learn with more episodes (continuous needs more exploration)
agent = HierarchicalSRAgent(adapter, n_clusters=5)
agent.set_goal(goal_spec='right_hill')
agent.learn_environment(num_episodes=4000)

# Run episode
agent.reset_episode()
result = agent.run_episode_hierarchical(max_steps=500)
```

## Project Structure

```
├── core/                          # Core framework
│   ├── base_environment.py        # BaseEnvironmentAdapter ABC
│   ├── hierarchical_agent.py      # HierarchicalSRAgent
│   ├── state_space.py             # State space representations
│   └── visualization.py           # Plotting utilities
│
├── environments/                  # Environment adapters
│   ├── gridworld/                 # Standard MDP gridworld
│   ├── pomdp_gridworld/           # POMDP with belief tracking
│   ├── key_gridworld/             # Augmented state (location + has_key)
│   ├── mountain_car/              # Continuous Mountain Car
│   └── acrobot/                   # Continuous Acrobot
│
├── examples/                      # Working examples
│   ├── run_gridworld.py           # Basic MDP example
│   ├── run_pomdp_gridworld.py     # POMDP with belief tracking
│   ├── run_key_gridworld.py       # Augmented state example
│   ├── run_mountain_car.py        # Continuous control
│   └── run_acrobot.py             # Another continuous example
│
├── unified_env/                   # Unified gridworld environments
│   ├── base.py                    # Base gridworld with noise
│   ├── standard.py                # Standard gridworld
│   ├── key_gridworld.py           # Key pickup variant
│   └── transitions.py             # Transition matrix generation
│
├── data/                          # Cached learned matrices
├── figures/                       # Output visualizations
└── videos/                        # Recorded episodes
```

## Environments

### MDP Gridworld
Standard fully observable gridworld with walls. Agent knows exact state.

### POMDP Gridworld
Partially observable gridworld where:
- Agent receives noisy observations (configurable noise level)
- Maintains belief distribution via Bayesian inference
- Plans using expected values over belief states
- Supports "hallway" states with extra observation noise

### Key Gridworld
Augmented state space `(location, has_key)` where agent must:
1. Navigate to key location
2. Pick up the key
3. Navigate to goal (only reachable with key)

### Mountain Car / Acrobot
Continuous control environments from Gymnasium, discretized for SR learning.

## Key Concepts

### Successor Representation (SR)
The SR matrix `M[s, s']` represents the expected discounted future occupancy of state `s'` starting from state `s`:

```
M = (I - γB)^(-1)
```

where `B` is the transition matrix and `γ` is the discount factor.

### Value Function via SR
Given preferences `C` over states (rewards), the value function is simply:

```
V = M @ C
```

### Hierarchical Planning
1. **Macro-level**: Spectral clustering on SR discovers "rooms" or regions
2. **Bottleneck detection**: Identify states that connect macro-states
3. **Hierarchical policy**: Plan at macro-level, execute at micro-level

### POMDP Belief Updates
For partially observable environments, beliefs are updated via Bayes rule:

```
b'(s') ∝ P(o|s') Σ_s P(s'|s,a) b(s)
```

## Adding New Environments

1. Create a new adapter inheriting from `BaseEnvironmentAdapter`
2. Implement required abstract methods:
   - `reset()`, `step()`, `get_current_state_index()`
   - `create_goal_prior()`, `get_transition_matrix()`
   - Matrix operations: `multiply_B_s()`, `multiply_M_C()`
3. Add to `environments/__init__.py`

Example:
```python
from core.base_environment import BaseEnvironmentAdapter

class MyEnvironmentAdapter(BaseEnvironmentAdapter):
    def reset(self, init_state=None):
        # Reset environment, return initial state
        ...

    def step(self, action):
        # Take action, return new state
        ...

    # ... implement other required methods
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hierarchical_active_inference,
  author = {Rangarajan, Prashant},
  title = {Hierarchical Active Inference with Successor Representations},
  year = {2024},
  url = {https://github.com/your-username/hierarchical-active-inference}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
