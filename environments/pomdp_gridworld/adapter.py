"""POMDP Gridworld environment adapter for Hierarchical SR Active Inference.

This adapter extends the MDP gridworld with partial observability:
- Agent receives noisy observations instead of exact state
- Maintains belief state over possible locations
- Uses Bayesian inference to update beliefs
"""

from typing import Any, List, Optional, Tuple
import numpy as np
from scipy import stats

from core.base_environment import BaseEnvironmentAdapter
from core.state_space import GridStateSpace

def log_stable(x):
    """Numerically stable log."""
    return np.log(x + 1e-10)

def softmax(dist):
    """Compute softmax of a distribution."""
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

def entropy(A):
    """Compute entropy of observation model columns."""
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A

def kl_divergence(q1, q2):
    """Compute KL divergence between two distributions."""
    return (log_stable(q1) - log_stable(q2)) @ q1

def infer_states(observation_index: int, A: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Bayesian belief update given observation.

    Args:
        observation_index: Index of the observation received
        A: Observation likelihood matrix P(o|s), shape (n_obs, n_states)
        prior: Prior belief over states P(s), shape (n_states,)

    Returns:
        Posterior belief P(s|o), shape (n_states,)
    """
    log_likelihood = log_stable(A[observation_index, :])
    log_prior = log_stable(prior)
    posterior = softmax(log_likelihood + log_prior)
    return posterior

class POMDPGridworldAdapter(BaseEnvironmentAdapter):
    """Adapter for POMDP gridworld with noisy observations.

    State representation: (x, y) grid coordinates (hidden state)
    Observation: noisy version of true state
    Belief: probability distribution over states

    The agent operates on beliefs rather than true states, using
    Bayesian inference to update beliefs based on observations.
    """

    def __init__(self, env, grid_size: int, noise_level: float = 0.1,
                 noisy_states: Optional[List[int]] = None,
                 noise_spread: float = 3.0,
                 use_true_state_for_learning: bool = False):
        """
        Args:
            env: SR_Gridworld environment instance
            grid_size: Size of the grid
            noise_level: Probability of getting a noisy observation (default 0.1)
            noisy_states: List of state indices with extra noise (e.g., hallways)
            noise_spread: How spread out the noise is for noisy states (gamma scale)
            use_true_state_for_learning: If True, get_current_state_index() returns true state
                during learning (hybrid approach). If False, always returns belief state
                (full POMDP). Default False for full POMDP behavior.
        """
        self._env = env
        self.grid_size = grid_size
        self._state_space = GridStateSpace(grid_size)
        self._n_actions = 4
        self._n_states = grid_size ** 2

        # Hidden state (true position)
        self._current_state = None  # One-hot encoded

        # POMDP learning mode
        self._use_true_state_for_learning = use_true_state_for_learning
        self._in_learning_mode = False  # Set by agent during learning

        # Observation model A: P(o|s), shape (n_obs, n_states)
        self._A = self._create_observation_model(noise_level, noisy_states, noise_spread)

        # IMPORTANT: Set the observation model on the environment so it generates noisy observations
        self._env.set_likelihood_dist(self._A)

        # Belief state: P(s), shape (n_states,)
        self._belief = None
        self._belief_idx = None  # Most likely state index

        # History tracking
        self._s_array = []  # True state indices
        self._o_array = []  # Observation indices
        self._b_array = []  # Belief state indices
        self._a_array = []  # Actions taken

        # Beta for information gain weighting
        self.beta = 1.0

    def _create_observation_model(self, noise_level: float,
                                   noisy_states: Optional[List[int]],
                                   noise_spread: float) -> np.ndarray:
        """Create the observation likelihood matrix A.

        Default: Identity matrix with small noise.
        Noisy states: Much higher noise (more uniform observations = higher entropy).

        The noise_spread parameter controls how noisy the designated states are:
        - noise_spread=1.0: noisy states have same noise as base (no effect)
        - noise_spread=3.0: noisy states are 3x noisier than base
        - noise_spread=5.0: noisy states are 5x noisier (very ambiguous)

        Args:
            noise_level: Base noise level for all states (probability of wrong obs)
            noisy_states: States with extra observation noise (higher entropy)
            noise_spread: Multiplier for noise in designated states

        Returns:
            A: Observation model, shape (n_obs, n_states)
        """
        N = self._n_states

        # Start with near-identity (most likely observe true state)
        A = np.eye(N) * (1 - noise_level)

        # Add small uniform noise
        A += noise_level / N

        # Make designated states noisier by increasing their uniform noise component
        # Higher noise_spread → observations more uniform → higher entropy
        if noisy_states is not None:
            noisy_noise = min(noise_level * noise_spread, 0.95)  # Cap at 95%
            for state_idx in noisy_states:
                # Reconstruct column with higher noise level
                A[:, state_idx] = noisy_noise / N  # uniform component
                A[state_idx, state_idx] += (1 - noisy_noise)  # identity component

        # Normalize columns (each column should sum to 1)
        A = A / A.sum(axis=0, keepdims=True)

        return A

    def set_observation_model(self, A: np.ndarray):
        """Set a custom observation model.

        Args:
            A: Observation model, shape (n_obs, n_states)
        """
        assert A.shape == (self._n_states, self._n_states), \
            f"A must have shape ({self._n_states}, {self._n_states})"
        self._A = A / A.sum(axis=0, keepdims=True)  # Ensure normalized

    @property
    def observation_model(self) -> np.ndarray:
        """Get observation model A."""
        return self._A

    @property
    def A(self) -> np.ndarray:
        """Alias for observation_model."""
        return self._A

    @property
    def state_space(self) -> GridStateSpace:
        return self._state_space

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def env(self) -> Any:
        return self._env

    @property
    def transition_matrix_shape(self) -> Tuple[int, ...]:
        N = self.n_states
        return (N, N, self._n_actions)

    @property
    def successor_matrix_shape(self) -> Tuple[int, ...]:
        N = self.n_states
        return (N, N)

    @property
    def belief(self) -> np.ndarray:
        """Get current belief state."""
        return self._belief

    @property
    def belief_idx(self) -> int:
        """Get index of most likely state in belief."""
        return self._belief_idx

    # ==================== Belief Operations ====================

    def _belief_to_idx(self, belief: np.ndarray) -> int:
        """Convert belief to most likely state index (MAP estimate).

        Uses argmax of belief, excluding wall states.
        """
        walls = set(self.get_wall_indices())

        # Mask wall states so they can't be selected
        masked = belief.copy()
        for w in walls:
            masked[w] = -1.0

        return int(np.argmax(masked))

    def update_belief(self, observation_idx: int, action: int) -> np.ndarray:
        """Update belief given observation and previous action.

        Args:
            observation_idx: Index of observation received
            action: Action taken before observation

        Returns:
            Updated belief state
        """
        # Predict: P(s_t | s_{t-1}, a) using transition model
        # This is done externally via B matrix

        # Update: P(s_t | o_t, s_{t-1}, a) using observation
        self._belief = infer_states(observation_idx, self._A, self._belief)
        self._belief_idx = self._belief_to_idx(self._belief)

        return self._belief

    # ==================== Environment Interaction ====================

    def reset(self, init_state: Optional[Any] = None) -> np.ndarray:
        """Reset environment and initialize belief.

        Args:
            init_state: Initial state as flat index or (x, y) tuple

        Returns:
            Initial belief state (one-hot or uniform if unobserved)
        """
        # Reset hidden state
        if init_state is not None:
            if isinstance(init_state, int):
                self._current_state = self._env.reset(init_state)
            else:
                idx = self.state_space.state_to_index(init_state)
                self._current_state = self._env.reset(idx)
        else:
            self._current_state = self._env.reset()

        # Initialize belief: uniform over non-wall states
        N = self._n_states
        wall_idx = self.get_wall_indices()
        allowed_idx = list(set(range(N)) - set(wall_idx))

        prior_belief = np.zeros(N)
        prior_belief[allowed_idx] = 1.0 / len(allowed_idx)

        # Get initial observation
        o_idx = self._env.get_obs_idx()

        # Update belief with first observation
        self._belief = infer_states(o_idx, self._A, prior_belief)
        self._belief_idx = self._belief_to_idx(self._belief)

        # Reset history
        s_idx = self._env.get_state_idx()
        self._s_array = [s_idx]
        self._o_array = [o_idx]
        self._b_array = [self._belief_idx]
        self._a_array = []

        return self._belief

    def step(self, action: int) -> np.ndarray:
        """Take action and return updated belief state.

        In a POMDP, the agent doesn't have access to the true state.
        Instead, it maintains a belief (probability distribution) over states,
        updated via Bayesian inference using the observation.

        The belief update follows:
        1. Predict: b'(s') = sum_s B(s'|s,a) * b(s)
        2. Update: b(s') = P(o|s') * b'(s') / P(o)

        Returns:
            Updated belief state (probability distribution over states)
        """
        # Take action in environment (updates hidden state - but agent doesn't see this)
        self._current_state = self._env.step(action)

        # Get noisy observation (this IS what the agent observes)
        o_idx = self._env.get_obs_idx()

        # Record true state (for evaluation only) and observation
        s_idx = self._env.get_state_idx()
        self._s_array.append(s_idx)
        self._o_array.append(o_idx)
        self._a_array.append(action)

        # Update belief using Bayes rule with the observation
        # Note: We use the environment's true B matrix for belief prediction
        # In practice, the agent would use its learned B matrix
        B_true = self._env.get_transition_dist()
        prior = B_true[:, :, action] @ self._belief
        self._belief = infer_states(o_idx, self._A, prior)
        self._belief_idx = self._belief_to_idx(self._belief)
        self._b_array.append(self._belief_idx)

        # Return belief (what the agent actually has access to)
        return self._belief

    def predict_belief(self, B: np.ndarray, action: int) -> np.ndarray:
        """Predict next belief using transition model (before observation).

        Args:
            B: Transition matrix, shape (N, N, n_actions)
            action: Action being taken

        Returns:
            Predicted belief (prior for next timestep)
        """
        return B[:, :, action] @ self._belief

    def step_with_belief_update(self, action: int, B: np.ndarray) -> Tuple[np.ndarray, int]:
        """Take action and fully update belief (convenience method).

        Args:
            action: Action to take
            B: Transition matrix

        Returns:
            Tuple of (updated belief, observation index)
        """
        # Predict
        prior = self.predict_belief(B, action)
        self._belief = prior

        # Take action (updates hidden state)
        self._current_state = self._env.step(action)

        # Get observation
        o_idx = self._env.get_obs_idx()
        s_idx = self._env.get_state_idx()

        # Update belief
        self._belief = infer_states(o_idx, self._A, prior)
        self._belief_idx = self._belief_to_idx(self._belief)

        # Record history
        self._s_array.append(s_idx)
        self._o_array.append(o_idx)
        self._b_array.append(self._belief_idx)
        self._a_array.append(action)

        return self._belief, o_idx

    def get_current_state(self) -> Tuple[int, int]:
        """Get believed current (x, y) position.

        In POMDP, the agent doesn't know its true state - it only has beliefs.
        This returns the position corresponding to the most likely state in the belief.
        """
        return self.state_space.index_to_state(self._belief_idx)

    def set_learning_mode(self, is_learning: bool):
        """Set whether the agent is in learning mode.

        In learning mode with use_true_state_for_learning=True, the agent
        can access true states for building transition/successor matrices.
        """
        self._in_learning_mode = is_learning

    def get_current_state_index(self) -> int:
        """Get current state index for the agent.

        The behavior depends on the POMDP mode:
        - Full POMDP (use_true_state_for_learning=False):
          Always returns belief state index
        - Hybrid (use_true_state_for_learning=True):
          Returns true state during learning, belief state during execution

        For the true state (debugging/evaluation only), use get_true_state_index().
        """
        if self._use_true_state_for_learning and self._in_learning_mode:
            return self._env.get_state_idx()
        return self._belief_idx

    def get_true_state_index(self) -> int:
        """Get true current state index (for debugging/evaluation only).

        WARNING: In a real POMDP, the agent should NOT have access to this.
        This is only for evaluation and debugging purposes.
        """
        return self._env.get_state_idx()

    def get_true_state(self) -> Tuple[int, int]:
        """Get true current (x, y) position (for debugging/evaluation only).

        WARNING: In a real POMDP, the agent should NOT have access to this.
        """
        return self._env.get_state_loc()

    def get_current_observation(self) -> int:
        """Get current observation index."""
        return self._env.get_obs_idx()

    def get_observation_entropy(self) -> np.ndarray:
        """Get entropy of observation model for each state.

        High entropy = more uncertain observations from that state.
        """
        return entropy(self._A)

    # ==================== Matrix Operations ====================

    def multiply_B_s(self, B: np.ndarray, state: np.ndarray, action: Optional[int]) -> np.ndarray:
        """Multiply transition matrix with state/belief vector."""
        if action is not None:
            return B[:, :, action] @ state
        else:
            result = np.zeros_like(state)
            for a in range(self._n_actions):
                result += B[:, :, a] @ state
            return result / self._n_actions

    def multiply_M_C(self, M: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Multiply successor matrix with preference vector."""
        return M @ C

    def compute_expected_free_energy(self, M: np.ndarray, C: np.ndarray,
                                     beta: float = 1.0) -> np.ndarray:
        """Compute expected free energy incorporating observation uncertainty.

        G = pragmatic_value - beta * epistemic_value

        Args:
            M: Successor matrix
            C: Preference vector
            beta: Weight for epistemic value (observation entropy)

        Returns:
            Expected free energy for each state
        """
        # Pragmatic value: expected reward
        pragmatic = M @ C

        # Epistemic value: expected observation entropy (information gain)
        epistemic = M @ entropy(self._A)

        # Combined (note: higher entropy = more information gain = good)
        return pragmatic - beta * epistemic

    # ==================== Transition Matrix ====================

    def get_transition_matrix(self) -> np.ndarray:
        """Get true transition matrix from environment."""
        return self._env.get_transition_dist()

    def normalize_transition_matrix(self, B: np.ndarray, goal_states: List[int] = None) -> np.ndarray:
        """Normalize transition matrix."""
        N = self.n_states

        for col in range(N):
            for action in range(self._n_actions):
                col_sum = np.sum(B[:, col, action])
                if col_sum == 0:
                    B[col, col, action] = 1

        B = B / B.sum(axis=0, keepdims=True)

        if goal_states:
            for gs in goal_states:
                B[:, gs, :] = 0
                B[gs, gs, :] = 1

        return B

    def learn_transition_from_beliefs(self, action: int, prior_belief: np.ndarray,
                                      posterior_belief: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Update transition matrix based on belief change.

        Since we don't observe true states, we update B based on beliefs.

        Args:
            action: Action taken
            prior_belief: Belief before action
            posterior_belief: Belief after action and observation
            B: Current transition matrix estimate

        Returns:
            Updated transition matrix
        """
        # Update B column for prior belief's most likely state
        prior_idx = self._belief_to_idx(prior_belief)
        B[:, prior_idx, action] += posterior_belief
        return B

    # ==================== Goal/Reward ====================

    def create_goal_prior(self, goal_states: List[int], reward: float = 100.0,
                          default_cost: float = -0.1) -> np.ndarray:
        """Create goal preference vector."""
        C = np.ones(self.n_states) * default_cost
        for gs in goal_states:
            C[gs] = reward
        return C

    def create_goal_prior_with_info_gain(self, goal_states: List[int],
                                          reward: float = 100.0,
                                          default_cost: float = -0.1,
                                          beta: float = 1.0) -> np.ndarray:
        """Create goal preference incorporating observation entropy.

        States with noisier observations are penalized (harder to localize).
        """
        C = np.ones(self.n_states) * default_cost
        for gs in goal_states:
            C[gs] = reward

        # Subtract observation entropy (states with high obs entropy are harder)
        C = C - beta * entropy(self._A)

        return C

    def get_goal_states(self, goal_spec: Any) -> List[int]:
        """Convert goal specification to state indices."""
        if isinstance(goal_spec, int):
            return [goal_spec]
        elif isinstance(goal_spec, (tuple, list)):
            return [self.state_space.state_to_index(goal_spec)]
        else:
            raise ValueError(f"Invalid goal specification: {goal_spec}")

    # ==================== Walls/Obstacles ====================

    def get_wall_indices(self) -> List[int]:
        """Get wall state indices."""
        walls = self._env.get_walls()
        return [self.state_space.state_to_index(w) for w in walls]

    # ==================== History Access ====================

    @property
    def state_history(self) -> List[int]:
        """Get history of true states (for evaluation)."""
        return self._s_array

    @property
    def observation_history(self) -> List[int]:
        """Get history of observations."""
        return self._o_array

    @property
    def belief_history(self) -> List[int]:
        """Get history of belief state indices."""
        return self._b_array

    @property
    def action_history(self) -> List[int]:
        """Get history of actions."""
        return self._a_array

    # ==================== Visualization ====================

    def render_state(self, state_index: int) -> Tuple[int, int]:
        """Get (x, y) coordinates for visualization."""
        return self.state_space.index_to_state(state_index)

    def get_state_label(self, state_index: int) -> str:
        """Get human-readable state label."""
        x, y = self.state_space.index_to_state(state_index)
        return f"({x},{y})"

    def onehot_to_index(self, one_hot: np.ndarray) -> int:
        """Convert one-hot or belief to index.

        For POMDP, the input may be a belief distribution (normalized probability
        vector) rather than a true one-hot. We detect this by checking if the
        maximum value is close to 1 (true one-hot) or spread out (belief).

        Args:
            one_hot: Either a one-hot vector or a belief distribution

        Returns:
            State index - argmax for one-hot, sampled for beliefs
        """
        max_val = np.max(one_hot)

        # If max value is close to 1, it's effectively one-hot - use argmax
        if max_val > 0.95:
            return np.argmax(one_hot)

        # Otherwise it's a spread belief distribution - sample or use mode
        return self._belief_to_idx(one_hot)
