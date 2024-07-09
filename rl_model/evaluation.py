import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from model import *

# Ensure that the theme is set
sns.set_theme()

class Params(NamedTuple):
    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    map_size: int
    seed: int
    is_slippery: bool
    n_runs: int
    action_size: int
    state_size: int
    proba_frozen: float
    savefig_folder: Path

# Define the parameters
params = Params(
    total_episodes=1000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder=Path("./assets/plots/"),
)

# RNG setup
rng = np.random.default_rng(params.seed)

# Create the figure folder if it doesn't exist
params.savefig_folder.mkdir(parents=True, exist_ok=True)

# Environment Setup
env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    ),
)

# Creating the Q-table
params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
print(f"Action size: {params.action_size}")
print(f"State size: {params.state_size}")

# Define the Qlearning and EpsilonGreedy classes (provided previously)

# Observing the environment
learner = Qlearning(
    learning_rate=params.learning_rate,
    gamma=params.gamma,
    state_size=params.state_size,
    action_size=params.action_size,
)
explorer = EpsilonGreedy(
    epsilon=params.epsilon,
)

def run_env():
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []
    exploration_rates = np.zeros((params.total_episodes, params.n_runs))

    for run in range(params.n_runs):
        learner.reset_qtable()
        exploration_rate = []

        for episode in tqdm(episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False):
            state = env.reset(seed=params.seed)[0]
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                all_states.append(state)
                all_actions.append(action)

                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_rewards += reward
                step += 1
                state = new_state

            exploration_rates[episode, run] = explorer.epsilon
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions, exploration_rates

def plot_convergence(qtables, params):
    """Plot the convergence of Q-values over time."""
    qtable_mean = qtables.mean(axis=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(qtable_mean, annot=True, cmap="coolwarm", cbar=True)
    plt.title("Convergence of Q-values")
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.savefig(params.savefig_folder / "convergence_q_values.png")
    plt.show()

def plot_cumulative_rewards(rewards, params):
    """Plot the cumulative reward over time."""
    cum_rewards = rewards.cumsum(axis=0)
    mean_cum_rewards = cum_rewards.mean(axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(mean_cum_rewards)
    plt.title("Cumulative Rewards Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.savefig(params.savefig_folder / "cumulative_rewards.png")
    plt.show()

def plot_exploration_exploitation(exploration_rates, params):
    """Plot the exploration-exploitation trade-off over time."""
    mean_exploration_rates = exploration_rates.mean(axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(mean_exploration_rates)
    plt.title("Exploration-Exploitation Trade-off")
    plt.xlabel("Episodes")
    plt.ylabel("Exploration Rate")
    plt.savefig(params.savefig_folder / "exploration_exploitation.png")
    plt.show()

def plot_learning_curve(steps, params):
    """Plot the learning curve (performance of the agent over time)."""
    mean_steps = steps.mean(axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(mean_steps)
    plt.title("Learning Curve (Steps per Episode)")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.savefig(params.savefig_folder / "learning_curve.png")
    plt.show()

def evaluate_policy(learner, env, params):
    """Evaluate the policy in an episodic MDP."""
    state = env.reset(seed=params.seed)[0]
    done = False
    total_rewards = 0

    while not done:
        action = np.argmax(learner.qtable[state, :])
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_rewards += reward
        state = new_state

    return total_rewards

# Running the environment
rewards, steps, episodes, qtables, all_states, all_actions, exploration_rates = run_env()

# Plotting results
plot_convergence(qtables, params)
plot_cumulative_rewards(rewards, params)
plot_exploration_exploitation(exploration_rates, params)
plot_learning_curve(steps, params)

# Evaluating policy
policy_reward = evaluate_policy(learner, env, params)
print(f"Total reward obtained by the policy: {policy_reward}")