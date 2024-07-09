import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from model import run_env, params, env, learner

# Ensure that the theme is set
sns.set_theme()


# RNG setup
rng = np.random.default_rng(params.seed)

# Create the figure folder if it doesn't exist
params.savefig_folder.mkdir(parents=True, exist_ok=True)

# Creating the Q-table
params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
print(f"Action size: {params.action_size}")
print(f"State size: {params.state_size}")


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


def plot_cumulative_rewards(rewards, params=params):
    """Plot the cumulative reward over time."""
    cum_rewards = rewards.cumsum(axis=0)
    mean_cum_rewards = cum_rewards.mean(axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(mean_cum_rewards)
    plt.title("Cumulative Rewards Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.savefig(params.savefig_folder / "cummulative_rewards.png")
    plt.show()


def plot_exploration_exploitation(exploration_rates, params=params):
    """Plot the exploration-exploitation trade-off over time."""
    mean_exploration_rates = exploration_rates.mean(axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(mean_exploration_rates)
    plt.title("Exploration-Exploitation Trade-off")
    plt.xlabel("Episodes")
    plt.ylabel("Exploration Rate")
    plt.savefig(params.savefig_folder / "exploration-exploitation.png")
    plt.show()


def plot_learning_curve(steps, params=params):
    """Plot the learning curve (performance of the agent over time)."""
    mean_steps = steps.mean(axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(mean_steps)
    plt.title("Learning Curve (Steps per Episode)")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.savefig(params.savefig_folder / "learning-curve.png")
    plt.show()


def evaluate_policy(learner, env, params=params):
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
rewards, steps, episodes, qtables, all_states, all_actions, exploration_rates = (
    run_env()
)


# Plotting results
plot_convergence(qtables, params)
plot_cumulative_rewards(rewards, params)
plot_exploration_exploitation(exploration_rates, params)
plot_learning_curve(steps, params)

# Evaluating policy
policy_reward = evaluate_policy(learner, env, params)
print(f"Total reward obtained by the policy: {policy_reward}")
