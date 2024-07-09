
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

env = gym.make("CarRacing-v2")
wrapped_end = FlattenObservation(env)
print(wrapped_end.observation_space.shape, env.observation_space.shape)
