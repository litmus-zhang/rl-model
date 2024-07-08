from motor import FourWheelMotor as Motor

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

env = gym.make("CarRacing-v2")
wrapped_end = FlattenObservation(env)
print(wrapped_end.observation_space.shape, env.observation_space.shape)


def move_robot():
    wheels = Motor(0, {11, 12, 13, 15})

    wheels.move_forward()
    print(wheels)
    print(wheels.get_pins())
    print(wheels.get_speed())
