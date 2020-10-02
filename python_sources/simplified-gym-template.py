import gym
import numpy as np

env = gym.make("Gaussian_c1-v0")

observation = env.reset()
print("Train has {} rows".format(len(observation.train)))
prediction_column_name = observation.target.columns[-1]
print('Prediction column name: "{}"'.format(prediction_column_name))

step = 0

while True:
    target = observation.target
    step += 1
    if step % 100 == 0:
        print("Step #{}".format(step))

    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break