import gym
import numpy as np

env = gym.make()

observation = env.reset()
prediction_column = observation.target.columns[-1]
previous_instrument_sums   = {}
previous_instrument_counts = {}

while True:
    features = observation.features
    target = observation.target
    previous_target = observation.previous_target

    if previous_target is not None:
        for ix in previous_target.index:
            instrument_id = previous_target.get_value(ix, "id")
            if instrument_id not in previous_instrument_counts:
                previous_instrument_counts[instrument_id]=0
                previous_instrument_sums[instrument_id]=0
            previous_instrument_counts[instrument_id] += 1
            previous_instrument_sums[instrument_id] += previous_target.get_value(ix, "y")

    for ix in target.index:
        instrument_id = target.get_value(ix, "id")
        if instrument_id in previous_instrument_sums:
            previous_mean = (previous_instrument_sums[instrument_id]/
                             previous_instrument_counts[instrument_id])
            target.set_value(ix, "y", previous_mean)
        else:
            target.set_value(ix, "y", 0)    
    
    observation, reward, done, info = env.step(target)

    if done:
        print("Public score: {}".format(info["public_score"]))
        break