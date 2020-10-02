#!/usr/bin/env python
# coding: utf-8

# ## Intro
# 
# Bayesian inference requires 3 things
# - a priori model
# - a generative model
# - and ... data
# 
# ## Example
# 
# Your marketing team surveyed 60 Danes and you got 16 positive answers. That's 26%, but you want to know, what confidence this result gives you, when you project it to the whole Dane market.

# In[ ]:


import pandas as pd
import numpy as np

prior_sample_size = 60
prior_subscribers = 16

def simulate_survey(sample_size, percent_subscribes):
    """simulation as a data pipe: this function returns
       the simulation results together with the input,
       randomly chosen from the input distribution"""
    return (
        sample_size,
        percent_subscribes,
        # assuming 'percent_subscribes' popularity on the Dane
        # market, simulates a survey by rolling a dice 'sample_size'
        # times, and counting the positive answers
        sum(
            [percent_subscribes >= np.random.randint(0,100) 
                 for _ in range(sample_size)]
        )
    )

# turning the function numpy-friendly
vectorized_simulation = np.vectorize(simulate_survey)
repetitions = 100000

# this is the post-simulation dataset of our popularity
posterior = pd.DataFrame(
    list(vectorized_simulation(
        # all surveys are with 60 people
        np.full(repetitions, prior_sample_size),
        # this is our prior model: discrete uniform (0..100)
        np.random.randint(0, 100, repetitions))),
    ).T


# ## What just happened?!
# 
# 1. We set up an apriori model
#     - we said, that the reception can be anything between 0 and 100%
#     - this is represented in that `randint(0,100)` uses uniform distribution
# 2. We wrote a generative model
#     - it takes a sample size
#     - a 'p' probability describing, how likely the subscription is
#     - and rolls the 'p' dice 'sample size' times
# 3. We vectorized the generative model, so that it runs faster
# 4. We ran it 100k times (Monte Carlo)
# 5. We loaded the MC simulation results into a Pandas DataFrame
# 
# ## Correct, but not optimal
# 
# The computation we used in the simulation is called *Approximate Bayesian Calculation (ABC)*. It is simple to understand, but computationally expensive, because it is a naive, bruteforce algorithm, and converges to the posterior distribuion really slowly.
# 
# The good thing is that the [less computationally intensive algorithms](https://towardsdatascience.com/from-scratch-bayesian-inference-markov-chain-monte-carlo-and-metropolis-hastings-in-python-ef21a29e25a) yield exactly the same result, just more efficiently, so the concept stays.
# 
# 

# ## Answers

# In[ ]:


ax = posterior[posterior[2] == 16][1].plot.hist()
ax = posterior[posterior[2] == 16][1].plot.kde(
    bw_method=.7, secondary_y=True, ax=ax)


# In[ ]:


# Statistics about the result
posterior[posterior[2] == 16][1].describe(percentiles=[.1,.25,.75,.9])


# Here is the answer to our question. A probability distribution that answers the question:
# >"What reception is likely on the Dane market, if on my survey 16 people subscribed out of 60?"
# 
# We already knew the 26% answer $(60/16)$, but we did not know the certainity. Now we know, and can say:
# 
# > We can be 95% sure that our reception on the Dane market is $26\% \pm 11\%$.
# 
# 
