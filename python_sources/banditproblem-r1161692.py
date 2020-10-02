# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import plotly.graph_objects as go

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
# Any results you write to the current directory are saved as output.
def selectFunc(Q,epsilon):
    if (random.random()>epsilon):
        return np.argmax(Q)
    else:
        return np.random.randint(0,len(Q)-1)

def bandit(A,q):
    R = np.random.normal(q[A],1)
    return  R

num_exps = 10
num_timesteps = 1000
epsilon = 0.01
rhist = np.zeros(num_timesteps)
ahist = np.zeros(num_timesteps)
for i in range(num_exps):
    Q  = np.zeros(10)
    N  = np.zeros(10)
    q = np.array([np.random.standard_normal() for i in range(10)])
    optimal_action = np.argmax(q)
    reward_history = []
    optimal_action_history = []
    for i in range(num_timesteps):
    #     A = Q.index(max(Q))
        A = selectFunc(Q,0.01)
        R = bandit(A,q)
        reward_history.append(R)
        if (A==optimal_action):
            optimal_action_history.append(1)
        else:
            optimal_action_history.append(0)
        N[A] +=1
        Q[A] += 1/N[A]*(R-Q[A])
    rhist += np.array(reward_history)
    ahist += np.array(optimal_action_history)
rhist /= np.float(num_exps)
ahist /= np.float(num_exps)
time = np.array([i for i in range(num_timesteps)])
fig = go.Figure(data=go.Scatter(x=time,y=ahist))
fig