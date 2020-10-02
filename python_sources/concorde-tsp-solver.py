#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sympy import isprime
from concorde.tsp import TSPSolver
import matplotlib.pyplot as plt


# In[ ]:


cities = pd.read_csv('../input/cities.csv')


# In[ ]:


display(cities.describe())
cities.head()


# In[ ]:


cities_len = len(cities)
print("len: ", cities_len)
cities_xy = np.stack((cities.X.values, cities.Y.values), axis=1)
cities_xy[0:10]


# In[ ]:


def get_score(path):
    xy = cities_xy[path]
    values = np.linalg.norm(xy-np.roll(xy, -1, axis=0), axis=1)
    prime_value = (values[9::10] * [int(isprime(i)==False)*0.1 for i in path[9::10]]).sum()
    return values.sum() + prime_value


# In[ ]:


solver = TSPSolver.from_data(cities.X,cities.Y,norm="EUC_2D")

best_score = 10**10
best_path = None
for i in range(2):
    tour_data = solver.solve(time_bound = 60.0)
    path = np.append(tour_data.tour,[0])
    score = get_score(path)
    if score < best_score:
        best_score = score
        best_path = path
print('Best Score: ', best_score)
print(best_path)


# In[ ]:


pd.DataFrame({'Path': best_path}).to_csv('submission.csv', index=False)


# In[ ]:


plt.figure(figsize=(32,18))
plt.plot(cities.X[path], cities.Y[path])
plt.show()

