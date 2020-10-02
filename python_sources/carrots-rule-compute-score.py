#!/usr/bin/env python
# coding: utf-8

# # Carrots rule - Compute score
# This kernel is designed to propose a computation of the score taking into account the carrots rule

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')


# ## Simplified map for evaluation purposes

# ### Dataframe of the cities
# Coordinates of the cities and a column specifying wether the CityId is a prime number or not

# In[ ]:


from sympy.ntheory.primetest import isprime

cities = pd.DataFrame({'CityId': range(11), 'X' : range(11), 'Y' : [0]*11})
cities.loc[11] = [11, 9, 1]
cities.loc[12] = [12, 10, 1]
cities = cities.drop(cities.index[[0]])

cities['IsPrime'] = cities['CityId'].apply(lambda val: isprime(val))
cities = cities.reset_index(drop=True)
display(cities)


# ### Map of the cities
# In red the ones with prime CityId

# In[ ]:


plt.figure(figsize=(15, 2))
sns.regplot(data=cities, x="X", y="Y", fit_reg=False, marker="+", color="blue")
p1=sns.regplot(data=cities, x="X", y="Y", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':400})

# add annotations one by one with a loop
for line in range(0, cities.shape[0]):
    if cities.IsPrime[line]:
        color = 'red'
    else:
        color = 'black'
    p1.text(cities.X[line]+0.2, cities.Y[line], cities.CityId[line], 
             horizontalalignment='left', size='medium', color=color, weight='semibold')


# ## Paths
# Reaching the city 12 in 10 steps.
# ### Path with 10th step coming from a regular city (10)

# In[ ]:


path_10th_step_regular = list(range(1, 13))
path_10th_step_regular.remove(11)
display(path_10th_step_regular)


# ### Path with 10th step coming from a prime city (11)

# In[ ]:


path_10th_step_prime = list(range(1, 13))
path_10th_step_prime.remove(10)
display(path_10th_step_prime)


# ## Scoring function

# In[ ]:


from scipy.spatial import distance

def compute_score(cities, path):
    score = np.float64()
    score = 0
    for step in range(1,len(path)):
        cityId_start = path[step-1]
        cityId_end   = path[step]
        
        city_start = cities.loc[cities['CityId'] == cityId_start]
        city_end   = cities.loc[cities['CityId'] == cityId_end]
        
        city_start_coord = np.array([city_start.iloc[0]['X'], city_start.iloc[0]['Y']])
        city_end_coord   = np.array([city_end.iloc[0]['X'], city_end.iloc[0]['Y']])

        distance_step = distance.euclidean(city_start_coord, city_end_coord)

        if step%10 == 0 and  not city_start.iloc[0]['IsPrime']:
            distance_step = 1.1 * distance_step

        score = score + distance_step

    return score


# ## Tests

# ### Path with 10th step coming from a regular city (10)

# In[ ]:


compute_score(cities, path_10th_step_regular)


# ### Path with 10th step coming from a prime city (11)

# In[ ]:


compute_score(cities, path_10th_step_prime)

