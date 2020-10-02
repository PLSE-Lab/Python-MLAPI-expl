#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Read in the cities data

# In[ ]:


cities = pd.read_csv("../input/cities.csv")


# In[ ]:


cities.head()


# ### Define a function to determine which city ID's are prime

# In[ ]:


def is_prime(n):
    """Determines if a positive integer is prime."""

    if n > 2:
        i = 2
        while i ** 2 <= n:
            if n % i:
                i += 1
            else:
                return False
    elif n != 2:
        return False
    return True


# In[ ]:


#Create a column within the cities dataframe to flag prime cities
cities['is_prime'] = cities.CityId.apply(is_prime)


# ### Visualize the city X & Y values

# In[ ]:


fig = plt.figure(figsize=(18,18))
plt.scatter(cities.X, cities.Y, c=cities['is_prime'], marker=".", alpha=.5);


# In[ ]:




