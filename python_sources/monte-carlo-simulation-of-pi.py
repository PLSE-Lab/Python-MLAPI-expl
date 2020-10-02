#!/usr/bin/env python
# coding: utf-8

# # What is Monte Carlo?
# Monte Carlo is a statistical technique which can be used to simulate areas. We will first start with an example on how to calculate the value of $\pi$ (~3.14159). Then we will use the same idea to calculate the definite integral of a function.
# 
# ## Formulas necessary for calculating PI
# 1. $Area_{square} = (length_{squareSide})^2$
# 2. $Area_{circle} = \pi * r^2$ where $r$ is the radius of the circle.
# 3. 
# 
# ## Idea Behind Monte Carlo
# You can think of Monte Carlo as "throwing darts" at a dartboard (but for Monte Carlo you need to be really drunk becuase the darts must hit the dartboard in completely random locations). Take the following dart board which is a circle inside a square.

# In[28]:


import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = plt.axes()

ax.set_aspect(1)
theta = np.linspace(-np.pi, np.pi, 200)
plt.plot(np.sin(theta), np.cos(theta))

plt.show()


# Now lets randomly throw some darts at the dart board
