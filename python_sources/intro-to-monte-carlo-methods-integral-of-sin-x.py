#!/usr/bin/env python
# coding: utf-8

# **Using Monte Carlo method to calculate the integral of Sin(x) from 0 to Pi**
# 
# To achieve this, random points will be generated in the cartesian space within a rectangle of height 1 and width Pi (or 3.141592 aprox.) All the random points falling below the curve Sin(x) from 0 to Pi will be counted. The area under the curve (which is the value of the integral) is then the fraction of the points below the curve times the area of the enclosing rectangle (whose area is Pi).

# The expected value of the integral is 2.
# ![](http://)![sin.PNG](attachment:sin.PNG)

# In[ ]:


from random import random
from random import uniform
from math import sin
from math import pi

# Number of random points to be generated
N=1000000


# Random points are generated in the rectangle, and tested if they fall within the area of interest. 
# If so, they are counted.

# ![](http://)![sin.jpg](attachment:sin.jpg)

# In[ ]:


count=0
for i in range(N):
    point=(uniform(0,pi), random())
    if point[1] < sin(point[0]):
        count+=1
        
answer=(float(count)/float(N))*pi
print(answer)


# The first question you should be asking is, 'Why would anyone do this?' If the program has an equation for the shape, shouldn't it be possible to find the area by numerically integrating? At higher dimensions, though, direct numerical integration becomes more dificult. The number of function calculations for numeric integration grows exponentially with dimension. Monte Carlo integrations also increase in dificulty with dimension, but the increase is linear.
