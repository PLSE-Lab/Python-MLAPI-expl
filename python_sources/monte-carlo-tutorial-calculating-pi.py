#!/usr/bin/env python
# coding: utf-8

# A simple tutorial for doing a Monte Carlo simulation using Python.   
# 
# From geometry, we know the ratio of the area of an circle inscribed in a square to that square is pi/4.  For our Monte Carlo simulation, we will generate a bunch of points in a square and determine which ones are in the circle and which ones are not.  The ratio of the points inside circle to the total number of points is pi/4.

# First import some libraries,

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
    
from numpy.random import random
from numpy import pi

import os
print(os.listdir("../input"))


# Next, we're going to set up a function in Python to: 
# 1. Generate points within a square
# 2. Determine which points are also in the circle
# 3. Plot the points to visualize those points
# 4. Plot the estimate for pi versus the iteration number, to see how quickly we converge on the true value of pi (spoiler: it's noisy)

# **Drawing a square and a circle**
# 
# We want our to be within a 2x2 grid centered at the origin.  The corners of the square will be at (1,1), (1,-1), (-1,-1), and (-1,1).  The circle will be a unit circle with radius 1.
# 
# For the square, we can generate a list for X and Y values using the corner points.
# 
# The circle we can generate using our knowledge of trigonometry.  The X value of a particular point will be cosine of the angle, and the Y value will be its sine.  The total angle of the circle is 2pi radians, and a for loop will allow a quick calculation of the X and Y values of the circle ever degree.
# 
# **Initial Values**
# 
# The points inside the circle, outside the circle, what iteration we're on, and current value for pi are all empty lists, and we can append those lists with new values later.  insideCounter is the number of points that appear within the circle, which starts at 0 since we have no points.
# 
# **Generating points in the square**
# 
# Numpy's random() function will generate a uniform random number between 0 and 1.  We want a number between -1 and 1, so we must first transform it.  Subtracting 0.5 from random() gives a value between -0.5 and 0.5, and multiplying that by 2 gives a number between -1 and 1.  We will do this twice, once for x, once for y.
# 
# Pythagoras tells us the distance from the origin (r) of our randomly generated points is the square root of the sum of their squares.  If this r value is less than 1, we will add 1 to our insideCounter.  Points inside the circle get appended to the lists for insideX and insideY, and points outside the circle get appended to the lists outsideX and outsideY.  Regardless of whether the new points are inside or outside the circle, we append the current value of pi.
# 
# Once we're done iterating, we can calculate the final value of pi, and its error versus the true value.
# 
# **Drawing the plots**
# 
# We want two plots, one is the points with relation to the square and circle, the other the estimate of pi over a number of iterations.

# In[ ]:


def MonteCarloPi(maxIterations):
#Draw a square and a circle to frame out simulation
    squareX = [1,-1,-1,1,1]
    squareY = [1,1,-1,-1,1]
    circleX,circleY = [],[]
    
    for i in range(361):
        circleX.append(np.cos(np.pi*i/180))
        circleY.append(np.sin(np.pi*i/180))

#Start keeping track of values we're interested in
    insideX,insideY,outsideX,outsideY,Iteration,CurrentPi = [],[],[],[],[],[]
    insideCounter = 0

#Generate a bunch of values of x and y between -1 and 1, then assess their combined radius on an xy plane
    for i in range(maxIterations):
        x = 2*(random()-0.5)
        y = 2*(random()-0.5)
        r = np.sqrt(x**2+y**2)
        Iteration.append(i)
        if r <= 1:
            insideCounter +=1
            insideX.append(x)
            insideY.append(y)
        else:
            outsideX.append(x)
            outsideY.append(y)
        CurrentPi.append(4*insideCounter/(i+1))

    piValue = 4*insideCounter / maxIterations
    piError = round(100*((piValue-pi)/pi),4)

#Draw a 2D plot of where our iterations landed compared to the square and circle
    rcParams['figure.figsize'] = 5, 5
    plt.plot(squareX,squareY,color='#000000')
    plt.plot(circleX,circleY,color='#0000CC')
    plt.scatter(insideX,insideY,color='#00CC00',marker=".")
    plt.scatter(outsideX,outsideY,color='#CC0000',marker=".")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
#Draw a psuedo-time series plot of current estimate of pi vs. iteration number
    plt.plot(Iteration,CurrentPi,color='#009900')
    plt.axhline(y=pi,color='#0F0F0F',ls='--')
    plt.axis([0,maxIterations,0,4.1])
    plt.xlabel('Iteration Number')
    plt.ylabel('Estimate for pi')
    plt.show()

#print out our final estimate and how it compares to the true value
    print('\n' + f'Pi is approximately {piValue}\n')
    print(f'This is {piError}% off the true value.\n')


#  Now with the code complete, we can run the function for a desired number of iterations.  The more iterations run, the better the estimate.  Here's how it looks with 500 and 50,000 iterations.

# In[ ]:


MonteCarloPi(500)


# In[ ]:


MonteCarloPi(50000)


# In[ ]:




