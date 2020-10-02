#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## A Tutorial on Principal Component Analysis
# 
# This is a notebopok version of the [tutorial](https://www.cs.cmu.edu/~elaw/papers/pca.pdf). I think one should read that! 
# 

# TODO
# * Create data for the spring example described 
# * Create the animation for the above generated data.
# * Explain the data
# * Try different explaining the different sections of the tutoral 
# 

# I would like to start by saying, **Principal Component Analysis**(PCA) is widely used and poorly understood tool in the aresnal of many data scientists. It is one of the most valuable result of the applied linear alzebra.
# 
# Lets take the example of motion of mass attached to a massless spring. The idea here is to find what is the primary direction of motion and after that possibly able to model the dynamics of the above system *(which we will do in another tutorial)*
# ![experimental setup](https://www.projectrhea.org/rhea/images/9/9c/Spring.png)
# 
# **I have avoided units for the sake of simplicity, advanced readers can assume approriate units.**
# 
# We know the above system is $x(t) = x_{0}sin(\omega t)$. Which is the primiary motion. 
# We are planning to do the following things.
# * assume the above is ture and use three cameras each of which are pointing to x, y and z directons. There can be in place rotations. see **II. MOTIVATION: A TOY EXAMPLE** for more details 
# * we will assume ceratin in plane rotation for all these cameras and add certain gaussian noise.
# * We will vary the standard devation of gaussian noise and check how well PCA is able to identify the primary motion. 
# * As we dont know how many measurements to take in order to find the motion of the block, we have choosen three similar to the paper.
# * Lets assume camera always sees along its Z axis perpendicular to lense. Position of the camera lens
#     * Position of block at equilibrium in the world cordinates is $(100, 120, 60)$ and motion of the block is along $ Z-axis $ with amplitude 20. So the position of block can be between 40 to 80 along $ Z-axis $
#     * Camera A
#         * Camera Postion $(0,0,0))$ 
#         * $Z axis$ of camera parallel to $X axis$ of the world. $Z_{camera}  {\parallel}  X_{world}$ 
#         * $X axis$ of camera makes $\pi/6$ angle with $Y axis $ of the world.
#         * matric of rotation from world to camera \begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix}
#         

# In[ ]:




