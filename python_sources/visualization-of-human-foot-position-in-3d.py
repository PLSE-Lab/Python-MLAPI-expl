#!/usr/bin/env python
# coding: utf-8

# **Let's try the visualize the gyro sensor data to see foot position.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Let's import libraries and the data.**

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('/kaggle/input/human-gait-phase-dataset/data/GP1_0.6_marker.csv')
data.head()


# **I choose LRFCC sensor data to visualize with both right and left foot.**

# In[ ]:


timer = np.arange(0,12000)
plt.scatter(timer,data['L_FCC_x'],c='b')
plt.scatter(timer,data['R_FCC_x'],c='orange')
plt.title('x-axis marker position change due to time')
plt.xlabel('time')
plt.ylabel('x-position')
plt.grid()
plt.show()


# **Let's look y axis of LRFM1 sensor data to see diffrent variations of sensor data.**

# In[ ]:


plt.scatter(timer,data['L_FM1_y'],c='b')
plt.scatter(timer,data['R_FM1_y'],c='r')
plt.title('y axis position change due to time')
plt.xlabel('time')
plt.ylabel('y-position')
plt.grid()
plt.show()


# **Let's look z axis change of LRFM2 sensor.**
# 

# In[ ]:


plt.scatter(timer,data['L_FM2_z'],c='b')
plt.scatter(timer,data['R_FM2_z'],c ='red')
plt.xlabel('time')
plt.ylabel('z-position')
plt.title('z axis position change due to time')
plt.grid()
plt.show()


# Let's visualize all axis in 3D graph.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data['L_FCC_x'],data['L_FCC_y'], data['L_FCC_z'], color='red', marker='o')
ax.scatter(data['R_FCC_x'],data['R_FCC_y'], data['R_FCC_z'], color='blue', marker='.')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title("3D Sag-Sol ayak goruntuleme")
plt.show()


# **As you see, the figure above show us how a human's right and left foot move.(in 0,6  m/s,GP1)**
