#!/usr/bin/env python
# coding: utf-8

# # DATA VISUALIZATION

# **Data visualization is the graphic representation of data. It involves producing images that communicate relationships among the represented data to viewers of the images. This communication is achieved through the use of a systematic mapping between graphic marks and data values in the creation of the visualization.(def taken from - wikipedia)**

# ![](https://i.pcmag.com/imagery/articles/02Xkt5sp3fVl5rGUtk3DXMi-7.fit_scale.size_2698x1517.v1569485349.jpg)
# 
# https://i.pcmag.com/imagery/articles/02Xkt5sp3fVl5rGUtk3DXMi-7.fit_scale.size_2698x1517.v1569485349.jpg

# Let me know if you have anything to ask. Do Upvote the story if you liked it.

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


# # Common Question :
# * What is data visualization ?
# * why is it important?
# 
# ## Basic answer is  Data visualization is the way of represent the data into visualization format i.e like graph,pie chart,hist , boxplot etc..
# 
# This is one of the best way to understand the data very well manner i.e 
# * Understand the pattern between variables/dataset/feature.
# * Communicate the information clearly and efficiently.
# * This is the basic and important steps for data analysis.
# * Communicate the data with images.
# 
# 
# 

# ### In this exercise we will use matplotlib library to visualize the Data and get important information from the data.

# # Matplotlib

# Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
# It was created by John Hunter. He created it to try to replicate MatLab's (another programming language) plotting capabilities in Python.
# It is an excellent 2D and 3D graphics library for generating scientific figures. 
# Some Important feature of  Matplotlib are:
# * easy to undersatnd and get started for simple plots
# * high-quality visualization output in many formats
# * given privilege to control of every element in a figure
# * it support the custom labels and texts
# 
# Matplotlib allows to create reproducible figures programmatically. Let's explore how to use it! , to explore more go to the official Matplotlib web page: http://matplotlib.org/

# In[ ]:


# Import the matplotlib.pyplot module 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
x= np.linspace(0,90, 5000)
y = (x ** 4)/2


# In[ ]:


print(x)
print("*" * 50)
print(y)


# # Matplotlib command
#  plt.plot(*args, **kwargs)
# Plot y versus x as lines and/or markers.

# In[ ]:


plt.plot(x,y,'*')
plt.xlabel('X data')
plt.ylabel('Y data')
plt.title('Ploating data X vs Y')


# ### multiple plot in same frame

# In[ ]:


plt.subplot(1,5,1)
plt.plot(x,y,'r-')
plt.subplots_adjust(right=2)
plt.subplot(1,5,2)
plt.plot(y,x,'g-')
plt.subplot(1,5,3)
plt.plot(x**5,y**2,'g-')
plt.subplot(1,5,4)
plt.plot(y**6,x,'b-')
plt.subplot(1,5,5)
plt.plot(y*x,x,'g-')


# In[ ]:


#Matplotlib having object oriented api , we can instantiate figure object and then call methods or attributes from that object

fig = plt.figure()
axes = fig.add_axes([1,1,1,1])
axes.plot(x,y*2)
axes.set_xlabel('x axes')
axes.set_ylabel('y axes')


# advantage of object oriented is that  have full control of where the plot axes are placed, and we can easily add more than one axis to the figure

# In[ ]:


fig = plt.figure()
axes1 = fig.add_axes([.1,.1,.8,.8])
axes2 = fig.add_axes([.3,.5,.3,.3])
axes1.plot(x,y**2,'g')
axes2.plot(y,x**2,'b')
axes1.set_xlabel('x axes')
axes1.set_ylabel('y axes')
axes2.set_xlabel('y axes')
axes2.set_ylabel('x axes')
axes1.set_title('x vs y - square')
axes2.set_title('y vs x - square ')


# Lagend

# In[ ]:


fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x, x**2, label="x**2")
ax.plot(x, x**3, label="x**3")
ax.legend()


# In[ ]:


fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x, x**2, label="x**2")
ax.plot(x, x**3, label="x**3")
ax.legend(loc=1)  #ax.legend(loc=1) # upper right corner , ax.legend(loc=2) # upper left corner  ,ax.legend(loc=3) # lower left corner ,ax.legend(loc=4) # lower right corner


# # Scatter Plot

# In[ ]:


x = np.linspace(0, 5, 11)
y = x ** 2
plt.scatter(x,y)


# # Histogram

# In[ ]:


from random import sample
data = sample(range(1, 2000), 30)
plt.hist(data)


# # Boxplot

# In[ ]:


data = [np.random.normal(0, std, 100) for std in range(1, 4)]
#  box plot
plt.boxplot(data,vert=True,patch_artist=True);   

