#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Visualization with Matplotlib (Basic Topics)**
# 
# Hi! Today I'm going to explain you basic commands and topics about matplotlib. So let's begin.
# 
# Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK+.Matplotlib graphics can be edited very elegantly, but here we will focus on the basic features. We have an Iris dataset here. Now we're going to apply matplotlib and see some of basic topics like "line plot", "scatter plot", "bar plot", "subplot" and "histogram".

# First of all we should write data's path.  

# In[ ]:


df = pd.read_csv("../input/Iris.csv")


# With .info() we can easily see basic informations about data frame.

# In[ ]:


df = pd.read_csv("../input/Iris.csv")
df.info()


# In[ ]:


By using .describe() we can see numeric values (non-string values like mean, standard deviation). 


# In[ ]:


df = pd.read_csv("../input/Iris.csv")
print(df.describe())


# For example if you want to compare setosa's and versicolor's mean value than you can simply write following lines. 

# In[ ]:


setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]

print(setosa.describe())
print(versicolor.describe())


# Pyplot is used for visualising. If we want to make visualization according to species -> 
# (Note: First, You should write "from matplotlib import pyplot as plt" on the top.)

# In[ ]:


setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]

plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm,color = "green", label = "setosa")
plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm, color = "red", label = "versicolor")
plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm, color = "blue", label = "virginica")

plt.xlabel("PetalWidthCm")
plt.ylabel("PetalLengthCm")

plt.legend() #It shows the text on the top right.
plt.show()

print("")

df.plot(grid = True, alpha = 0.9) #Divides the background into squares and alpha sets transparency.
plt.show()


# Subplot shows piece by piece ->

# In[ ]:


df.plot(grid = True, alpha = 0.9, subplots = "True")
plt.show()

print(" ")

setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]

plt.subplot(2,1,1) #2 means there're two sublots, and 1 stands for "we got the first one".
plt.plot(setosa.PetalLengthCm, color = "black", label = "setosa")
plt.ylabel("setosa.petalLengthCm")

plt.subplot(2,1,2)
plt.plot(versicolor.PetalLengthCm, color = "gray", label = "versicolor")
plt.ylabel("versicolor.petalLengthCm")
plt.show()


# Scatter plot is mostly used to compare two futures. Let's compare PetalLength and PetalWidth.

# In[ ]:



setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]

plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm,color = "red", label = "setosa")
plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm, color = "green", label = "versicolor")
plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm, color = "blue", label = "virginica")

plt.legend()
plt.xlabel("PetalLenghtCm")
plt.ylabel("PetalWidthCm")
plt.title("Scatter Plot")
plt.show()


# "Histogram" is a diagram consisting of rectangles whose area is proportional to the frequency of a variable and whose width is equal to the class interval.

# In[ ]:


plt.hist(setosa.PetalLengthCm, bins=35) #As bins increases, it becomes thinner.
plt.xlabel("PetalLengthCm Values")
plt.ylabel("Frequency")
plt.title("Hist")
plt.show()
#When we look at the histogram, there were 14 setosa that has 1.5 petalLenght


# "Bar Plot" (or barchart) is one of the most common type of plot. It shows the relationship between a numerical variable and a categorical variable. (This is an example independent of Iris data set.)

# In[ ]:


x = np.array([1,2,3,4,5,6,7])
y = x*2+5

plt.bar(x,y)
plt.title("Bar Plot")
plt.show()

