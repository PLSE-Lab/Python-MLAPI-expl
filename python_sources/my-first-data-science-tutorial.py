#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #drawing library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# I read the data file

# In[ ]:


data = pd.read_csv("../input/Iris.csv")     


# In[ ]:


data.info()


# I saw the first 10 records

# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


data.Species.unique()


# In[ ]:


setosa = data[data.Species == "Iris-setosa"]
versicolor = data[data.Species == "Iris-versicolor"]
virginica = data[data.Species == "Iris-virginica"]


# In[ ]:


# Line plot
# color = color,label = label
# x = Id, 

plt.plot(setosa.Id, setosa.PetalLengthCm, color = "red", label ="setosa" )
plt.plot(versicolor.Id, versicolor.PetalLengthCm, color = "green", label ="versicolor" )
plt.plot(virginica.Id, virginica.PetalLengthCm, color = "blue", label ="virginica" )
plt.xlabel("Id")                          # label = name of label
plt.ylabel("PetalLengthCm")
plt.legend()  
plt.show()


# In[ ]:


# Scatter Plot 
# x = PetalLengthCm, y = PetalWidthCm

plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm,color="red",label="setosa")
plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm,color="green",label="versicolor")
plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm,color="blue",label="virginica")

plt.legend()
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.title("scatter plot")
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure
plt.hist(setosa.PetalLengthCm,bins=10)
plt.xlabel("PetalLengthCm values")
plt.ylabel("frekans")
plt.title("hist")
plt.show()


# In[ ]:


#subplot

plt.subplot(2,1,1)
plt.plot(setosa.Id, setosa.PetalLengthCm, color = "red"  )
plt.ylabel("setosa-PetalLengthCm")

plt.subplot(2,1,2)
plt.plot(versicolor.Id, versicolor.PetalLengthCm, color = "green" )
plt.ylabel("versicolor-PetalLengthCm")

plt.show()



# In[ ]:


# Filtering Pandas data frame
setosa = data['PetalLengthCm'] < 2   #All smaller than 2 cm because 50 records available for iris-setosa
data[setosa]

