#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


os.getcwd()


# In[ ]:


os.chdir('/kaggle/input/iris-data/')


# In[ ]:


os.getcwd()


# In[ ]:


iris = pd.read_csv('Iris.csv')


# In[ ]:


iris["Species"].value_counts()


# In[ ]:


iris.columns


# In[ ]:


plt.figure(1)

# 1st plot
plt.subplot(231)
sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="Species", size=4)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend();
plt.show()

# 2nd plot
plt.subplot(232)
sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="Species", size=4)    .map(plt.scatter, "PetalLengthCm", "PetalWidthCm")    .add_legend();
plt.show()

# 3rd plot
plt.subplot(233)
sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="Species", size=4)    .map(plt.scatter, "SepalLengthCm", "PetalWidthCm")    .add_legend();
plt.show()

# 4th plot
plt.subplot(234)
sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="Species", size=4)    .map(plt.scatter, "PetalLengthCm", "SepalWidthCm")    .add_legend();
plt.show()

# 5th
plt.subplot(235)
sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="Species", size=4)    .map(plt.scatter, "PetalLengthCm", "SepalLengthCm")    .add_legend();
plt.show()

# 6th
plt.subplot(236)
sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="Species", size=4)    .map(plt.scatter, "PetalWidthCm", "SepalWidthCm")    .add_legend();
plt.show();


# **I want to get 6 unique pair plots from below in 3x2 grid plot**

# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(iris, hue="Species", size=3);
plt.show()


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="Species", size=4)    .map(plt.scatter, "PetalLengthCm", "PetalWidthCm")    .add_legend();
plt.show()


# In[ ]:


g = sns.FacetGrid(iris, col="Species")


# In[ ]:


plt.scatter(iris['PetalLengthCm'], iris['SepalLengthCm'], label)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
data = (g1, g2, g3)
colors = ("red", "green", "blue")
groups = ("coffee", "tea", "water")plt.show()


# In[ ]:


#data = (iris[iris["Species"]=="	Iris-setosa"], iris[iris["Species"]=="Iris-virginica"], iris[iris["Species"]=="Iris-versicolor"])

g1 = (iris["PetalLengthCm", iris["SepalWidthCm"]])
g2 = (iris["PetalWidthCm", iris["SepalWidthCm"]])
g3 = (iris["PetalLengthCm", iris["SepaLengthCm"]])

data = (g1, g2, g3)
colors = ("red", "green", "blue")
groups = ("coffee", "tea", "water")

for data, color, group in zip(data, colors, groups):
    x, y = data
ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()


# In[ ]:


ratio = iris["SepalLengthCm"]/iris["SepalWidthCm"]

for name, group in iris.groupby("Species"):
    plt.scatter(group.index, ratio[group.index], label=name)

plt.legend()
plt.show()


# In[ ]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[ ]:


x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(1)
#for q in range(1,7):
for q, x_index, y_index in [(1,0,1), (2,0,2), (3,0,3), (4,2,1), (5,2,3), (6,3,1)]:
   # for x_index, y_index in zip([0, 0, 0, 2, 2, 3], [1, 2, 3, 1, 3, 1]):
        plt.subplot(2, 3, q)
    #    x_index = 0
    #    y_index = 1

        # this formatter will label the colorbar with the correct target names
        formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

        plt.figure(figsize=(5, 4))
        plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
        plt.colorbar(ticks=[0, 1, 2], format=formatter)
        plt.xlabel(iris.feature_names[x_index])
        plt.ylabel(iris.feature_names[y_index])

        #plt.tight_layout()
        plt.show()


# In[ ]:


plt.figure(1)               

plt.subplot(231)             
plt.plot([1, 2, 3])
plt.subplot(232)             
plt.plot([4, 5, 6])
plt.subplot(235)             
plt.plot([1, 2, 3])
plt.subplot(236)             
plt.plot([4, 5, 6])


# In[ ]:


for x_index, y_index in [(0,1), (0,2), (0,3), (2,1), (2,3), (3,1)]:
    print (x_index, y_index)


# In[ ]:




