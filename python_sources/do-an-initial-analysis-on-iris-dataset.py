#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 


# In[ ]:


# import the data
iris = pd.read_csv('../input/Iris.csv')


# In[ ]:


# Let's have a look 
iris.head()


# In[ ]:


iris.shape


# In[ ]:


# let's groupby species
species = pd.unique(iris['Species'])


# In[ ]:


species


# ## Ok, so far so good!

# In[ ]:


def group_them(data, column):
    
    result = []
    kinds = pd.unique(data[column])
    for kind in kinds:
        result.append(np.where(data[column] == kind)[0])
    
    return result


# In[ ]:


result = group_them(iris, 'Species')


# In[ ]:


# Let's check the number of each specie
for i in range(3):
    print('The number of %s is %d.' % (species[i], result[i].shape[0]))


# In[ ]:


# group them
setosa = iris.loc[result[0]]
versicolor = iris.loc[result[1]]
virginica = iris.loc[result[2]]


# ## Stats Analysis

# In[ ]:


stats_iris = pd.DataFrame(np.zeros((3,4)),
                          index=['setosa', 'versicolor', 'virginica'],
                          columns=['Avg_Sepal_Length', 'Avg_Sepal_Width', 'Avg_Petal_Length', 'Avg_Petal_Width'])


# In[ ]:


stats_iris.iloc[0] = setosa[setosa.columns[1:-1]].mean().values
stats_iris.iloc[1] = versicolor[versicolor.columns[1:-1]].mean().values
stats_iris.iloc[2] = virginica[virginica.columns[1:-1]].mean().values


# In[ ]:


stats_iris


# ## Now, we get the average of each feature. That's pretty cool!

# ## Ok, let's visualized it!

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300


# In[ ]:


species = ['setosa', 'versicolor', 'virginica']

fig, ax = plt.subplots()

plt.xlim(0.,32)
plt.ylim(0.,24)
fig.set_size_inches(12, 9)

grid = np.mgrid[0.2:1.4:4j, 1.0:0.2:-3j].reshape(2, -1).T * 20

plt.text(grid[1][0], grid[1][1]-0.01, 'Sepal', ha="center", family='sans-serif', size=20)
plt.text(grid[2][0], grid[2][1]-0.01, 'Pepal', ha="center", family='sans-serif', size=20)

colors = ['#2196F3', '#4CAF50', '#FF5722']

for i in range(3):
    plt.text(grid[3*i+3][0], grid[3*i+3][1],
             species[i], ha='center', family='sans-serif', size=20)
    sepal = Ellipse(grid[i*3+4],
                    width = stats_iris.iloc[i]['Avg_Sepal_Width'],
                    height = stats_iris.iloc[i]['Avg_Sepal_Length'])
    
    petal = Ellipse(grid[i*3+5],
                    width = stats_iris.iloc[i]['Avg_Petal_Width'],
                    height = stats_iris.iloc[i]['Avg_Petal_Length'])
    ax.add_artist(sepal)
    ax.add_artist(petal)
    sepal.set_color(colors[i])
    petal.set_color(colors[i])


# ## Now, we can find out the differences among the three kind of iris.

# #  Have Fun! 
