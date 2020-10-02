#!/usr/bin/env python
# coding: utf-8

# 

# **We will see in corrolation between Sepal measures and Petal sizes.
# We will use numpy,pandas matplot and seaborn to analyse the Iris Speices data.**

# In[ ]:




import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


data=pd.read_csv('../input/Iris.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# **In the following heatmap we can see the corrolation between Sepal Length,SepalWidth,PetalLength andPetalWidth**

# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.SepalLengthCm.plot(kind = 'line', color = 'g',label = 'SepalLengthCm',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.PetalLengthCm.plot(color = 'r',label = 'PetalLengthCm',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.SepalWidthCm.plot(kind = 'line', color = 'g',label = 'SepalWidthCm',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.PetalWidthCm.plot(color = 'r',label = 'PetalWidthCm',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# **We can see the Corroaltion between Sepal Length and Petal Lenght are quite obvious.**

# In[ ]:


data.plot(kind='scatter', x='SepalLengthCm', y='PetalLengthCm',alpha = 0.5,color = 'red')
plt.xlabel('SepalLengthCm')              # label = name of label
plt.ylabel('PetalLengthCm')
plt.title('SepalLengthCm PetalLengthCm Scatter Plot')            # title = title of plot


# In[ ]:


data.plot(kind='scatter', x='SepalWidthCm', y='PetalWidthCm',alpha = 0.5,color = 'red')
plt.xlabel('SepalWidthCm')              # label = name of label
plt.ylabel('PetalWidthCm')
plt.title('SepalWidthCm PetalWidthCm Scatter Plot')            # title = title of plot


# **We can see the common frequecy of the measures Sepal Length,SepalWidth,PetalLength andPetalWidth**

# In[ ]:


data.PetalLengthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


# Histogram
# bins = number of bar in figure
data.SepalLengthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


data.PetalWidthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


data.SepalWidthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()

