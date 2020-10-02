#!/usr/bin/env python
# coding: utf-8

# **We are working on Haberman_Cancer Survival Data set here .
# we are having size of 4  columns and 306  rows .
# Columns:-
# Age :- Age of the patient .
# year :- year of operation
# nodes :- No of nodes that are detected  positive
# status:- is patient surivied more than 5 years.

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


# In[ ]:


import pandas as pd
df=pd.read_csv('../input/haberman.csv')
#df.shape
df.columns
df


# In[ ]:


print("Min age is ",min(df['age']))
print("max age is ",max(df['age']))
print("Avergae Age is mean",df['age'].mean())
print("Average Age is in median",df['age'].median())


# we can test multiple use cases in the give data . 
# Lets test whether we can find any relations between them.
# 

# Is there any relation between age and nodes 

# In[ ]:


from matplotlib import pyplot as plt
x=df['age']
y=df['nodes']
plt.plot(x,y)

plt.show()
plt.bar(x,y)
plt.legend()

plt.show()
plt.scatter(x,y)
plt.show()


# * **Age and Nodes Relation**
# From the above graphs we cant find any relations correctly . we can see there are more than 30 nodes in our data we need to figure out whether it is a outlier or normal data 

# Lets try to find the relation between Nodes and status
# Hypothesis1:  if we have more nodes the chance of survival becomes less

# In[ ]:


"""from matplotlib import pyplot as plt
import numpy as np
x_axis=[np.array(df['age']),df['year'],df['nodes'],df['status']]
x_label=['age','year','nodes','status']
y_label=['age','year','nodes','status']
#y_label=df['nodes']
y_axis=[np.array(df['age']),df['year'],df['nodes'],df['status']]
#x=df['status']
xlab=0
for x in x_axis:
    xlab+=1
    ylab=0
    for y in y_axis:
        
        ylab+=1
        if not (x==y).all() :
            #plt.plot(x,y)
            plt.bar(x,y)
            plt.xlabel(x_label[xlab-1], fontsize=10)
            plt.ylabel(y_label[ylab-1],fontsize=10)
            plt.show()
            

extra=
"""
"""plt.plot(x,y)

plt.show()
plt.bar(x,y)
plt.legend()

plt.show()
plt.scatter(x,y)
plt.show()"""


# Hypothesis fails as we dont find any relation between nodes and survival

# Lets find out if we find any relation between 

# In[ ]:


import pandas as pd
df=pd.read_csv('../input/haberman.csv')
df.columns


# In[ ]:


print(df['status'].value_counts())


# In[ ]:


import matplotlib.pyplot as plt
df.plot(kind='scatter',x="age",y="nodes")
plt.show()


# In[ ]:


import seaborn as sns
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="status", height=4)    .map(plt.scatter, "age", "nodes")    .add_legend();
plt.show();


# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="status", height=3);
plt.show()


# In[ ]:


import numpy as np
more_than_5years = df.loc[df["status"] == 1];
less_than_5years = df.loc[df["status"] == 2];
plt.plot(more_than_5years["age"], np.zeros_like(more_than_5years['age']), 'o')
plt.plot(less_than_5years["age"], np.zeros_like(less_than_5years['age']), 'o')
plt.show()


# In[ ]:


plt.plot(more_than_5years["nodes"], np.zeros_like(more_than_5years['nodes']), 'o')
plt.plot(less_than_5years["nodes"], np.zeros_like(less_than_5years['nodes']), 'o')
plt.show()


# 

# In[ ]:


sns.FacetGrid(df, hue="status", height=5)    .map(sns.distplot, "nodes")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(df, hue="status", height=5)    .map(sns.distplot, "age")    .add_legend();
plt.show();


# In[ ]:


sns.FacetGrid(df, hue="status", height=5)    .map(sns.distplot, "year")    .add_legend();
plt.show();


# In[ ]:


sns.boxplot(x='status',y='nodes', data=df)
plt.show()


# In[ ]:


sns.boxplot(x='status',y='age', data=df)
plt.show()


# In[ ]:


sns.violinplot(x='status',y='nodes', data=df)
plt.show()

