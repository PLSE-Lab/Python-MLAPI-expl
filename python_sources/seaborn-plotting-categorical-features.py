#!/usr/bin/env python
# coding: utf-8

# Today we will cover how categorical features are plotted in seaborn .At least one of the features should be numeric.Lets load titanic dataset

# In[15]:


import seaborn as sns 
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
import pandas as pd 


# In[16]:


titanic=pd.read_csv("../input/train.csv")
titanic.head()


# The first kind of the plot is Bar Plot. Here the mean of each variable is plotted with the measure of a central tendency .

# In[17]:


sns.catplot(x="Sex", y="Survived", kind="bar",hue='Embarked', data=titanic);


# the above barplots divides male people into three categories according to the port of embarkment (s,c, or q). Then the mean o survival features in computed , and plotted in the two d graph

# Another case arises when the absolutte counts of a data needs to be plotted. We use count plots in that cases. This is a special case of a bar graph.

# In[26]:


sns.catplot(x="Sex",hue='Pclass' ,kind="count", data=titanic);


# Another variant of the barplot and countplot is the pointplot. It represents the mean statistics of each categorical variablle, along with an error bar. It also connects the points with same hue values .

# In[29]:


sns.catplot(x="Sex", y="Survived", kind="point",hue='Embarked',jitter=True, data=titanic);


# Now lets understand the concepts of markers in pointplots. Unlike the relational plots we do not have style marker to vary the natire of p lots. So lets use the palette and marker instead

# In[33]:


#changing the default line colurs
sns.catplot(x="Pclass", y="Survived", kind="point",hue='Sex', data=titanic);


# In[32]:


sns.catplot(x="Pclass", y="Survived", kind="point",hue='Sex',   palette={"male": "g", "female": "m"}, data=titanic);


# In[41]:


#chaging the arkers
sns.catplot(x="Pclass", y="Survived", kind="point",hue='Sex',palette={"male": "g", "female": "m"}, markers=["o", "x"], data=titanic);


# ***Faceting***
# 
# All the types of plots are built on the top of facetgrid.In other words, the x,y and hue variables can be traced onto the facet grid.

# In[44]:


titanic.head()


# > Suppose we want to know the relation between age distribtution and genders of the people ,against the class of ticket they purchased and port of embarkment.

# In[47]:


sns.catplot(x='Sex',y='Age',col='Survived',row='Pclass',kind='bar',data=titanic)


# 

# In[ ]:




