#!/usr/bin/env python
# coding: utf-8

# Hello folks! In this kernal session we will look at some data visualization techniques using the Seaborn library

# In[ ]:


import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
train.head()


# First let's look ar .countplt() which is a bar graph that shows the count

# In[ ]:


sns.countplot(x='Sex', data=train)


# Next we will see .barplot() 

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train)


# To show the quartiles and identify if there are outliers we can use the boxplot

# In[ ]:


sns.boxplot(x='Sex', y='Age', data=train)


# Now we will look at swarmplot and how useful it can be in visualizing your data

# In[ ]:


sns.swarmplot(x='Pclass', y='Fare', data=train)


# We can extend the use of swarmplot like this,

# In[ ]:


sns.swarmplot(x='Pclass', y='Fare', hue='Sex', data=train)


# As you can see now, we can understand how the gender is spread among the classes and fare

# For the next set of visualizations, we will use the iris dataset. 

# In[ ]:


iris = pd.read_csv('../input/iris/Iris.csv')
iris.head()


# Let's now look at heatmap. A very useful map that shows the correlations between features

# In[ ]:


sns.heatmap(iris.corr())


# In[ ]:


sns.heatmap(iris.corr(), cmap='coolwarm', annot=True)


# Here we changed the palette to 'coolwarm' to have a better understanding. If it's red we say the features
# are higly related and if it's blue we say the features correlation is less.

# There many more visualization methods in the Seaborn library. Please do check them in the offical documentation and explore themm all. Upvote this kernal if you found it useful. 
