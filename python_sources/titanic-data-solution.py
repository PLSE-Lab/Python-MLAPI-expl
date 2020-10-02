#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# In[ ]:


print(train_df.columns.values)


# In[ ]:


train_df.head()


# We can see that the column Survived, the value which we are attempting to predict, is a binary value. Either the passenger survived or didn't survive.

# In[ ]:


train_df.tail()


# Let's see which columns have null or missing values along with the data type to get a better feel for our data.

# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# It appears that cabin has the most missing values, followed by Age, followed by Embarked.

# In[ ]:


train_df.describe()


# Various records say that the actual number of passengers abord the Titanic for it's tragic maiden voyage was 2,224, which means our train_df dataset (whichc includes 891 passangers) represents 40% of all passangers. This above description of our data shows that around 38% of our represented passangers survived the sinking of the Titanic. It also shows that one ticket cost 512 dollars which would amount to around $13,000 in 2019 dollars (wow!).

# In[ ]:


train_df.describe(include=['O'])


# Now I will pivot the feature that are often assumed to be connected with the chance of surviving the Titanic incident. First of all I will check out the class of the ticket which was purchased 

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# There is a significant connection between class ranking and survival rate, with first class being the most likely to survive.

# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# There is significant connection between sex and survival rate, with women being much more likely to survive.

# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# 

# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Taking a looking at a visualization of the data will give us a better idea of the connection better survival.

# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# We can see the passenger's who did not survive (survived = 0) were disproportionally ages 15 through 25. And most of our passenger's were age's 15 to 35.

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=3, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# Here we can see that the passengers in first class were much more likely to survive than those in the lower classers. There is a dramatic contrast between those in first class who died and those  in 3rd class.

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# We can see that women had a much greater chance of survival than men. But there is an exception when it comes to the passengers who embarked from port C where men had a much greater since of survival. Males had a better rate of survival in pclass=3 when compaired to any other pclass=3.

# Since some features aren't contributing to our machine learning model, I dropped the features Cabin and Ticket across the training and testing tables to make the notebook faster.
