#!/usr/bin/env python
# coding: utf-8

# In[61]:


# Import the packages
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# **Okay**, let's load the data and sort them by their manufacturer name and their product name

# In[62]:


cr = pd.read_csv("../input/cereal.csv")
cr = cr.sort_values(by=["mfr","name"])
cr.head(10)


# Cool and good, let's see which manufacturer makes the highest average rated cereal 

# In[63]:


sns.set_style(style="darkgrid")
sns.boxplot(x="mfr", y="rating", data=cr)


# Apparently, manufacturer N or ***Nabisco* **makes the best rated cereal. Let's see what are their products

# In[64]:


nabisco = cr.loc[cr.mfr == "N"]
nabisco.sort_values(by="rating", ascending=False)


# In[65]:


nabisco.rating.mean()


# On average, they're rated 67.9686 with their best product has a rating of 74.473 and their least favorite product is rated 59.364. But if you look at the box plot, there's a product manufactured by ***Kellog's*** that is so highly rated that it's considered an outlier. Let's see what it is

# In[66]:


cr.loc[cr.rating >= 90]


# ***All-Bran with Extra Fiber***  is the highest rated cereal. This raises a question because most of the highly rated products are actually low on sugar. So let's see if the amount of sugar is correlated to the rating

# In[67]:


sns.regplot(x="sugars", y="rating", data=cr)


# There's a visible relationship between amount of sugar and cereal's rating. But if we look at the plot, there's a cereal that actually has a negative amount of sugar. So let's drop it and create a model

# In[68]:


cr_new = cr.loc[cr.sugars >= 0]
sns.regplot(x="sugars", y="rating", data=cr_new)


# In[69]:


model = smf.ols(formula = "rating~sugars", data=cr_new).fit()
model.summary()


# In[ ]:


model.pvalues


# Although the model isn't very accurate to predict new values (relatively low R squared), we can conclude that there's a significant relationship between amount of sugar contained in the cereal and their rating by looking at the low p-value.
