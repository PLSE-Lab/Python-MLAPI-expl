#!/usr/bin/env python
# coding: utf-8

# The dataset contains recipes and their details. For this problem, lets use Logistic Regression to classify whether a recipe is a dessert or no, based on the calories it has. A short notebook using seaborn and visualization to view the classification.

# ### Importing Relevant Libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading Data

# In[ ]:


recipes = pd.read_csv("../input/epi_r.csv")


# ### Cleaning Data
# 
# Let's limit the dataframe to have only recipes which are less than 10,000 calories. We also will clean the data by dropping rows which have null values.

# In[ ]:


recipes = recipes[recipes['calories'] < 10000].dropna()


# ### Visualization

# In[ ]:


sns.set(style="whitegrid")
g = sns.regplot(x="calories", y="dessert", data=recipes, fit_reg=False)
g.figure.set_size_inches(10, 10)


# Since, we're identifying whether the recipe is a dessert or not - its a category. Directly plotting it using seaborn gives us the below graph. As seen, as the calories increase, the category of the recipe moves towards '1' i.e. its a dessert. 

# In[ ]:


sns.set(style="whitegrid")
g = sns.regplot(x="calories", y="dessert", data=recipes, logistic=True)
g.figure.set_size_inches(10,10)


# The shaded area is our error in prediction. Since we've not set or created a model but used the inbuilt regression, we can see that the error is less when the calories are low. As calories increase, our error increases too. This is ideally not a good thing to have in a model.
# 
# This completes this notebook.
#                                                         
