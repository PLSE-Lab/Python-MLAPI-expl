#!/usr/bin/env python
# coding: utf-8

# Analysing McDonald's Menu

# Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')


# In[4]:


menu = pd.read_csv('../input/menu.csv')
menu.head(4)


# In[8]:


menu.tail()


# Let's check it out if null values are in

# In[9]:


print(menu.isnull().any())


# In[11]:


print(menu.describe())


# In[13]:


print(menu.columns)


# In[21]:


menu.get_dtype_counts()


# In[22]:


menu.info()


# In[24]:


menu.shape


# What kind of categories in it?

# In[51]:


menu.Category.unique()


# There are 9 category

# In[25]:


sns.boxplot("Calories","Total Fat",data=menu);


# In[26]:


sns.boxplot("Calories","Sugars",data=menu);


# In[26]:


menu.Calories.max()


# In[28]:


menu.Item[menu.Calories == 1880]


# In[30]:


menu.Calories.min()


# In[32]:


menu.Sugars.max()


# In[33]:


menu.Item[menu.Sugars == 128]


# Oh.. McFlurry with M&M's candies....

# In[34]:


menu.Item[menu.Sugars >= 100]


# Mostly shakes and McFlurry

# Menu which has 0 calories

# In[31]:


menu.Item[menu.Calories == 0]


# 
# Let's analyze breakfast menu
# ----------------------------
# 
# 

# In[29]:


breakfast


# Most high calories in breakfast menu

# breakfast.Calories.max()

# Which menu has 1150 kcal as breakfast?

# In[15]:


breakfast.Item[breakfast.Calories == 1150]


# then low calories?

# In[17]:


breakfast.Calories.min()


# which menu is it?

# In[18]:


breakfast.Item[breakfast.Calories == 150]


# which are the breakfast menu more than 500 calories?

# In[19]:


breakfast.Item[breakfast.Calories >= 500]


# Is there any menu which have Sugars more than 10?

# In[23]:


breakfast.Item[breakfast.Sugars > 10]


# In[24]:


sns.boxplot("Calories","Total Fat",data=breakfast);


# In[25]:


sns.boxplot("Calories","Sugars",data=breakfast);


# In[66]:





# In[ ]:




