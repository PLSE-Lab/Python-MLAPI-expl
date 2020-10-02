#!/usr/bin/env python
# coding: utf-8

# # Analyzing the dataset

# ## 1. Importing Data

# In[90]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir('../input'))


# In[92]:


data = pd.read_csv('../input/Transformed Data Set - Sheet1.csv');


# In[6]:


data.sample(3)


# In[7]:


data.info()


# ## 2. Visualization

# For visualization, I will use count plots because all my plots will involve two categorical variables.

# In[98]:


# sns.countplot('Gender', data=data, palette='Greens') # Just like 'Greens', there's also 'Reds', 'Purples' etc... Also, you can append to any of these _r which means reverse or _d which means dark.
data['Gender'].value_counts().plot.pie(explode=[0,0.1], shadow=True)


# In[87]:


sns.countplot('Favorite Color', hue='Gender', data=data, palette='Greens')


# The data suggests that there might be a female preference for warmer colors and a tendency for males to prefer cooler colors.

# In[88]:


sns.countplot(y='Favorite Music Genre', hue='Gender', data=data, palette='Greens');


# The data seems to indicate that females might prefer rock, jazz/blues and pop over males, who might prefer R&B/soul, electronic and hip hop music.

# In[99]:


sns.countplot('Favorite Beverage', hue='Gender', data=data, palette='Greens');


# It seems that people who don't drink are more likely to be males. If this is true, one potential explanation is that people who don't drink usually belong to conservative families and that I was more likely to come across a male than a female participant from a conservative family.

# In[100]:


sns.countplot(y='Favorite Soft Drink', hue='Gender', data=data, palette='Greens');


# It might be the case that 7UP/Sprite is more popular with females while Fanta is more popular with males.

# ## 3. Conclusion

# The data indicates that there is a potential for discovering statistically significant patterns in male and female preferences. It justifies obtaining a larger dataset and applying statistical models.
