#!/usr/bin/env python
# coding: utf-8

# Curious on Epicurious recipes !
# -----------------------------
# Let's explore the taste of foodies .
# 
#  1. What is the relation between rating and calorie of the recipes?
#  2. Will there be any correlation between Nutrition features and Rating's of Recipes?
#  3. What is the composition of vegetarian and non-vegetarian recipes in 5 start rating?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.gridspec as gridspec
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[ ]:


recipes =pd.read_csv("../input/epi_r.csv").dropna()
recipes.head(2)


# ## What is the relation between rating and calorie of the recipes? ##

# In[ ]:


ax=sns.pointplot(x="rating",y="calories", data=recipes)
ax.set(ylabel='calories')
plt.title("Relationship between rating and calories")


# The recipes with 4.375 has higher calories when compared to best rating 5. So **recipes with more calories get's rating around 4 than getting 5.**

# ## Will there be any correlation between Nutrition features and Rating's of Recipes? ##

# In[ ]:


correlation = recipes.iloc[:,1:6].corr()


# In[ ]:


plt.figure(figsize=(8,7))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between rating & nutrition features of recipes')


# It looks like there is no correlation between Nutrients and Rating. But, we could see **stronger relation between sodium and fat**.

# ## What is the composition of vegetarian and non-vegetarian recipes in 5 start rating? ##

# In[ ]:


recipes_5 = recipes[recipes["rating"] == 5]
veg_nv = recipes_5.groupby(["vegetarian"])["title"].count()
plt.figure(figsize=(4,4))
plt.pie(veg_nv,labels=["non-vegeterian","vegeterian"],autopct='%1.1f%%', startangle=90, colors=["tomato","lightgreen"])
plt.axis("equal")
plt.title("Composistion of Veg and Non-Veg Recipes in 5-star ratings")


# Non-Veg Recipes have more 5star rating than Veg Recipes .
