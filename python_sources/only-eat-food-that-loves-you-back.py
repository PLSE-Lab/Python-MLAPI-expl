#!/usr/bin/env python
# coding: utf-8

# Analysis of a [nutrition data set](https://www.kaggle.com/crawford/80-cereals) about cereals.
# 
# 
# ### Import the relevant modules

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import boxcox1p


# ### Read the data

# In[ ]:


data = pd.read_csv('../input/cereal.csv')
data.head()


# In[ ]:


# checking the rows and cols
data.shape


# In[ ]:


# checking for null values
data.isnull().sum()


# In[ ]:


# checking the data types
data.dtypes


# ## Are customers generous when giving ratings?
# 
# The ratings distribution is slightly skewed but we will leave it as it is. 

# In[ ]:


b = sns.distplot(data['rating'], hist=False)
b.axes.set_title('Distribution of Ratings by customers', fontsize = 30)
b.set_xlabel('Rating', fontsize = 20)
b.set_ylabel('Count', fontsize = 20)
plt.show()


# ## Top products
# 
# Let's check the products who have a rating > 60

# In[ ]:


data[data['rating'] > 60]


# ## High is low?
# 
# One thing that catches my eye seeing the output is that some of the columns like fat, sodium and sugar have values in their lower range for the highest rated products.

# In[ ]:


interesting_cols = ['fat', 'sodium', 'sugars', 'vitamins']
data[data['rating'] > 60][interesting_cols]


# ## Worst products
# 
# Now we check for products with a rating < 20

# In[ ]:


data[data['rating'] < 20]


# These products seem to have fat, sodium, sugars and vitamins in the **higher range**. However this is just a simple observation, and can be verified by the scatterplot shown below but nothing can be concluded as yet.

# In[ ]:


plt.figure(figsize=(12, 8))
b = sns.swarmplot(data['sugars'], data['rating'])
b.axes.set_title('Do products with high rating have low sugar content?', fontsize = 30)
b.set_xlabel('Grams of sugar', fontsize = 20)
b.set_ylabel('Rating', fontsize = 20)
plt.show()


# # Quality vs Quantity
# 
# Do the manufacturers who produce most products also produce the best ones? Let's find that out next

# In[ ]:


b = sns.countplot(data['mfr'])
b.axes.set_title('Who produces most?', fontsize = 30)
b.set_xlabel('Manufacturer', fontsize = 20)
b.set_ylabel('Number of products', fontsize = 20)
plt.show()


# # K & G
# 
# The plot shows that most manufacturers produce less than 10 products, while **K and G** produce over 20 products. But do they produce quality products?

# In[ ]:


b = sns.countplot(data[data['rating'] > 60]['mfr'])
b.axes.set_title('Who produces the best products?', fontsize = 30)
b.set_xlabel('Manufacturer', fontsize = 20)
b.set_ylabel('Count (rating > 60)', fontsize = 20)
plt.show()


# # Quantity != Quality
# 
# Even though **K and G** produce the **most products**, they don't produce the best ones. **N** and **Q** seem to be the **most efficient** at that, while **K** barely produces a single good product. **G** does not even feature in the list. It surely is in the other list. Since rating < 20 has only 2 products we shall find ratings < 25.

# In[ ]:


b = sns.countplot(data[data['rating'] < 25]['mfr'])
b.axes.set_title('Who produces the worst products?', fontsize = 30)
b.set_xlabel('Manufacturer', fontsize = 20)
b.set_ylabel('Count (rating < 20)', fontsize = 20)
plt.show()


# The plot shows that **G** infact produces the worst products. This is even more evident if we see ratings < 30. Also notice **Q** in both the plots. This shows that **Q** either produces the best products or the worst ones

# ## Cold or hot
# 
# Most food items are served cold which is understandable!

# In[ ]:


data['type'].value_counts()


# In[ ]:


b = sns.countplot(data['type'])
b.axes.set_title('Type of food items', fontsize = 30)
b.set_xlabel('Type', fontsize = 20)
b.set_ylabel('Count', fontsize = 20)
plt.show()


# The rest of the columns are not categorical so it will be better if we find the correlation and then analyse them.

# In[ ]:


# find the correaltions
corr = data.corr().sort_values('rating')
corr['rating'].head()


# ## Read the last column of the heatmap

# In[ ]:


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# plot the heatmap
with sns.axes_style("white"):
    ax = sns.heatmap(corr, vmax=.3, square=True, annot=True, linewidths=3)
plt.show()


# # Plotting the 4 most negatively correlated variables
# 
# 

# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(15,15))


a = sns.swarmplot(data['sugars'], data['rating'], ax=axes[0][0])
# b.axes.set_title('Do products with high rating have low calorie content?', fontsize=30)
a.set_xlabel('Grams of sugar', fontsize = 20)
a.set_ylabel('Rating', fontsize = 20)


b = sns.swarmplot(data['calories'], data['rating'], ax=axes[0][1])
# b.axes.set_title('Do products with high rating have low calorie content?', fontsize=30)
b.set_xlabel('Calories per serving', fontsize = 20)
b.set_ylabel('Rating', fontsize = 20)


c = sns.swarmplot(data['fat'], data['rating'], ax = axes[1][0])
# c.axes.set_title('Do products with high rating have low fat content?', fontsize = 30)
c.set_xlabel('Grams of Fat', fontsize = 20)
c.set_ylabel('Rating', fontsize = 20)


d = sns.swarmplot(data['sodium'], data['rating'], ax=axes[1][1])
# d.axes.set_title('Do products with high rating have low sugar content?', fontsize = 30)
d.set_xlabel('milligrams of sodium', fontsize = 20)
d.set_ylabel('Rating', fontsize = 20)

plt.show()


# # Plotting the 3 most positively correlated variables
# 
# 
# ## note :
# We choose only 3 here because the 4th variable carbohydrates has a very low correlation coefficient.

# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(15,15))

a = sns.swarmplot(data['fiber'], data['rating'], ax=axes[0][0])
# b.axes.set_title('Do products with high rating have low calorie content?', fontsize=30)
a.set_xlabel('grams of dietary fiber', fontsize = 20)
a.set_ylabel('Rating', fontsize = 20)


b = sns.swarmplot(data['protein'], data['rating'], ax=axes[0][1])
# b.axes.set_title('Do products with high rating have low calorie content?', fontsize=30)
b.set_xlabel('grams of protein ', fontsize = 20)
b.set_ylabel('Rating', fontsize = 20)


c = sns.swarmplot(data['potass'], data['rating'], ax = axes[1][0])
# c.axes.set_title('Do products with high rating have low fat content?', fontsize = 30)
c.set_xlabel('milligrams of potassium', fontsize = 20)
c.set_ylabel('Rating', fontsize = 20)

plt.show()


# The trends look pretty obvious from both the subplots.
# 
# ### Now we shall use these variables to train a model

# In[ ]:




