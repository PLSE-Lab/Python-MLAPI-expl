#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model as lm


# **1. Select 5 variables from your dataset. For each, draw a boxplot and analyze your observations.**

# **'accommodates'**

# In[39]:


o = pd.read_csv("../input/listings_detail_uploaded.csv")
o1 = o[["accommodates"]]
print(o1["accommodates"].value_counts())
plt.boxplot(o1["accommodates"])
plt.show()
print(o1.describe())


# **'bedrooms'**

# In[16]:


o2 = o[["bedrooms"]]
print(o2["bedrooms"].value_counts())
plt.boxplot(o2["bedrooms"])
plt.show()
print(o2.describe())


# **'availability_30'**

# In[17]:


o3 = o[["availability_30"]]
print(o3["availability_30"].value_counts())
plt.boxplot(o3["availability_30"])
plt.show()
print(o3.describe())


# **'number_of_reviews'**

# In[43]:


o4 = o[["number_of_reviews"]]
print(o4["number_of_reviews"].value_counts())
plt.boxplot(o4["number_of_reviews"])
plt.show()
print(o4.describe())


# **'price'**

# In[44]:


o5 = o[["price"]]
print(o5["price"].value_counts())
plt.boxplot(o5["price"])
plt.show()
print(o5.describe())


# **2.	Select four pairs of variables from your dataset. Draw a scatterplot for each pair and make your visual observations.**

# In[32]:


# Beds vs bedrooms
o6 = o[["beds"]]
plt.scatter(o6,o2)
plt.xlabel('Beds')
plt.ylabel('Bedrooms')
plt.title('Beds vs bedrooms')
plt.show()


# In[33]:


# availability_30 vs review_scores_rating
o7 = o[["review_scores_rating"]]
plt.scatter(o3,o7)
plt.xlabel('availability_30')
plt.ylabel('review_scores_rating')
plt.title('availability_30 vs review_scores_rating')
plt.show()


# In[42]:


# accommodates vs bathrooms
o8 = o[["bathrooms"]]
plt.scatter(o1,o8)
plt.xlabel('accommodates')
plt.ylabel('bathrooms')
plt.title('accommodates vs bathrooms')
plt.show()
# print(o1)


# In[45]:


# number_of_reviews vs price
plt.scatter(o4,o5)
plt.xlabel('number_of_reviews')
plt.ylabel('price')
plt.title('number_of_reviews vs price')
plt.show()


# In[ ]:




