#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv", index_col=0)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


#the amount of missing values per feature
sns.set(rc={'figure.figsize':(11,4)})
pd.isnull(data).sum().plot(kind='bar')
plt.ylabel('Number of missing values')
plt.title('Missing Values per Feature');


# # EDA

# In[ ]:


# Lets see the age distribution first
sns.set(rc={'figure.figsize':(11,5)})
plt.hist(data.Age, bins=40)
plt.xlabel('Age')
plt.ylabel('Reviews')
plt.title('Number of reviews per Age');


# We can say that, age group 25-50 is the most revieving age group (target :))
# 
# Now, let's have a look at the distribution of ratings per age.

# In[ ]:


sns.set(rc={'figure.figsize':(11,6)})
sns.boxplot(x = 'Rating', y = 'Age', data = data)
plt.title('Rating Distribution per Age');


# It looks like upper/lower quartiles and medians are pretty much close to each other, which means rating scores are not mainly related with the age. Or we can't say that, the young is more happy.
# And pls note this is not the distribution of rating scores. It's coming next.

# In[ ]:


sns.set(rc={'figure.figsize':(6,6)})
plt.title('Distribution of Ratings')
sns.countplot(x = 'Rating', data = data);


# Rating scores are showing a general satisfaction but of course companies need to work more to make it better.

# Now let's have a look at the distributions of Division/Department/Class

# In[ ]:


fig = plt.figure(figsize=(14, 14))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = plt.xticks(rotation=45)
ax1 = sns.countplot(data['Division Name'])
ax1 = plt.title("Reviews in each Division")


ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2 = plt.xticks(rotation=45)
ax2 = sns.countplot(data['Department Name'])
ax2 = plt.title("Reviews in each Department")


ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3 = plt.xticks(rotation=45)
ax3 = sns.countplot(data['Class Name'])
ax3 = plt.title("Reviews in each Class")


# And now, let's check the rate of recommendations.

# In[ ]:


recommended = data[data['Recommended IND']==1]
not_recommended = data[data['Recommended IND']==0]


# In[ ]:


fig = plt.figure(figsize=(14, 14))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = sns.countplot(recommended['Division Name'], color = "green", alpha = 0.8, label = "Recommended")
ax1 = sns.countplot(not_recommended['Division Name'], color = "red", alpha = 0.8, label = "Not Recommended")
ax1 = plt.title("Recommended Items in each Division")
ax1 = plt.legend()

ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2 = sns.countplot(recommended['Department Name'], color="yellow", alpha = 0.8, label = "Recommended")
ax2 = sns.countplot(not_recommended['Department Name'], color="purple", alpha = 0.8, label = "Not Recommended")
ax2 = plt.title("Recommended Items in each Department")
ax2 = plt.legend()

ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3 = plt.xticks(rotation=45)
ax3 = sns.countplot(recommended['Class Name'], color="cyan", alpha = 0.8, label = "Recommended")
ax3 = sns.countplot(not_recommended['Class Name'], color="blue", alpha = 0.8, label = "Not Recommended")
ax3 = plt.title("Recommended Items in each Class")
ax3 = plt.legend()


# OMG! Dresses deparment need urgent care

# What's the most popular item?

# In[ ]:


fig = plt.figure(figsize=(14, 9))
plt.xticks(rotation=45)
plt.xlabel('item ID')
plt.ylabel('popularity')
plt.title("Top 50 Popular Items")
data['Clothing ID'].value_counts()[:50].plot(kind='bar');


# This plot might be higly valuable for the buyers of the company

# In[ ]:


fig = plt.figure(figsize=(14, 9))

ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = sns.boxplot(x="Division Name", y="Rating", data=data)
ax1 = plt.title('Rating Distribution per Divison')

ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2 = sns.boxplot(x="Department Name", y="Rating", data=data)
ax2 = plt.title('Rating Distribution per Department')

ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3 = plt.xticks(rotation=45)
ax3 = sns.boxplot(x="Class Name", y="Rating", data=data)
ax3 = plt.title('Rating Distribution per Class')


# I can't understand why the Dress department has so many negative recommendations but high ratings.
# I'll come back to this point later
# 
# Next ML Models/NLP
