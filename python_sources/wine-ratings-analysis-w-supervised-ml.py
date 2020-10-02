#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Plot / Graph stuffs
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ## Initial Overview of Data
# I plan to parse the data and view it in its most raw form.

# In[2]:


init_data = pd.read_csv("../input/winemag-data_first150k.csv")
print("Length of dataframe before duplicates are removed:", len(init_data))
init_data.head()


# ### Drop Duplicates and NaNs (nulls)
# I need to drop the duplicates and null values from the data.

# In[3]:


parsed_data = init_data[init_data.duplicated('description', keep=False)]
print("Length of dataframe after duplicates are removed:", len(parsed_data))

parsed_data.dropna(subset=['description', 'points'])
print("Length of dataframe after NaNs are removed:", len(parsed_data))

parsed_data.head()


# ## First Data Look
# 
# Let's take a look at our data "description" vs "points":

# In[4]:


dp = parsed_data[['description','points']]
dp.info()
dp.head()


# How many wines per points?
# That's basically the distribution of our data model

# In[5]:


fig, ax = plt.subplots(figsize=(30,10))
plt.xticks(fontsize=20) # X Ticks
plt.yticks(fontsize=20) # Y Ticks
ax.set_title('Number of wines per points', fontweight="bold", size=25) # Title
ax.set_ylabel('Number of wines', fontsize = 25) # Y label
ax.set_xlabel('Points', fontsize = 25) # X label
dp.groupby(['points']).count()['description'].plot(ax=ax, kind='bar')


# **Conclusion:** 
# 
# A lot of wines from 83 to 93 points. Which also matches the market (not a lot of really good wines). If we wanted a better model, we could try to scrape more wine reviews based on points to flatten this.

# ## Taking a look at Description length vs points

# In[6]:


dp = dp.assign(description_length = dp['description'].apply(len))
dp.info()
dp.head()


# In[7]:


fig, ax = plt.subplots(figsize=(30,10))
sns.boxplot(x='points', y='description_length', data=dp)
plt.xticks(fontsize=20) # X Ticks
plt.yticks(fontsize=20) # Y Ticks
ax.set_title('Description Length per Points', fontweight="bold", size=25) # Title
ax.set_ylabel('Description Length', fontsize = 25) # Y label
ax.set_xlabel('Points', fontsize = 25) # X label
plt.show()


# ## Simplifying the model
# 
# Having too many different possibility for "points" would burden our model.
# A 90 points wine is not that different from a 91 points wine, the description is probably not that different also. We can this throughout the descriptions' length as well.
# 
# Let's try to simplify the model with 5 different values:
# 
# 1 -> Points 80 to 84 (Under Average wines)
# 
# 2 -> Points 84 to 88 (Average wines)
# 
# 3 -> Points 88 to 92 (Good wines)
# 
# 4 -> Points 92 to 96 (Very Good wines)
# 
# 5 -> Points 96 to 100 (Excellent wines)

# In[8]:


#Transform method taking points as param
def transform_points_simplified(points):
    if points < 84:
        return 1
    elif points >= 84 and points < 88:
        return 2 
    elif points >= 88 and points < 92:
        return 3 
    elif points >= 92 and points < 96:
        return 4 
    else:
        return 5

#Applying transform method and assigning result to new column "points_simplified"
dp = dp.assign(points_simplified = dp['points'].apply(transform_points_simplified))
dp.head()


# How many wines per simplified points?
# That's the new distribution of our data model

# In[9]:


fig, ax = plt.subplots(figsize=(30,10))
plt.xticks(fontsize=20) # X Ticks
plt.yticks(fontsize=20) # Y Ticks
ax.set_title('Number of wines per points', fontweight="bold", size=25) # Title
ax.set_ylabel('Number of wines', fontsize = 25) # Y label
ax.set_xlabel('Points', fontsize = 25) # X label
dp.groupby(['points_simplified']).count()['description'].plot(ax=ax, kind='bar')


# In[10]:


fig, ax = plt.subplots(figsize=(30,10))
sns.boxplot(x='points_simplified', y='description_length', data=dp)
plt.xticks(fontsize=20) # X Ticks
plt.yticks(fontsize=20) # Y Ticks
ax.set_title('Description Length per Points', fontweight="bold", size=25) # Title
ax.set_ylabel('Description Length', fontsize = 25) # Y label
ax.set_xlabel('Points', fontsize = 25) # X label
plt.show()


# ## Description Vectorization
# 
# Let's see what CountVectorizer output on this collection

# In[11]:


X = dp['description']
y = dp['points_simplified']

vectorizer = CountVectorizer()
vectorizer.fit(X)
#print(vectorizer.vocabulary_)


# Let's vectorize X based on the trained data.

# In[12]:


X = vectorizer.transform(X)
print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print('Density: {}'.format((density)))


# ## Training data and test data
# 
# 90% of the dataset will be used for training. 10% of the dataset will be used for testing.

# In[13]:


# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Testing the model
predictions = rfc.predict(X_test)
print(classification_report(y_test, predictions))


# ## Result With CountVectorizer: 97% precision, recall and f1
# It's super good!
# 
# Multiple hypothesis to improve this model: 
# 1. We could try a different vectorizer (TfidfVectorizer for example)
# 2. We could try cleaning the dataset using stopwords
# 3. We could add other metrics like price, description length etc... to our model

# ## Trying TfidfVectorizer
# 
# TfidfVectorizer has some advantages over the simpler CountVectorizer.
# 
# CountVectorizer just counts the word frequencies. That's all.
# 
# With TfidfVectorizer the value increases proportionally to count, but is offset by the frequency of the word in the total corpus. This is called the IDF (inverse document frequency part).
# This allow the Vectorizer to adjust with frequent words like "the", "a" etc...

# In[14]:


X = dp['description']
y = dp['points_simplified']

# Vectorizing model
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)


# In[ ]:


# Training model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Testing model
predictions = rfc.predict(X_test)
print(classification_report(y_test, predictions))


# ## Results with TfidfVectorizer: 97% precision, recall and f1
# Slightly better precisions for some buckets. TfidfVectorizer might also be more performant with larger volumes of data, knowing how the algorithm works.
# 
# # Conclusion
# The CountVectorizer/TfidfVectorizer and RandomForestClassifier looks like a great combinaison to analyze wine ratings on our dataset!
