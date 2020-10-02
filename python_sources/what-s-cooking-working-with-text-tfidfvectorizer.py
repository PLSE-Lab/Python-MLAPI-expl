#!/usr/bin/env python
# coding: utf-8

# ## What's Cooking

# ### Loading necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))


# ### Loading data

# In[ ]:


train = pd.read_json('../input/train.json')
train.set_index('id' , inplace= True)
label = train['cuisine']
train.drop('cuisine' , axis = 1 , inplace= True)
test = pd.read_json('../input/test.json')


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## EDA

# In[ ]:


print('Number of train data ' , len(train))
print('Number of test data ' , len(test))


# ### Numbers of cuisine

# In[ ]:


len(label.unique())


# There are 20 types of cuisine in this dataset.

# ### Distribution of cuisines. 

# In[ ]:


plt.figure(figsize=(16, 6))
sns.countplot(y = label , order = label.value_counts().index)


# 1. Most of the cuisines are Italian and Mexican.
# 2. Least data is available for the Russian and Brazilian.
# 3. Imbalanced dataset

# ### Number of ingredient 

# In[ ]:


type(train.ingredients[0])


# #### The datatype of values in the ingredients column is a list.

# In[ ]:


print('Maximum ingredients used in a single cuisine' , train.ingredients.apply(len).max())
print('Minimum ingredients used in a single cuisine' , train.ingredients.apply(len).min())


# ## ML

# ### Let's define a function to convert a list to a string.

# In[ ]:


def list_to_text(data):
    return (" ".join(data)).lower()


# ##### Lets test it

# In[ ]:


list_to_text(['a' , 'b'])


# Ok its working

# ### Converting ingredients columns from a list to string.

# In[ ]:


train.ingredients = train.ingredients.apply(list_to_text )
test.ingredients = test.ingredients.apply(list_to_text)


# In[ ]:


train.head()


# In[ ]:


test.head()


# ### Working with text features

# In[ ]:



tfidf = TfidfVectorizer()


# In[ ]:


X_train = tfidf.fit_transform(train.ingredients)
X_test = tfidf.transform(test.ingredients)


# In[ ]:


l = LabelEncoder()
label = l.fit_transform(label)


# In[ ]:


label


# In[ ]:


clf = XGBClassifier()
scores = cross_val_score(clf, X_train, label, cv=3).mean()
scores


# In[ ]:


clf.fit(X_train , label)
pre = clf.predict(X_test)


# In[ ]:


pre


# ### Inverse the prediction to its name/label using LabelEncoder's inverse_transform

# In[ ]:


pre = l.inverse_transform(pre)
pre


# ### Prepare the submission file.

# In[ ]:


submit = pd.read_csv('../input/sample_submission.csv')
submit.head()


# In[ ]:


submit.cuisine = pre
submit.id = test.id


# In[ ]:


submit.to_csv('submit.csv' , index= False)


# In[ ]:


get_ipython().system('ls')


# In[ ]:




