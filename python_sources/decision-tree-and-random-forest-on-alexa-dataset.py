#!/usr/bin/env python
# coding: utf-8

# **dataset** [link](https://www.kaggle.com/sid321axn/amazon-alexa-reviews)

# # Using Decision Tree Classifier
# ----------------------------------------
# 
# [**Dataset**](https://www.kaggle.com/sid321axn/amazon-alexa-reviews) - Kaggle Amazon Alexa Reviews
# 
# ## 1. importing library

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[ ]:


import os
os.listdir('../input/amazon-alexa-reviews')


# ## 2. Analyzing Data

# In[ ]:


data=pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv',sep='\t')
data.head()


# In[ ]:


data.drop(columns=['date'],inplace=True)
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# ## 3. Visualizing data

# In[ ]:


sns.countplot(x='rating',data=data,hue='feedback')


# In[ ]:


sns.distplot(data['rating'])


# In[ ]:


sns.countplot(x='feedback',data=data)


# In[ ]:


plt.figure(figsize=(24,12))
sns.countplot(x='variation',hue='feedback',data=data)


# ## Transforming the object columns and making pipeline

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.compose import ColumnTransformer as ct
from sklearn.pipeline import make_pipeline as mp
from sklearn.tree import DecisionTreeClassifier as dtc


# In[ ]:


## Important: i have passed the columns a string to CV and list of columns to OHE
transformer=ct(transformers=[('review_counts',cv(),'verified_reviews'), 
                             ('variation_dummies', ohe(),['variation'])
                            ],remainder='passthrough')


# In[ ]:


pipe= mp(transformer,dtc(random_state=42))
pipe


# ## 4. Splitting the data for training and testing

# In[ ]:


from sklearn.model_selection import train_test_split as tts


# In[ ]:


data.head()


# **Note**
# ---------------
# Now we need to pass the desired column to the ohe as list and strings to cv.
# [Reason](https://stackoverflow.com/a/61838828/12210002)

# In[ ]:


x= data[['rating','variation','verified_reviews']].copy()
y= data.feedback
x.head()


# In[ ]:


x_train,x_test,y_train,y_test= tts(x,y,test_size=0.3,random_state=42,stratify=y)


# In[ ]:


x_train.shape,y_train.shape


# ## 5. Training the model

# In[ ]:


pipe.fit(x_train,y_train)


# ## 6. Testing the model

# In[ ]:


pred=pipe.predict(x_test)


# ## 7. Evaluating the model
# 

# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[ ]:


accuracy_score(y_test,pred)                 #Accuracy of 100%


# In[ ]:


sns.heatmap(confusion_matrix(y_test,pred),annot=True,fmt='.0f')


# **Note**
# 
# As we are achieveing the 100% Accuracy, Thus we are not tuning the hyperparameters.
# 
# 
# # 2. Using RandomForestClassifier
# ----------------------------------------------

# ## 1. Training the model
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier as rfc


# In[ ]:


pipe= mp(transformer, rfc(n_estimators=150, random_state=42))


# In[ ]:


pipe.fit(x_train,y_train)


# In[ ]:


pred=pipe.predict(x_test)


# ## 2. Testing the model

# In[ ]:


accuracy_score(y_test,pred)  # 99% accuracy


# In[ ]:


sns.heatmap(confusion_matrix(y_test,pred),annot=True,fmt='.0f')


# ## Note
# --------------
# 
# We are able to achieve the accuracy of **100% in decision tree classifier** and **99% in random forest classifier**.
# 
# Refer : [Notebook](https://www.kaggle.com/chitralc1/amazon-alexa-review-analysis/notebook) [ColumnTransformer](https://towardsdatascience.com/columntransformer-meets-natural-language-processing-da1f116dd69f)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




