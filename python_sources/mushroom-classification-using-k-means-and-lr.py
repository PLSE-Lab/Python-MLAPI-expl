#!/usr/bin/env python
# coding: utf-8

# **MUSHROOM CLASSIFICATION**
# 
# This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.
# This dataset can be found on Kaggle

# 1)IMPORT NECESSARY LIBRARIES

# In[ ]:


import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.head()


# 2)DATA PREPROCESSING

# In[ ]:


df.isnull().sum()


# There are no null values

# In[ ]:


df['class'].unique()


# There are two categories, poisonous and edible

# Let us encode the binary categories to zeroes and ones

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in df.columns:
    if len(df[col].value_counts())==2:
        
        df[col]=labelencoder.fit_transform(df[col])
    
df.head()


# For the rest of the cloumns with more than one categories, let us employ one hot encoding

# In[ ]:


df=pd.get_dummies(df)
df.head()


# In[ ]:


X=df.drop(['class'],axis=1)
X.head()


# In[ ]:


Y=df['class']
Y.head()


# In[ ]:


Y=Y.to_frame()
Y.head()


# In[ ]:


X.describe()


# 3) K-MEANS CLUSTERING
# 
# This is an unsupervised classification technique and hence does not require labels

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


kmeans=KMeans(n_clusters=2)


# In[ ]:


kmeans.fit(X)


# In[ ]:


clusters=kmeans.predict(X)


# In[ ]:


cluster_df = pd.DataFrame()

cluster_df['cluster'] = clusters
cluster_df['class'] = Y
cluster_df.head()


# In[ ]:


cluster0_df=cluster_df[cluster_df['cluster']==0]
cluster0_df.head()


# In[ ]:


cluster1_df=cluster_df[cluster_df['cluster']==1]
cluster1_df.head()


# In[ ]:


sns.countplot(x="class", data=cluster0_df)


# In[ ]:


sns.countplot(x='class',data=cluster1_df)


# We can conclude that the first cluster has almost all the poisonous mushrooms and almost no edible mushrooms. However, the second cluster has almost all the edible mushrooms and around 500 poisonous mushrooms.

# 4) LOGISTIC REGRESSION
# 
# This is a supervised learning algorithm which works well for binary classification as well as multi-class classification

# In[ ]:


Y.describe()


# Let us split our data into training and testing, with 20% of data allocated for testing

# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.20)


# In[ ]:


#K-Means Clustering with two clusters
kmeans = KMeans(n_clusters=2)

#Logistic Regression with no special parameters
logreg = LogisticRegression()


# In[ ]:


kmeans.fit(train_X)

logreg.fit(train_X, train_y)


# 5) MAKING PREDICTIONS ON OUR MODELS

# In[ ]:


kmeans_pred = kmeans.predict(test_X)

logreg_pred = logreg.predict(test_X)


# In[ ]:


test_y


# 6) MODEL EVALUATION

# In[ ]:


#This DataFrame will allow us to visualize our results.
result_df = pd.DataFrame()

#The column containing the correct class for each mushroom in the test set, 'test_y'.
result_df['test_y'] = test_y['class'] 

#The predictions made by K-Means on the test set, 'test_X'.
result_df['kmeans_pred'] = kmeans_pred
#The column below will tell us whether each prediction made by our K-Means model was correct.
result_df['kmeans_correct'] = result_df['kmeans_pred'] == result_df['test_y']

#The predictions made by Logistic Regression on the test set, 'test_X'.
result_df['logreg_pred'] = logreg_pred
#The column below will tell us whether each prediction made by our Logistic Regression model was correct.
result_df['logreg_correct'] = result_df['logreg_pred'] == result_df['test_y']


# In[ ]:


result_df


# In[ ]:


sns.countplot(x=result_df['kmeans_correct'], order=[True,False]).set_title('K-Means Clustering')


# In[ ]:


sns.countplot(x=result_df['logreg_correct'], order=[True,False]).set_title('Logistic Regression')


# 7) VERDICT
# 
# Logistic Regression is the clear winner

# In[ ]:




