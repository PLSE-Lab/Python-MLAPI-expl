#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # KNN Means Model

# As a beginner to data science I wanted to participate in this competition as a project based learning opportunity. Thank you to everyone who has shared their notebooks, I have discovered vast amounts of valuable insights and methods.
# 
# Please enjoy my exploration into the PBS Kids Measure Up data, and a simple implementation of a KNN Means algorithm using the sklearn library. 

# In[ ]:


sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")
specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")
test = pd.read_csv("../input/data-science-bowl-2019/test.csv")
train = pd.read_csv("../input/data-science-bowl-2019/train.csv")
train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")


# 
# ##Simple Data Exploration

# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


train_labels.head()


# In[ ]:


train_labels.title.unique()


# ## Transform the Data

# In[ ]:


#Drop unnecessary columns
train = train.drop(['event_data', 'event_id', 'timestamp'], axis = 1)


# In[ ]:


#Merge train and train_lables dataframes
group = pd.merge(train, train_labels, on= ['game_session', 'installation_id', 'title']).sort_values(['game_session','installation_id'])
train.head()


# In[ ]:


#Condense rows into summarized dataframe
group.groupby(['game_session', 'installation_id', 'title', 'type', 'world', 'num_correct', 'num_incorrect','accuracy_group']).agg({'game_time':'sum', 'event_code':list})


# ## Prepare Data for KNN Model

# In[ ]:


#Import label encoder
from sklearn import preprocessing

#Create label encoder
le = preprocessing.LabelEncoder()


# In[ ]:


# Create encoded variables to use in model

# First Feature
#group['title'] = group['title'].astype('category').cat.codes
title = le.fit_transform(group['title'])

#Second Feature
game_time = le.fit_transform(group['game_time'])

#Target variable from training labels
outcome = le.fit_transform(group['accuracy_group'])


# In[ ]:


#Combine features into single list of tuples
features = list(zip(title,game_time))


# In[ ]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size = .8)


# In[ ]:


## KNN Model


# In[ ]:


#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
model = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
model.fit(X_train,y_train)


# In[ ]:


#Predict the response for test dataset
predicted = model.predict(X_test)


# ## Determine Accuracy

# In[ ]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, predicted))


# 
# source: https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
