#!/usr/bin/env python
# coding: utf-8

# In this notebook, I try as much as possible to share the intuition that goes into every cell. I'm not an expert at this but I hope others can build on my mistakes and also arrive at better my results. I have looked at other notebooks and realized that the accuracies are pretty low. My goal here is to use a different approach to improve results. I do not use any fancy Machine Learning (ML) models here. I use a KNN and a Decision Tree model. I also avoid aggregating the data based on time and panel. This complicates things. Again, the gola here is to see what will happen if I reduce the complexities in the data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import additional modules for visualization (exploratory data analysis)

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# load all datasets
dsb_train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
dsb_test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
dsb_trainlabels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
dsb_specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

# i've had to make this edit because of cpu usage on kaggle
dsb_train = dsb_train.drop(['event_data'], 1)
dsb_test = dsb_test.drop(['event_data'], 1)


# In[ ]:


# examine features in each dataset
print("train :", dsb_train.keys())
print("test :", dsb_test.keys())
print("labels :", dsb_trainlabels.keys())
print("specs :", dsb_specs.keys())


# From the above, we can conclude that both train and test have the same headers. This is a good start. Next, lets check to see the number of observations contained in each dataset. This is important because it will help to understand the split for train and test later when we implement our selected ML models

# In[ ]:


# examine number of rows / observations in each dataset
print("train :", dsb_train.shape)
print("test :", dsb_test.shape)
print("labels :", dsb_trainlabels.shape)
print("specs :", dsb_specs.shape)


# Think about the above results for a minute. Do you notice any challenge from the results? Remember, the goal for this project is to predict the "accuracy_group". So the obvious task is to merge the "train" and "train_labels" datasets to match the target variable with the explanatory variables. However, the labels dataset has only 17690 observations, which is way less than the 11,341,042 in the train dataset.
# 
# To work around this challenge, lets attempt to merge the two datasets based on the common identifying features in both; installation_id or game_session. But first, we will look at the headers from both datasets to see if they give us any aditional information on these two features.

# In[ ]:


dsb_train.head()


# In[ ]:


dsb_trainlabels.head()


# From the two dataframes, we notice that game_session and installation_id can be repeated (not unique). However, we can still merge them using game sessions. We can infer that game sessions in the labels dataset has multiple counts in the train dataset, hence, the difference in observations. We can look at the unique values to see if they are close.

# In[ ]:


# import module for counting
import collections
collections.Counter(dsb_train['game_session'])


# In[ ]:


collections.Counter(dsb_trainlabels['game_session'])


# The above confirms that game_sessions are not repeated in the label set. Next, lets merge the datasets to see how it goes.

# In[ ]:


new_train = pd.merge(dsb_train, dsb_trainlabels, on='game_session')


# Time for the verdict. Now, we explore the new dataset and hope it makes sense :)

# In[ ]:


new_train.shape


# We have a new problem... It appears merging the dataset reduced our training observations to 867,447 from 11,341,042. That's a huge drop. Can we blame ourselves? If you find a better way to do this, kindly share. As it stands, there are more observations in the test dataset than the train. However, it's not the end of the world. It only shows how particularly challenging this dataset is. Next, we explore the dataset using seaborn visualizations.

# In[ ]:


# list features in merged dataset 
new_train.keys()


# In[ ]:


# drop repeated columns in merged data (installation id and title features)
new_train = new_train.drop(['installation_id_y', 'title_y'],1)

# rename installation_id_x and title_x
new_train = new_train.rename(columns={"title_x": "title", "installation_id_x": "installation_id"})

# check to see if renaming worked
new_train.keys()


# In[ ]:


# create pairplot to identify interesting trends
# sns.pairplot(new_train, hue="title") had to comment this out because of low computing power


# In[ ]:


# distribution of accuracy groups (0,1,2,3)
sns.countplot(x="accuracy_group", data=new_train)


# The cell below displays the breakdown of accuracy groups by the game title. Evidently, Mushroom sorter and and Cauldron filter seem to have the highest number of first-time completions.

# In[ ]:


sns.countplot(x="accuracy_group", data=new_train, hue="title")


# Now that we have a fair idea of what our dataset looks like, lets prepare the two datasets for our ML models.
# 
# Next, we generate dummies for the categorical variables in both train and test dataset. Note that we want train and test columns to be equal. Hence, any categorical value that is not in both train and test datasets must not be considered in the final model. You can compare unique categorical features by using the code "*dataframe[feature].unique*"

# In[ ]:


# check unique values
#print(dsb_test['world'].unique())
#print(dsb_train['world'].unique())
#print(dsb_test['event_code'].unique())
#print(new_train['event_code'].unique())
#print(dsb_trainlabels['event_code'].unique()) etc


# In[ ]:


train2 = pd.get_dummies(new_train, columns=['event_code', 'world','title'], drop_first=True)
test2 = pd.get_dummies(dsb_test, columns=['event_code', 'world','title'], drop_first=True)

# quick note: drop_first is True if we want to drop the original column we are converting.


# Next, lets compare to see if we have the same number of columns. We need to do this everytime we modify the dataframes

# In[ ]:


print("train shape ", train2.shape) # you can also use .keys() depending on what you are looking out for
print("test shape ", test2.shape)


# New challenge! Columns in test frame do not match train. This will create more problems for us. However, we soldier on! Lets drop all columns in the test frame that are not in the train frame.

# In[ ]:


# list of train features
train_list = train2.keys()

# next, drop test feature if not in train list
test3 = test2.drop(columns=[col for col in test2 if col not in train_list])

# print shapes to check if dropping worked
print("train shape ", train2.shape) 
print("test shape ", test3.shape)


# Perfect! The above is what we expect. Remember train has additional features such as "*accuracy, accuracy_group, num_correct, num_incorrect*". Remeber, this is not the best way to go about this. However, my target is simplicity. Next step, lets convert the timestamp column into period of day and day of the week. This will help us get some more interpretation from our imperfect data.

# In[ ]:


# import date and time models
from datetime import datetime
import time


# In[ ]:


# parse timestamp columns as timestamp dtypes
train2['date'] = pd.to_datetime(train2['timestamp']).astype('datetime64[ns]')
test3['date'] = pd.to_datetime(test3['timestamp']).astype('datetime64[ns]')


# In[ ]:


# create new columns: hour and days

# 1. create hour feature (0 - 24)
train2['t_hour'] = (train2['date']).dt.hour
test3['t_hour'] = (test3['date']).dt.hour

# 2. create day feature (0-Sunday,..., 6-Saturday)
train2['t_day'] = (train2['date']).dt.weekday
test3['t_day'] = (test3['date']).dt.weekday 

# print shapes to check if we are on track
print("train shape ", train2.shape) 
print("test shape ", test3.shape)


# We are almost ready now. Next, lets drop columns that will not be useful for our simple model. Remember this is a panel data and so more work may be required to arrive at acceptable results. However, I'm curious to see what accuracies our efforts will reslut in.

# In[ ]:


# These are the features I don't believe are useful for our simple analysis

train3 = train2.drop(['date','event_id','game_session','installation_id','type','num_correct',
       'num_incorrect','accuracy','timestamp'], 1)
test4 = test3.drop(['date', 'event_id','game_session','installation_id','type','timestamp'], 1)

# print shapes to check if we are on track
print("train shape ", train3.shape) 
print("test shape ", test4.shape)


# Great! Now with our new features, lets see if there are any interesting correlations using a heatmap

# In[ ]:


sns.set(font_scale=1.5) #increast size of plot font
plt.figure(figsize=(20, 10)) #increase size of plot
sns.heatmap(train3[['accuracy_group','title_Cart Balancer (Assessment)',
       'title_Cauldron Filler (Assessment)', 'title_Chest Sorter (Assessment)',
       'title_Mushroom Sorter (Assessment)', 't_hour', 't_day']].corr(), annot = True)


# I'll leave the intepretation to you. You can try additional features to see what comes out of it. Next, lets prepare our datasets for Machine Learning predictions. Since the target variable has three classes (0-3), we use models that are a good fit for classification problems.

# In[ ]:


# Select X and y features for both train and test

# for train
train_X = train3.drop(['accuracy_group'], 1)
train_y = train3['accuracy_group']

# for test
test_X = test4


# Start modelling already!!

# In[ ]:


# DECISION TREE MODEL
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dtree = DecisionTreeClassifier()
dtree_model = dtree.fit(train_X, train_y)

dtree_train_y = dtree_model.predict(train_X) #The Decision tree prediction for the train_X data.
dtree_val_y = dtree_model.predict(test_X) #The Decision tree prediction for the val_X data.
dtree_train_accuracy = accuracy_score(train_y, dtree_train_y) #The accuracy for the dtree_train_y prediction.

# Print Accuracies for Decision Tree
print("Decision Tree Training Accuracy: ", dtree_train_accuracy)


# In[ ]:


# i've had to comment this out as well due to cpu usage
"""
# K NEAREST NEIGHBOR MODEL
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier() # you can opt to qualify number of neighbors here (eg. n_neighbors = 5)
knn_model.fit(train_X, train_y)

#This creates the prediction. 
knn_train_y = knn_model.predict(train_X) #The KNN prediction for the train_X data.
knn_val_y = knn_model.predict(test_X) #The KNN prediction for the val_X data.
knn_train_accuracy = accuracy_score(train_y, knn_train_y) #The accuracy for the knn0_train_y prediction.

# Print Accuracies for Decision Tree
print("KNN Training Accuracy: ", knn_train_accuracy)
"""


# The above are two examples of classification models. Since the decision tree model provides the best accuracy, I stick to that. Remember in Machine Learning, accuracies are not really significant in determining how good your model is. There are several ways to test for this. For example, there could be overfitting.
# 
# Next, we need to append the classification from the decision tree, to our initial test dataset (ie. dsb_test)

# In[ ]:


dsb_test['accuracy_group'] = dtree_model.predict(test_X)

# check test dataframe to see if appending worked
dsb_test.head()


# Awesome! Appending worked. now we need to export our submission using the instalation_id and accuracy_group columns.

# In[ ]:


import csv

# create dataframe
sub = dsb_test.loc[:,['installation_id','accuracy_group']]
submission = sub.drop_duplicates(subset="installation_id", keep="last") # dropping duplicates in test data


# In[ ]:


# create csv file from submission dataframe
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.shape


# Thank you for your time! I hope this wasn't too boring. All the best with the competition!

# ## THE END!
