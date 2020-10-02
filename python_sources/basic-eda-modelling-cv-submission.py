#!/usr/bin/env python
# coding: utf-8

# 
# This is my first published kernel on Kaggle.
# 
# I'm just going to go through some basic visualisations and simple EDA (exploratory data analysis) and give a basic example of modelling using XGBoost and k-means cross validation. Then I'll use the same algorithm to create a basic benchmark entry. I'm not going to do any feature engineering or parameter tuning so the model isn't going to be that accurate. This is basically just an example.
# 
# Hope you guys find it useful.

# # Getting libraries and Datasets
# 
# I'm going to keep this pretty simple, so won't need that many libraries. Just the basics for now.

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Load data
df_train = pd.read_csv('../input/learn-together/train.csv')
df_test = pd.read_csv('../input/learn-together/test.csv')
#Concatenate train and test sets into one DataFrame
df_full = pd.concat([df_train,df_test], sort=True, ignore_index = True)


# In[ ]:


# count the number of missing data and show the top ten
total = df_full.isnull().sum().sort_values(ascending=False)
percent = (df_full.isnull().sum()/df_full.isnull().count()).sort_values(ascending=False)*100
missing_data = pd.concat([total, round(percent,2)], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# No missing data?! That's gotta be a first on Kaggle.
# 
# Let's look at the training set info.

# In[ ]:


df_train.info()


# Looks pretty uniform.
# Let's look at some stats for the same set.

# In[ ]:


df_train.describe()


# Looks like all the Soil_Type columns are binary dummy variables (one column for each soil type with a 1 indicating which type it is). This could be simplified if we need to but this format is useful for some types of ML.
# 
# The Elevation attribute is simply the elevation in meters above sea level. There is a total vertical range in this set of 1986m. Lets see how this is distributed.

# In[ ]:


sns.distplot(df_train['Elevation'])


# Ok, looks like a trimodal distribution. Now let's see how the Hillsideshade columns compare to each other.

# In[ ]:


sns.distplot(df_train['Hillshade_9am'], hist_kws={'color':'b'}, kde_kws={"label":"9am",'color':'b'}) 
sns.distplot(df_train['Hillshade_Noon'], hist_kws={'color':'r'}, kde_kws={"label":"Noon", 'color':'r'})
sns.distplot(df_train['Hillshade_3pm'], hist_kws={'color':'g'}, kde_kws={"label":"3pm", 'color':'g'})


# Let's take a look at how many of each different type of tree is included in the training set:

# In[ ]:


df_train['Cover_Type'].value_counts(ascending = True)


# Very even! That's useful, means the data isn't weighted one way or the other.
# Let's look at a scatterplot of the vertical and horizontal distance to surface water.

# In[ ]:


sns.scatterplot(df_train['Horizontal_Distance_To_Hydrology'],df_train['Vertical_Distance_To_Hydrology'])


# Ok, it shows a (sort of) linear relationship, I guess that makes sense.
# 
# Now let's make the same graph, but increase the axis size and add a 'hue' parameter for the Horizontal_Distance_To_Fire_Points attribute, using the whole set, not just the training set.

# In[ ]:


dims = (20, 12)
fig, ax = plt.subplots(figsize=dims)
sns.scatterplot(df_full['Horizontal_Distance_To_Hydrology'],df_full['Vertical_Distance_To_Hydrology'], hue = df_full['Horizontal_Distance_To_Fire_Points'], ax=ax)


# Now we'll make a very simple XGBoost model. 
# First we'll split up the training set to create the model, do cross validation and then apply the model to the test set.

# In[ ]:


X = df_train.iloc[:, 1:55].values #Excludes the Id attribute
y = df_train.iloc[:, -1].values #Separates the dependant variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #Uses 80% for training and 20% for testing

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()


# Ok, about 75% accuracy, not too bad for a completely untrained model.
# Now for the test set.

# In[ ]:


# Predicting the Test set results
test_id = df_test['Id']
test_set = df_test.iloc[:,1:55]
X_train = pd.DataFrame(X_train)
test_set = pd.DataFrame(data=test_set.values)
test_set = np.array(test_set)


# In[ ]:


y_pred_test = classifier.predict(test_set)


# In[ ]:


sub = pd.DataFrame({'Id':  test_id, 'Cover_Type': y_pred_test})
sub.to_csv('submission_xgb.csv', index = False)


# In[ ]:


y_pred_test

