#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, roc_auc_score ,recall_score, 
                             precision_score, confusion_matrix, classification_report, f1_score, auc)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# setting the size of the dataframe to disply
pd.set_option('max_columns', 25)


# ## Exploring Data

# In[ ]:


# loading the data

mushroom_df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

# printing the first five row of the dataframe
mushroom_df.head()


# In[ ]:


# no of rows and columns

mushroom_df.shape


# In[ ]:


# checking for any null values

mushroom_df.isnull().sum()


# In[ ]:


# check information about dataset 

mushroom_df.info()


# In[ ]:


# detail about dataset

mushroom_df.describe()


# In[ ]:





# In[ ]:


# function to print the value coutn go each columm

cols = mushroom_df.columns.to_list()
def value_count(cols):
    each_cols = mushroom_df[cols]
    for i in each_cols:
        print('Number of unique value in column "{}" is : {} -->  {} \n'.format(i.upper(), len(dict(each_cols[i].value_counts())) ,dict(each_cols[i].value_counts())))
        #print(dict(each_cols['class'].value_counts()))

     
value_count(cols)


# In[ ]:


# Convert categories to numbers using one hot encoder

x = pd.get_dummies(mushroom_df[mushroom_df.columns[1:]])
x.head()


# In[ ]:


# converting output value to numberic
labe_encode = LabelEncoder()
y = labe_encode.fit_transform(mushroom_df['class'])


# In[ ]:


# checking the no or columns after one hot encoding

x.shape[1]

# there are 117 columsn, excluding output column class


# ## spliting data into train and test

# In[ ]:


# spliting data into train and test

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=1, stratify = y)


# ## check no of row and columns in train and test

# In[ ]:


print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_train:', y_test.shape)


# ## Modeling

# In[ ]:


# create an instance of decision tree

tree_classifier = DecisionTreeClassifier(criterion='gini', random_state=42)

# fit the data to model

tree_classifier.fit(x_train, y_train)


# In[ ]:


# make a prediction with testing data

y_predict = tree_classifier.predict(x_test)
y_predict_train = tree_classifier.predict(x_train)


# # Model Evaluation

# In[ ]:


# checking model accuracy

print('Model accuracy: ', accuracy_score(y_test, y_predict))


# In[ ]:


# cross validation to evaluate model

cross_metrix = cross_val_score(tree_classifier, x, y, scoring='accuracy')

print(cross_metrix)
print(cross_metrix.mean())
print(cross_metrix.std())


# In[ ]:


# confusion matrix

confusion_score = confusion_matrix(y_test, y_predict)
confusion_score


# In[ ]:


# plotting confusion matrix

sns.heatmap(confusion_score, annot=True, annot_kws={'size':16})


# In[ ]:


# slice confusion matrix into four pieces

TP = confusion_score[1, 1]
TN = confusion_score[0, 0]
FP = confusion_score[0, 1]
FN = confusion_score[1, 0]


# ## Metrics computed from a confusion matrix

# In[ ]:


# classification Erroe rate

print('Error rate: ', 1 - accuracy_score(y_test, y_predict))

# 0.0 shwos that there is no error, out model is perfect


# In[ ]:


# True positive (Recall or sensitivity)
print('True positive (Recall or sensitivity)', recall_score(y_test, y_predict))


# True Negative (sensitivity)
print('True positive (specificity)', TN/ (TN + FP))


# False Positive 
print('False positive (Recall or sensitivity)', FP/ (FP + TN))


# Precision
print('Precision', precision_score(y_test, y_predict))


# In[ ]:


# print classification report

print('classification report: \n', classification_report(y_test, y_predict))


# In[ ]:


#  predicted responses

fpr,tpr,thresholds=roc_curve(y_test,y_predict)

# calculate acu curv
roc_auc=auc(fpr,tpr)


# In[ ]:


# we pass y_test and y_pred_prob
# we do not use y_pred_class, because it will give incorrect results without generating an error
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# Use both of these whenever possible
# 
# **Confusion matrix advantages:**
# 
# - Allows you to calculate a variety of metrics
# - Useful for multi-class problems (more than two response classes)
# 
# **ROC/AUC advantages:**
# 
# - Does not require you to set a classification threshold
# - Still useful when there is high class imbalance

# In[ ]:




