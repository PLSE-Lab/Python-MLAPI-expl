#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### This notebook will provide the ML techniques for sampling imbalanced datasets
# This dataset is a HIGHLY unbalanced dataset, meaning the target classes are unbalanced
# Class = 0,1
# * 0 = Non-fraud
# * 1 = Fraud

# In[1]:


import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter


# In[2]:


data = pd.read_csv("../input/creditcard.csv")


# In[ ]:


data.head()
#31 columns of data. As we can see,all the columns starting wiith 'v' seem to be data extracted from the Principal Components of the original data.


# In[ ]:


#let's try to learn more about the data.
#Target variable is the 'Class' variable, where 0 means 'fraudulent' and 1 means 'genuine'
data.describe()


# In[ ]:


len(data)
#A total of 284,807 rows


# In[ ]:


#let's check the distribution of the target variable 'Class'
Counter(data['Class'])


# We can see that the data is HIGHLY unbalanced, meaning the data in class variable of 0 is very high compared to the class variable of 1 

# In[ ]:


#plot a barplot to see how unbalanced the data looks graphically
import seaborn as sns


# In[ ]:


count_class = pd.value_counts(data['Class'])
print(count_class)


# In[ ]:


plt.figure(figsize=(16,10))
count_class.plot(kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Class')


# In[ ]:


data.columns


# The columns starting with 'V' are already the Principal Components. So there is nothing much that we can do with these columns.
# Now, the column "Time" is the time elapsed between each transaction. Is this really an important field for us? No. It really will not have any significance in detecting whether a transaction is fraudulent or genuine. So let's drop this column.
# The "Amount" column is necessary. So we shall standardise the Amount data, since all the other columns have been obtained through PCA, and PCA ALWAYS works on Standardised data.

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data['s_Amount']=sc.fit_transform(data['Amount'].reshape(-1,1))


# Drop the "Time" field and the "Amount" field. We shall use the new "s_Amount field" with scaled data for our modelling

# In[ ]:


data = data.drop(['Time', 'Amount'], axis=1)
data.columns


# 
# 
# Split the data into x and y variables
# 

# In[ ]:


x = data.loc[:, data.columns!= 'Class']
y = data.loc[:, data.columns == 'Class']


# Split the data into train and test data

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state = 62)


# I will attempt to perform 3 methods of sampling to resolve this imbalanced dataset issue
# 1. Undersampling 
# 2. Oversampling
# 3. SMOTE - Synthetic Minority Oversampling Technique 

# ### 1. Undersampling
# The process of reducing the class instances of the MAJORITY class is called Undersampling. I will attempt to undersample the data and give a 50/50 ratio to each of the class's instances.
# Following are the steps for undersampling
# 1. Find the number of the minority class
# 2. Find the indices of the majority class
# 3. Find the indices of the minority class
# 4. Randomly sample the majority indices with respect to the minority numbers
# 5. Concat the minority indices with the indices from step 4
# 6. Get the balanced dataframe - This is the final undersampled data
# 
# **Disadvantage** is you will lose critical data as you are reducing the instances of the majority class.
# 

# In[ ]:


#1. Find the number of the minority class
number_fraud = len(data[data['Class']==1])


# In[ ]:


number_non_fraud = len(data[data['Class']==0])


# In[ ]:


print(number_fraud)
print(number_non_fraud)


# In[ ]:


#2. Find the indices of the majority class
index_non_fraud = data[data['Class']==0].index


# In[ ]:


#.3 Find the indices of the minority class
index_fraud = data[data['Class']==1].index


# In[ ]:


#4. Randomly sample the majority indices with respect to the number of minority classes
random_indices = np.random.choice(index_non_fraud, number_fraud,replace='False')


# In[ ]:


len(random_indices)


# In[ ]:


#5. Concat the minority indices with the indices from step 4
under_sample_indices = np.concatenate([index_fraud,random_indices])


# In[ ]:


#Get the balanced dataframe - This is the final undersampled data
under_sample_df = data.iloc[under_sample_indices]


# In[ ]:


under_sample_df


# In[ ]:


Counter(under_sample_df['Class'])


# In[ ]:


under_sample_class_counts = pd.value_counts(under_sample_df['Class'])


# In[ ]:


under_sample_class_counts.plot(kind='bar')


# We can see that the classes are now equally distributed. Now, split the data into x, y, train, and test

# In[ ]:


x_under = under_sample_df.loc[:, under_sample_df.columns!='Class']
y_under = under_sample_df.loc[:, under_sample_df.columns=='Class']
x_under.columns
y_under.columns


# In[ ]:


x_under_train, x_under_test, y_under_train, y_under_test = train_test_split(x_under, y_under, test_size=0.25, random_state=100)


# In[ ]:


x_under_train.head()


# In[ ]:


y_under_train.head()


# Run a Logistic Regression Classifer

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr_under = LogisticRegression()
lr_under.fit(x_under_train, y_under_train)


# In[ ]:


from sklearn.metrics import accuracy_score, recall_score
lr_under_predict = lr_under.predict(x_under_test)
lr_under_accuracy = accuracy_score(lr_under_predict, y_under_test)
lr_recall = recall_score(lr_under_predict, y_under_test)
print(lr_under_accuracy)
print(lr_recall)


# We can see that the recall is 98%, which is a great number. We can say that our model is correctly classifying data as 'fraudulent' with 98% accuracy.
# However, we see that accuracy is lesser than recall. This is normal, as we have undersampled our data.

# ### 2. Oversampling
# The process of increasing the class instances of the MINORITY class is called Oversampling. 
# **Disadvantage** is it will cause OVERFITTING as we are increasing the samples of the minority class..

# In[ ]:


fraud_sample = data[data['Class']==1].sample(number_non_fraud, replace=True)


# In[ ]:


#create a new dataframe containing only non-fraud data
df_fraud = data[data['Class']==0]


# In[ ]:


over_sample_df = pd.concat([fraud_sample,df_fraud], axis=0)


# In[ ]:


over_sample_class_counts=pd.value_counts(over_sample_df['Class'])


# In[ ]:


over_sample_class_counts.plot(kind='bar')
plt.xlabel = 'Class'
plt.ylabel = 'Frequency'


# We can now see that through oversampling, the counts of both the classes in the data set are equal.
# Now, we model using Logistic Regression.
# Split the data into x,y, train, and test

# In[ ]:


x_over = data.loc[:,over_sample_df.columns!='Class']
y_over = data.loc[:,over_sample_df.columns=='Class']


# In[ ]:


x_over_train, x_over_test, y_over_train, y_over_test = train_test_split(x_over, y_over, test_size = 0.25)


# In[ ]:


lr_over = LogisticRegression()
lr_over.fit(x_over_train,y_over_train)
lr_over_predict=lr_over.predict(x_over_test)


# In[ ]:


lr_over_accuracy = accuracy_score(lr_over_predict, y_over_test)
lr_over_recall = recall_score(lr_over_predict, y_over_test)
print(lr_over_accuracy)
print(lr_over_recall)


# We can see that the accuracy is VERY high, but the recall is very low compared to what we saw for the undersampling recall. This is because oversampling causes OVERFITTING, as the data is multiplicated.

# ### 3. SMOTE - Syntetic Minority Over Sampling Technique
# The right way to work on imbalanced data and SMOTE is to oversample only on the training data, and leave the test data unseen
# 1. Split the training data further into train and validation data
#  1. original test data = x_test, y_test
#  2. original train_data = x_train, y_train
# 2. I will further split x_train, y_train to x_val, y_val, x_train_new, y_train_new
# 3. I will build the models on x_val and y_val, and check the model for performance on x_train_new, y_train_new
# 4. Finally I will check the performace of the model on the unseen x_test, y_test

# In[ ]:


import imblearn
from imblearn.over_sampling import SMOTE


# In[ ]:


x_val, x_train_new, y_val,y_train_new = train_test_split(x_train, y_train, test_size = 0.25, random_state=12)


# In[ ]:


sm = SMOTE()


# In[ ]:


x_train_res, y_train_res = sm.fit_sample(x_val, y_val)


# Here SMOTE.fit_sample gives me the resampled data i.e the oversampled data

# In[ ]:


x_train_res, y_train_res = sm.fit_sample(x_val, y_val)
Counter(y_train_res)


# As seen above, the result of SMOTE gives us equal distribution of the 2 target classes.

# In[ ]:


lr_smote = LogisticRegression()
lr_smote.fit(x_train_res, y_train_res)
#predict on the train data
lr_smote_predict = lr_smote.predict(x_train_new)


# In[ ]:


#print accuracy and recall on train data
print(accuracy_score(lr_smote_predict,y_train_new))
print(recall_score(lr_smote_predict,y_train_new))


# In[ ]:


#predict on the test data
lr_smote_predict_test = lr_smote.predict(x_test)
print(accuracy_score(lr_smote_predict_test,y_test))
print(recall_score(lr_smote_predict_test,y_test))


# We can see that the recall score for both the train data and the 'unseen' test data are almost similar. This means that we have built a good model using SMOTE

# Using Random Forest Classifer

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train_res, y_train_res)
rf_smote_predict = rf.predict(x_train_new)
rf_smote_predict_test = rf.predict(x_test)


# Check accuracy and recall on the train data

# In[ ]:


print(accuracy_score(rf_smote_predict,y_train_new))
print(recall_score(rf_smote_predict,y_train_new))


# Check accuracy and recall on the unseen test data

# In[ ]:


print(accuracy_score(rf_smote_predict_test,y_test))
print(recall_score(rf_smote_predict_test,y_test))


# We can see that Random Forest performs MUCH BETTER in predicting the frauds in our dataset with the accuracy being 85% on the test data. Also, since the recall on the unseen test data is close to the recall of the train data, this model would perform well un production (as in prod, our model will be predicting on unseen test data)'

# In[ ]:




