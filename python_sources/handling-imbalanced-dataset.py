#!/usr/bin/env python
# coding: utf-8

# **IMBALANCED DATASET ** 
# 
# The dataset may contain uneven samples /instances , so that it makes the algorithm to predict with accuracy of 1.0 each time u run the model. For example, if u have simple dataset with 4 features and output(target) feature with 2 class, then total no. of instances/samples be 100. Now, out of 100, 80 instances belongs to category1 of the output(target) feature and only 20 instances contribute to the category2 of the output(target) feature. So, obviously, this makes bias in training and predicting the model. So, this dataset refers to Imbalanced dataset.
# 
# Importing Neccessary Packages and reading the csv file and printing the head of the csv file.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

file = pd.read_csv("../input/ecoli.csv")
print(file.head())


# Checking whether any column in the dataset contains NaN values.

# In[ ]:


file.isnull().values.any()


# Computing the Basic Statistics(Descriptive) of the "Class" feature in the dataset. It shows that there are two unique values(positive and negative), with positive value counts upto 143 and negative 77.

# In[ ]:


file['Class'].describe()


# Now, we just grouped the datset based on the 'class' feature to visualize the counts of positive and negative values.

# In[ ]:


f = file.groupby("Class")
f.count()


# We are converting the 'class' feature from text to int using .map function.

# In[ ]:


file['Class'] = file['Class'].map({'positive': 1, 'negative': 0})
print(file['Class'].head())


# Now using the Seaborn's pairplot, we can visualize the features plotted against each other.

# In[ ]:


sns.pairplot(file,hue='Class')


# In[ ]:


file['Class'].hist()


# Now using the sklearn library, we import train_test_test from cross validation and split the original dataset into training and test dataset(80,20).

# In[ ]:


from sklearn.cross_validation import train_test_split
train, test = train_test_split(file,test_size=0.2)
features_train=train[['Mcg','Gvh','Lip','Chg','Aac','Alm1','Alm2']]
features_test = test[['Mcg','Gvh','Lip','Chg','Aac','Alm1','Alm2']]
labels_train = train.Class
labels_test = test.Class
print(train.shape)
print(test.shape)
print(features_train.shape)
print(features_test.shape)
print(labels_train.shape)
print(labels_test.shape)
print(labels_test.head())


# Now using the RandomForest Classifier we can select the most important features in the dataset.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
model = clf.fit(features_train, labels_train)
feature_labels = ['Mcg','Gvh','Lip','Chg','Aac','Alm1','Alm2']
for feature in zip(feature_labels,model.feature_importances_):
    print(feature)


# As you can see, the feature 'Chg' and 'Lip' are contributing very low. So we can slice the dataset with only limited features.

# In[ ]:


new_file = file[['Mcg','Gvh','Aac','Alm1','Alm2','Class']]
new_file.head()


# Now once again, using the sklearn library, we import train_test_test from cross validation and split the **new sliced dataset** into training and test dataset(80,20).

# In[ ]:


new_train, new_test = train_test_split(new_file,test_size=0.2)
new_features_train=new_train[['Mcg','Gvh','Aac','Alm1','Alm2']]
new_features_test = new_test[['Mcg','Gvh','Aac','Alm1','Alm2']]
labels_train = new_train.Class
labels_test = new_test.Class
print(train.shape)
print(test.shape)
print(new_features_train.shape)
print(new_features_test.shape)
print(labels_train.shape)
print(labels_test.shape)
print(labels_test.head())


# Now using Random forest Classifier and Logistic Regression we calculate the accuracy.
# 

# In[ ]:


clf = RandomForestClassifier()
model = clf.fit(new_features_train, labels_train)
print("Accuracy of Randomforest Classifier:",clf.score(new_features_test,labels_test))


# As you see,  RandomForest  classifier produces accuracy of 100% and Logistic Regression 95%, which is biased due to the fact that there are more Positive classes than the Negative class ( **143 POSITIVE classes and 77 NEGATIVE classes.** )So this creates the biased results. So, this is a Imbalanced DataSet.
# 
# **There two main ways to handle the Imbalanced datset:**
# 
# 1.Over Sampling
# 
# 2.Under Sampling
# 
# **OVER SAMPLING:**
# 
# It is nothing but Sampling the minority class and making it equivalent to the majority class Ex:before sampling: Counter({1: 111, 0: 65}) after sampling: Counter({1: 111, 0: 111}) Note:The counts of 1's and 0's before and after sampling
# 
# **UNDER SAMPLING:**
# 
# It is nothing but Sampling the majority class and making it equivalent to the minority class Ex:before sampling: Counter({1: 111, 0: 65}) after sampling: Counter({0: 65, 1: 65})
# 
# There are several algorithms for over sampling and under sampling. The one we use here is,
# 
# **OVER SAMPLING ALGORITHM:**
# 
# **1.SMOTE** - *Synthetic Minority Over Sampling Technique *.
# A subset of data is taken from the minority class as an example and then new synthetic similar instances are created. These synthetic instances are then added to the original dataset. The new dataset is used as a sample to train the classification models
# 
# **UNDER SAMPLING ALGORITHM:**
# 
# **1.RandomUnderSampler** - Random Undersampling aims to balance class distribution by randomly eliminating majority class examples. This is done until the majority and minority class instances are balanced out.
# 
# **2.NearMiss** - selects the majority class samples whose average distances to three closest minority class samples are the smallest.

# In[ ]:


from collections import Counter
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(new_features_train, labels_train)
print("before sampling:",format(Counter(labels_train)))
print("after sampling:",format(Counter(y_resampled)))


# Now you can see that the counts of Positive and Negative classes are equal, as we over-sampled the Negative class to match the counts of the positive class.

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression()
clf1.fit(X_resampled, y_resampled)
print('Accuracy:',clf1.score(new_features_test,labels_test))

