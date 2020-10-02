#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load the dataset

# In[ ]:


iris_data = pd.read_csv('/kaggle/input/iris-dataset/Iris.csv')
print(iris_data.shape)
print(iris_data.head())


# We remove the Id column as it has no significance on determining the class labels

# In[ ]:


iris_data.drop('Id',inplace=True,axis=1)
print(iris_data.head())


# **Let us see the class distribution in the dataset. We can use the value_counts function to get the count of unique values for a given column**

# In[ ]:


print(iris_data['Species'].value_counts())

There are 3 classes each with 50 samples(perfectly balanced). Let us check for presence of missing values
# In[ ]:


iris_data.isnull().sum()


# That's great! There are no missing values in the dataset

# # Partition the dataset into features and labels

# **We need to shuffle our dataset as the examples are ordered by classes. Training without shuffling the dataset will lead to poor generalization**

# In[ ]:


iris_data_shuffled = iris_data.sample(frac=1).reset_index(drop=True)
print(iris_data_shuffled)


# **we will create a new dataframe which consists of the 4 features and another dataframe which consists of the class labels**

# In[ ]:


#drop the Species column. Note that we are not doing it inplace so the original dataframe will not be modified
train_x = iris_data_shuffled.drop('Species',axis=1)
print(train_x.head())


# In[ ]:


train_y = iris_data_shuffled['Species']
print(train_y.head())


#  **Let us convert the labels into numeric values. We use LabelEncoder for the same**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(train_y)
print(labels)


# convert train_x to numpy array

# In[ ]:


X = train_x.values
y = labels
print(X.shape)
print(y.shape)


# **Let us perform logistic regression. First we split the data into training and testing sets**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)
print('no. of training samples:',len(X_train))
print('no. of testing samples:',len(X_test))


# **Let us train the model using logistic regression**

# In[ ]:


lr_model = LogisticRegression(multi_class='multinomial')
lr_model.fit(X_train,y_train)


# In[ ]:


y_pred = lr_model.predict(X_test)
print(y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('test accuracy:',accuracy_score(y_test,y_pred))


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:




