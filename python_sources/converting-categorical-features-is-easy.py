#!/usr/bin/env python
# coding: utf-8

# # Objectives
# ### Hello Kaggler!, <span style="color:PURPLE">Objective of this short kernal is to</span> <span style="color:red">demonstrate simple ways you could use to convert Categorical Features</span>
# 
# Curerntly this kernel demonstrates following encoding techniques,
# * One-Hot-Encoding (OHE) (dummy encoding)
# * Label Encoding
# 
# If you have ideas on improving this kernel please comment!
# 
# To make this very easy to grasp I have used infamouse Titanic data set to train the ML model.
# 
# 
# 
# 
# ### Let's Start!

# # Why do we need to convert categorical features?
# 
# * Because Many machine learning tools will only accept numbers as input.

# ### Preparing Example Dataset for demonstration

# In[ ]:


import numpy as np 
import pandas as pd 
import os
data = pd.read_csv("../input/train.csv")
data.drop(columns = ['Name','Ticket','Fare','Cabin','SibSp','Parch','Age','PassengerId','Survived','Pclass'], inplace=True)
data['Embarked'].fillna('S',inplace=True)
data.head(10)


# In[ ]:


print('Unique Values of Columns')
print('\tSex \t\t: ',data.Sex.unique())
print('\tEmbarked \t: ',data.Embarked.unique())


# # Method 1: One-Hot-Encoding (OHE) (dummy encoding) 

# Learn on One-Hot-Encoding :https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/
# 
# #### Code & Output

# In[ ]:


columnsToEncode = ['Sex','Embarked']
One_Hot_encoded = pd.get_dummies(data,columns= columnsToEncode)
One_Hot_encoded.head()


# # Method 2: Label Encoding

# Learn on Label Encoding : https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
# 
# lets lable encode columns 'Sex' and 'Embarked'

# ## Initial data frame

# In[ ]:


data.head()


# ## Using sklearn.preprocessing.LabelEncoder

# In[ ]:


# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()

data_for_Label_Encoding = data.copy()
data_for_Label_Encoding['Sex'] = le.fit_transform(data_for_Label_Encoding[['Sex']])
data_for_Label_Encoding['Embarked'] = le.fit_transform(data_for_Label_Encoding[['Embarked']])


# In[ ]:


data_for_Label_Encoding.head()


# ## Using pandas.DataFrame.replace

# In[ ]:


data_for_Label_Encoding = data.copy()
data_for_Label_Encoding['Sex'].replace(['male','female'],[1,2],inplace=True)
data_for_Label_Encoding['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data_for_Label_Encoding.head()


# # Credits
# 
# * https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
# * https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8
# * https://chrisalbon.com/machine_learning/preprocessing_structured_data/convert_pandas_categorical_column_into_integers_for_scikit-learn/
# * https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
# * https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/

# # Learn more on Categorical Feature Encoding
# * https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
# * https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8
# * https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
# * https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/

# ## Thank you!
# ### If you like the notebook and think that it helped you..PLEASE UPVOTE. It will keep me motivated :) :)
