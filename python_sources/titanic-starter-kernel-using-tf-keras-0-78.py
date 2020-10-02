#!/usr/bin/env python
# coding: utf-8

# # **Hello guys, I'm Jesudas DSouza, a mentor at SkillConnect, Mumbai. This is a demo tensorflow + keras tutorial that should get you started in Tensorflow. I aim to simplify everything as much as I can. This should be enough for you to get started.**
# **
# 
# *Tools i used in this notebook*
# 
#  matplotlib ---> data visualization ::: you can use this resource on kaggle learn https://www.kaggle.com/learn/data-visualization
#  
#  pandas ---> dataframe ::: you can utilise this resource on kaggle learn https://www.kaggle.com/learn/pandas
#  
#  numpy ---> easy scientific computing with python ::: #will update this kernel if i find a good course
#  
#  keras API with TensorFlow backend ---> making the neural network ::: #you should pretty much understand through this starter notebook
#  
#  *If you found this kernel useful, please upvote it*
#  *If you found a step / that i havent covered in my explanation, please leave a comment. I'll add an explanation in an update*
#  
#  **An update : In th initial kernel, I had not taken the Embarked feature. In this kernel, I'm including it.**

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


# On running the above code cell, we get the paths to the datasets. Load these paths into train and test path variables. Here, we have test_path and train_path. Now, using Pandas read_csv function, we load our csv files into a dataframe. Here, the train_data contains the dataframe for train.csv and likewise, test_data.  

# In[ ]:


test_path = '/kaggle/input/titanic/test.csv'
train_path = '/kaggle/input/titanic/train.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


# In[ ]:


train_data.head(10)

the head() function gives the top 5 rows in the dataset. You can display any no. of rows, by inserting the no. of rows you want inside the head() function. eg: train_data.head(10) will return 10 rows
# In[ ]:





# In[ ]:


train_data.head()
train_data.shape#(891, 12)
train_shape = train_data.shape[0]
test_shape = test_data.shape[0]


# In[ ]:


test_data.shape


# **Lets do some Visualization!** #visualization will be another update

# In[ ]:





# Now, let us observe our data. Are there missing values? lets check

# In[ ]:


train_data.isnull().sum()


# Oh Boy! Out of 891 entries (rows), Age data for 177 entries and Cabin data for 687 entries is missing! How should we deal with missing data? well, there are 3 ways
# 1. Ignore features with missing data
# 2. Replace missing values with average/median of the feature values
# 3. Assume a value for the feature
# for now, we'll use method 2, because it is the most logical option right now.
# 
# But, before that, lets select our most important features.
# 

# In[ ]:


train_data['Embarked'].fillna('S', inplace = True)
test_data['Embarked'].fillna('S', inplace = True)


# In[ ]:


cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_Train = train_data[:train_shape][cols]
X_test = test_data[:test_shape][cols]
y = train_data[:train_shape]['Survived'].astype(int) 


# In[ ]:





# **Now, we'll need to encode labels i.e, male and female. Since An ANN cannot really process a dataset which has string in a dataframe, we need to encode it.**

# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_Train = X_Train.copy()
label_X_test = X_test.copy()
Gender_col = ['Sex']
Embarked_col = ['Embarked']
# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col1 in Gender_col:
    label_X_Train[col1] = label_encoder.fit_transform(X_Train[col1])
    label_X_test[col1] = label_encoder.transform(X_test[col1])
    
for col2 in Embarked_col:
    label_X_Train[col2] = label_encoder.fit_transform(X_Train[col2])
    label_X_test[col2] = label_encoder.transform(X_test[col2])    


# In[ ]:





# In[ ]:


label_X_test.head()


# In[ ]:





# In[ ]:


label_X_Train.head()


# In[ ]:





# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer()


# In[ ]:


imputed_label_X_Train_data = pd.DataFrame(imputer.fit_transform(label_X_Train))
imputed_label_X_test_data = pd.DataFrame(imputer.transform(label_X_test))
imputed_label_X_Train_data.columns = label_X_Train.columns
imputed_label_X_test_data.columns = label_X_test.columns


# In[ ]:


#imputed_label_X_Train_data.shape
imputed_label_X_test_data.shape


# # Now, let us create a train-validation split. We'll use sklearn's train_test_split. Note, this is just made for testing our model. In a later kernel i'll be covering overfitting and underfitting

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(imputed_label_X_Train_data, y, test_size = 0.2)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
print(tf.__version__)


# In[ ]:


my_ann = Sequential()#initialising the ANN
my_ann.add(Dense(units = 4, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 7))
my_ann.add(Dense(units = 2, kernel_initializer = 'glorot_uniform', activation = 'relu'))
my_ann.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))


# ![This is a sigmoid graph](https://hvidberrrg.github.io/deep_learning/activation_functions/assets/sigmoid_function.png)

# **Notice how the curve goes from 0 to 1 on the y axis? This shows that the last layer of our ANN model will have values between 0 and 1**

# ![](https://cdn.tinymind.com/static/img/learn/relu.png)**For this graph, its basically a rectifier graph or more commonly known as a relu activtion**

# **We perform Stochastic Gragient Descent, more specifically 'adam' optimizer. You must have heard about Gradient Descent, right? For those uninitiated, it deals with finding the minima of the equation. Now, some of you must have already found a problem with this, i.e. a situation where we are stuck at the local minima of the graph rather than finding the global minima of the graph. This is where SGD or stochastic gradient descent comes into picture**
# 
# **With SGD, or any normal GD for the matter, the loss is always calculated in the Logarithm of the values. Here, we use binary cross entropy becayse out classification deals with the survival of a passenger i.e dead or alive**

# In[ ]:


my_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
my_ann.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose = 1)


# In[ ]:


y_pred = my_ann.predict(X_valid)
y_pred = [1 if y>=0.5 else 0 for y in y_pred]#list comprehension


# In[ ]:


from sklearn.metrics import f1_score
F1Score = f1_score(y_pred, y_valid)
print(F1Score)#f1_score of validation dataset


# In[ ]:


my_ann.fit(imputed_label_X_Train_data, y, batch_size = 10, epochs = 100, verbose = 1)


# In[ ]:


y_final = my_ann.predict(imputed_label_X_test_data)
y_final = [1 if y>=0.5 else 0 for y in y_final]


# In[ ]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_final })
output.to_csv("submission.csv", index = False)
print("done!")


# In[ ]:




