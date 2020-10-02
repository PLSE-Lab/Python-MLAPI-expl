#!/usr/bin/env python
# coding: utf-8

# **Predicting the given temperature as temperature in the room or temperature outside the room.**
# 
# 
# As we can see we have many variables but we are not concerned above the variables like "id" , "room_id/id" , "noted_date" because this are classified as noisy or
# we can say un-useful data because they don't have any impact on the class of our output.
# We can consider "noted_date" as a parameter but we will not in our example as it will increase the complexity of the algorithm.
# 

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


# Reading the csv file
df = pd.read_csv("/kaggle/input/temperature-readings-iot-devices/IOT-temp.csv")
# Printing the first 5 entries in the Dataframe
df.head()


# In[ ]:


# Getting the number of rows and columns
df.shape


# In[ ]:


# Getting the dimensions of the Dataframe
df.ndim


# In[ ]:


# Information regarding the dataset
df.info()


# In[ ]:


# Getting the unique values from the id columns
unique_id = df['id'].unique()
print(len(unique_id))


# In[ ]:


unique_room_id = df['room_id/id'].unique()
print(unique_room_id)


# Decribing our dataset to get the insights of dataset.

# In[ ]:


df.describe()


# Getting the userful data into the "data" dataframe

# In[ ]:


data = df.iloc[:,3:]
data.head()


# Confirming the dimensions of the data, as we can see we have 97606 rows and 2 columns

# In[ ]:


data.shape


# We can't work with text data so we encode it to get it in integer format

# In[ ]:


from sklearn.preprocessing import LabelEncoder
lec = LabelEncoder()
data['out/in'] = lec.fit_transform(data['out/in'])
data.head()


# **Here '0' means 'in' and '1' means 'out'.**

# Plotting the pair plot of the data

# In[ ]:


import seaborn as sns
sns.pairplot(data = data , hue = 'out/in')


# Plotting a scatter plot of temperature against the class

# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(data['temp'],data['out/in'])
plt.title('Temperatur Scatter Plot')
plt.xlabel('Temperature')
plt.ylabel('IN/OUT')
plt.show()


# Converting the data into numpy array

# In[ ]:


X = data['temp'].values
print(X[0:5])


# In[ ]:


Y = data['out/in'].values
print(Y[0:5])


# Reshaping the data, becuase after converting into numpy it will convert the dataframe into an array of values,
# so we will have to reshape it to be rows x columns

# In[ ]:


X.shape
X = X.reshape(-1,1)
X.shape


# As their is a diversity between input and output variables or we can say dependent and independent variables, we standardize the data to get better results

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X)
X = sc.transform(X)
print(X[0:5])


# Splitting data into training and testing

# In[ ]:


from sklearn.model_selection import train_test_split
X_train , x_test , Y_train , y_test = train_test_split(X,Y,test_size = 0.3)
print(X_train[0:5])


# In[ ]:


print(len(X_train))


# In[ ]:


print(len(x_test))


# Training our classifier on Logistic Regression as this is a cassification problem and we don't have diverse data

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'liblinear')
classifier.fit(X_train,Y_train)


# Cross validating the classifier on different partitions of data

# In[ ]:


from sklearn.model_selection import cross_validate
results = cross_validate(classifier , X , Y , cv=5)
print(results)


# Predicting the values for x_test

# In[ ]:


y_pred = classifier.predict(x_test)


# Finding the accuracy score of our classifier

# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))


# > **Using Support Vector Machine for Classification**

# I know that the data is not large, but the data has some temp reading that are corresponding to both in and out, so to split them, we have to introduce a kernel for that we use SVM.

# Training our model on SVM

# In[ ]:


from sklearn.svm import SVC
classifier2 = SVC(gamma = 'auto')
classifier2.fit(X_train,Y_train)


# Cross Validating the SVM classifier on different distributions of data

# In[ ]:


SVM_results = cross_validate(classifier2 , X, Y , cv=5)
print(SVM_results)


# Predicting and checking the accuracy score of our SVM model

# In[ ]:


y_pred_classifier2 = classifier2.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred_classifier2,y_test)


# As we can see SVM performed better then Logistic Regression as the their were some observations that fell into both classes , due to the hyperplane technique of SVM. it was able to classify the temp is of what class , if it would have been a clear classification problem Logistic Regression would have performed well on this dataset.
# So the final conclusion is we can assure a 80% accuracy from SVM and a 76% accuracy from Logistic Regression.
