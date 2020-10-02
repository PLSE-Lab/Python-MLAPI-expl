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


#    The next piece of code imports the dataset in csv format.And, it converts the dataset into a panda dataframe.

# In[ ]:


heart_d=pd.read_csv("../input/heart.csv")
heart_d.head()


# In[ ]:


heart_d.describe()


# In[ ]:


heart_d.info()


# In[ ]:


heart_d.isnull()


# From the above operations on the dataset we can conclude that all the values are numerical and does not have any missing value.
# Luckily there is no missing value!!

# **Data Visualization**
# This is the most important part of data preprocessing. It give us an idea how the features of the data ar related  to each other.
# There are many powerful plotting libraries provided by python. Some of the important plotting libraries are matplotlib, seaborn, etc.
# 

# FIrst, we plot each one of the features with respect to another. This gives us an empirical idea about the relation between each other. We plot this using **pairplot** function in **seaborn** library.

# 

# In[ ]:


import seaborn as sns
g=sns.pairplot(heart_d)


# Now, we go for plotting a heatmap that plots the correlation between each one of the features with another feature.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplots(figsize=(10,8))
sns.heatmap(heart_d.corr(),annot=True,linewidths=0.8,cmap='coolwarm')


# Now, we will plot scatter plot between two features, one after another two relate between them. We will use functions from both libraries matplotlib and seaborn.

# In[ ]:


plt.scatter(heart_d['chol'],heart_d['thalach'])


# In[ ]:


sns.jointplot(x='chol',y='thalach',data=heart_d,kind='kde',color="g")


# Now, we will start visualizing each feature one after one. Also, known as univariate visualization. Along, with this we will also plot each of the feature to study the characteristics of the data.

# In[ ]:


ax=plt.subplots(figsize=(10,8))
sns.boxplot(data=heart_d['trestbps'])


# In[ ]:


sns.barplot(heart_d['target'],heart_d['trestbps'])


# In[ ]:


plt.subplots(figsize=(10,8))
sns.boxplot(data=heart_d['chol'])


# In[ ]:


sns.barplot(heart_d['target'],heart_d['chol'])


# In[ ]:


sns.boxplot(data=heart_d['oldpeak'])


# In[ ]:


sns.barplot(heart_d['target'],heart_d['oldpeak'])


# Now, we come to the point where we divide the dataset between training and test set.
# As, this is a high quality dataset we do not need to divide it into a validation set.

# In[ ]:


from sklearn.model_selection import train_test_split
X=heart_d.drop("target",axis=1)
Y=heart_d["target"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=42)


# After this if we use shape function to the train and test sets we will see the number of samples taken into them.

# Now, comes the model fitting task here we import models from the sklearn library and also use the keras library to build a neural network.
# Also, we will check the accuracy of the model along with each of the application. 

# In[ ]:


from sklearn.metrics import accuracy_score


# **Support Vector Machine**

# In[ ]:


from sklearn import svm
sv=svm.SVC(kernel='linear')
sv.fit(X_train,Y_train)
pred_sv=sv.predict(X_test)


# In[ ]:


pred_sv.shape


# In[ ]:


score_svm=accuracy_score(pred_sv,Y_test)
print("The accuracy achieved using svm is"+" "+str((score_svm)*100))


# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(X_train,Y_train)
pred_lg=lg.predict(X_test)


# In[ ]:


score_lgr=accuracy_score(pred_lg,Y_test)
print("The accuracy achieved using Logistic regression"+" "+str((score_lgr)*100))


# **Neural Network**

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(11,input_dim=13,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


model.fit(X_train,Y_train,epochs=250)


# In[ ]:


pred_nn=model.predict(X_test)


# In[ ]:


round_nn=[round(x[0]) for x in pred_nn]
score_nn=accuracy_score(round_nn,Y_test)*100
print("The accuracy score of the neural network is"+" "+str(score_nn))


# In[ ]:


score=[score_svm*100,score_lgr*100,score_nn]
algorithms=["Support Vector Machine","Logistic Regression", "Neural Network"]
sns.barplot(algorithms,score)


# So we can see that all the algorithms gives the same accuracy as the data is of very high quality.
