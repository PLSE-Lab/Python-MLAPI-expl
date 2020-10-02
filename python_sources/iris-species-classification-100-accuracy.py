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


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset=pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


dataset.head()


# * our dataset contains the 6 cloumns and 150 rows. the last column is our dependant variable which tell the type of iris species in each of the features.
# 

# In[ ]:


dataset.head()


# the different types of the iris species and their respective numbers in the dataset is shown below

# In[ ]:


dataset['Species'].value_counts()


# 

# In[ ]:


sns.countplot(x='Species',data=dataset)
plt.savefig('count sprecied in train data.png')


# now since we have to fit the model to the logistic regression model we will have to convert the different species into labels for them to fit into the model other wise the model will cause an error

# In[ ]:


species={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}


# In[ ]:


dataset=dataset.replace({'Species':species})


# In[ ]:


dataset.head()


# we can now see that the statistics of our dataset

# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# ## Some features engineering

# # we will now see which of the columns contain the most variance in the species prediction

# In[ ]:


dataset.corr()[['Species']].sort_values(by='Species',ascending=False)


# we will remove our ID column column as it is only the serial number and does not convey any relevant information about the prediction

# In[ ]:


dataset.drop('Id',axis=1,inplace=True)


# In[ ]:


dataset.head()


# again finding the variance of the dataset

# In[ ]:


dataset.corr()[['Species']].sort_values(by='Species',ascending=False)


# we can see that the petal width and petal length convey more than 95 percent of the information for prediction of the species. sepal width convey negetive relation 

# In[ ]:


sns.regplot(x='PetalLengthCm',y='Species',data=dataset)
plt.savefig('dependance of petal length of iris flowers on prediction.png')


# In[ ]:


sns.regplot(x='PetalWidthCm',y='Species',data=dataset)
plt.savefig('dependance of petal width of iris flowers on prediction.png')


# In[ ]:


sns.regplot(x='SepalLengthCm',y='Species',data=dataset)
plt.savefig('dependance of sepal length of iris flowers on prediction.png')


# In[ ]:


sns.regplot(x='SepalWidthCm',y='Species',data=dataset)
plt.savefig('dependance of sepal width of iris flowers on prediction.png')


# now we split the dataset into our independant feature matrix and depeandant feature matrix

# In[ ]:


x_train=dataset.drop('Species',axis=1)
y_train=dataset[['Species']]


# now we split our dataset into training and test set for training and predicting respectively

# In[ ]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x_train,y_train,test_size=0.2,random_state=18)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# > **now we build our logistic regression classifier****

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(C=112,max_iter=50)


# **logistic regression object/instance craeted 
# now we will fit the object to the training data **

# In[ ]:


clf.fit(x_train,np.array(y_train).ravel())


# predicting values from the fitted classifier for the test data

# In[ ]:


y_predict=clf.predict(x_test)


# checking accuracy :-

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_predict)


# all the samples of the test data have been classified correctly

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)*100


# **accuracy of the model is 100 percent**

# In[ ]:


y_predict=pd.DataFrame(y_predict)
sns.countplot(x=0,data=y_predict)
plt.savefig('y_predict data.png')


# In[ ]:




