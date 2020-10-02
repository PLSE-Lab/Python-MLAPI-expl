#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing libraries that we will need

import numpy as np # linear algebra i.e for short matrix multiplication
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #For Plotting graphs


# In[ ]:


#Reading our csv file and storing it into dataframe using pandas 

dataframe = pd.read_csv("../input/heart.csv")

#Printing the first five instances of our dataframe
dataframe.head()


# In[ ]:


# Describing our data using the inbuilt pandas function describe()

dataframe.describe()


# In[ ]:


#Getting the info of our data

dataframe.info()


# In[ ]:


dataframe.plot()
plt.show()


# In[ ]:


dataframe.hist()
plt.show()


# In[ ]:


dataframe.target.unique()


# In[ ]:


dataframe.ca.unique()


# In[ ]:


dataframe.fbs.unique()


# In[ ]:


X = dataframe.iloc[:,0:13].values
print(X[0:5,:])


# In[ ]:


Y = dataframe["target"].values
print(Y[0:5])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test = train_test_split(X,Y)


# In[ ]:


print(X_train[0:5,:])


# In[ ]:


print(x_test[0:5,:])


# In[ ]:


print(Y_train[0:5])


# In[ ]:


print(y_test[0:5])


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)


# In[ ]:


classifier.score(x_test,y_test)


# In[ ]:


x = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
classifier.predict(x)

