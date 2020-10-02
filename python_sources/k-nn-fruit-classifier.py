#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

fruits = pd.read_table('../input/fruit_data_with_colors.txt') #Load data file 


# In[ ]:


fruits.head() #Dump out the first few rows of the data frame and looking at the data set


# In[ ]:


fruits.shape #Marking the size of the data(table)


# In[ ]:


#Since our only source of lable data is the data given so we split the dataset into 2 parts to estimate how well classifier will do on future samples

X= fruits[['mass','width','height']] 
y = fruits[['fruit_label']]
X_train , X_test , y_train , y_test = train_test_split(X,y,random_state =0) #Creating training and test splits


# In[ ]:


X_train.describe() #Getting a description on how our data to be trained looks like


# In[ ]:


X_test.describe() #Getting the description of the data to be tested in order to mark the difference 


# In[ ]:


#Defining a dictionary takes a numeric fruit label as input key and returns a string with the name of fruit   
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
lookup_fruit_name


# In[ ]:


#Create classifier object
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()


# In[ ]:


#Training the classifier using the training data
knn.fit(X_train,np.ravel(y_train))


# In[ ]:


#Estimating the accuracy
knn.score(X_test,y_test)


# In[ ]:


#Testing k-NN classifier to classify new objects that have never been tested before
fruit_prediction= knn.predict([[300,7,10]])
lookup_fruit_name[fruit_prediction[0]]


# In[ ]:


fruit_prediction= knn.predict([[20,4.3,5]])
lookup_fruit_name[fruit_prediction[0]]

