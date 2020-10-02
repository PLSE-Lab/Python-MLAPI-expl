#!/usr/bin/env python
# coding: utf-8

# In this kernal we will look into what KNN Algorithm is and how we can use it make predictions on datasets.
# I will be using the Breas Cancer Dataset for this kernal.

# ## KNN Algorithm
# It's a simple algorithm that stores all the available cases and classifies the new data based on **similarity measure**
# 
# To get a quick understanding on where you use such algorithms, take a moment how online stores like Amazon provides you with recommendations when you look up a specific item.
# ![](https://offerzen.ghost.io/content/images/2018/08/Inner-blog-image.png)
# Based on similarity measures or **nearest matching labels** the system will provide the results

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors


# In[ ]:


df = pd.read_csv('../input/breast-cancer-wisconsin.data.txt')
df.head()


# Let's quickly clean our dataset

# In[ ]:


df.replace('?', -99999, inplace=True) #Replace the missing values with an outlier number so that the algorith will identify it as an outlier
df.drop(['id'], 1, inplace=True) #id column is not useful in making predictions
df.head()


# Let's define our X and y

# In[ ]:


X = np.array(df.drop((['class']), 1))
y = np.array(df['class'])


# We will next split the data as train and test

# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)


# Now let's define our classifier which is KNN

# In[ ]:


clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)


# To see how our model is perfoming we can check its accuracy

# In[ ]:


accuracy = clf.score(X_test, y_test)
print(accuracy)


# So this states that our models accuracy is at 95% which is good for the little steps we did. But given a large dataset and the nature of it this score might not be enough. We can do more operations to improve our model. 

# If you need to make predictions you can simply input an numpy array to the model and output predictions

# In[ ]:


example = np.array([8,3,1,2,3,1,4,4,4])
example = example.reshape(1, -1)
predictions = clf.predict(example)
print("Class:"+ str(predictions))


# For a large number of data input you can use a loop and inplace of '1' in example.reshape() we will change it as example.reshape(len(example), -1) when we do not know the amount of input data.
# 
# That's it for this kernal. Hope you gained some knowledge on KNN. If you like this please upvote.

# In[ ]:




