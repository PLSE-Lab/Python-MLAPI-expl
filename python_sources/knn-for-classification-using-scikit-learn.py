#!/usr/bin/env python
# coding: utf-8

# **Scikit-learn is a very popular Machine Learning library for Python. In this kernel let us use it to build a machine learning model using k-Nearest Neighbors algorithm to predict whether the patients in the "Pima Indians Diabetes Dataset" have diabetes or not. **

# In[ ]:


#Load the necessary python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[ ]:


#Load the dataset
df = pd.read_csv('../input/diabetes.csv')

#Print the first 5 rows of the dataframe.
df.head()


# In[ ]:


#Let's observe the shape of the dataframe.
df.shape


# As observed above we have 768 rows and 9 columns. The first 8 columns represent the features and the last column represent the target/label. 

# In[ ]:


#Let's create numpy arrays for features and target
X = df.drop('Outcome',axis=1).values
y = df['Outcome'].values


# Let's split the data randomly into training and test set. 
# 
# We will fit/train a classifier on the training set and make predictions on the test set. Then we will compare the predictions with the known labels.
# 
# Scikit-learn provides facility to split data into train and test set using train_test_split method.

# In[ ]:


#importing train_test_split
from sklearn.model_selection import train_test_split


# It is a best practice to perform our split in such a way that out split reflects the labels in the data. In other words, we want labels to be split in train and test set as they are in the original dataset. So we use the stratify argument.
# 
# Also we create a test set of size of about 40% of the dataset.

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)


# Let's create a classifier using k-Nearest Neighbors algorithm.
# 
# First let us first observe the accuracies for different values of k.

# In[ ]:


#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 


# In[ ]:


train_accuracy


# In[ ]:


#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# We can observe above that we get maximum testing accuracy for k=7. So lets create a KNeighborsClassifier with number of neighbors as 7.

# In[ ]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=7)


# In[ ]:


#Fit the model
knn.fit(X_train,y_train)


# In[ ]:


#Get accuracy. Note: In case of classification algorithms score method represents accuracy.
knn.score(X_test,y_test)


# Hope you find it useful. :)please upvote

# In[ ]:




