#!/usr/bin/env python
# coding: utf-8

# ![](http://)![](http://)# k-NN classification on Iris flower dataset

# Importing the Iris dataset

# In[ ]:


from sklearn import datasets
myiris = datasets.load_iris()
x = myiris.data
y = myiris.target
type(x)
x.shape
type(y)


# Scaling the data using Min-max scalar

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x)


# Splitting data into training and testing sets

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


# Importing the Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# Instantiating the model with 3 neighbors

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)


# Training the model

# In[ ]:


knn.fit(x_train, y_train)


# Making predictions on test data

# In[ ]:


y_pred=knn.predict(x_test)


# Evaluating the model

# In[ ]:


#Accuracy
print('Accuracy = ', knn.score(x_test, y_test))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
print('\nConfusion matrix')
print(confusion_matrix(y_test, y_pred))

#Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report')
print(classification_report(y_test, y_pred))  

