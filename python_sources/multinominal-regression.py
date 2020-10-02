#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


glass = pd.read_csv('../input/glass.csv',sep='\,', 
                  names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "glasstype"])
glass.head()


# In[ ]:


print(glass.shape)
print('\n',glass.glasstype.head())
print(glass.glasstype.shape)
print(np.unique(glass.glasstype))


# In[ ]:


y=glass.glasstype
x=glass.drop('glasstype',axis=1)
y


# In[ ]:


#spliting the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x_train.head()


# In[ ]:


y_train.head()


# In[ ]:


x_train.shape
x_test.head()


# In[ ]:


# #Standardize the data:
# 
# scaler = StandardScaler()
# scaler.fit(x_train)
# # Apply transform to both the training set and the test set.
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# In[ ]:


#Fit the model:
model = LogisticRegression(solver = 'lbfgs')
model.fit(x_train, y_train)


# In[ ]:


# use the model to make predictions with the test data
y_pred = model.predict(x_test)
# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# In[ ]:


#### plot the grapth 
from sklearn.linear_model import LinearRegression as lm
model=lm().fit(x_train,y_train)
predictions = model.predict(x_test)
import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.show()


# In[ ]:


#plotting the graph between predi
fig, ax = plt.subplots()
ax.scatter(y_test, predictions)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

