#!/usr/bin/env python
# coding: utf-8

# **** Importing the Libraries and Loading the dataset

# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

iris = pd.read_csv('/kaggle/input/iris/Iris.csv')


# Getting the Headings

# In[ ]:


iris.head()


# In[ ]:


iris.tail()


# In[ ]:


# create pairplot of the dataset, which flower species seems to be most separable


# In[ ]:


sns.pairplot(iris, hue='Species')


# In[ ]:


# create a kde plot of sepal_length versus sepal width for sentosa species of flower


# setosa= iris[iris['Species']=='setosa']
# sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True)

# In[ ]:


setosa= iris[iris['Species']=='setosa']sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True)


# # train test split

# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


X=iris.drop('Species',axis=1)
y=iris['Species']
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3)


# Train Model

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc_model = SVC()


# In[ ]:


svc_model.fit(X_train, y_train)


# Model Evaluation

# #now get predictions from the model and create a confusion matrix and a aclassification report

# In[ ]:


prediction =svc_model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,prediction))


# In[ ]:


print(classification_report(y_test, prediction))


# Grid search

# In[ ]:


from sklearn.model_selection import GridSearchCV


# Create a dict called param_grid and fillout some parameters for C and gamma

# In[ ]:


param_grid ={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}


# Create a GridSearchCV object and fit it to the training data

# In[ ]:


grid =GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)


# Now take taht grid model and create some predictios using the test set and create classification report and classification reports and confusion metrics for them.

# In[ ]:


grid_predictions=grid.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, grid_predictions))


# In[ ]:


print(classification_report(y_test,grid_predictions))


# In[ ]:




