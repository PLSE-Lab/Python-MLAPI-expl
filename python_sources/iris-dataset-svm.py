#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


iris = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')


# In[ ]:


iris


# In[ ]:


iris.describe()


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x = iris.drop('species',axis=1)
y = iris['species']


# In[ ]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3,random_state=101)


# In[ ]:


x_train.shape , y_train.shape


# In[ ]:


x_test.shape , y_test.shape


# # SVM Model

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model = SVC()


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


pred = model.predict(x_test)


# # Model Evaluation

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,pred))
print("\n")
print(classification_report(y_test,pred))


# In[ ]:




