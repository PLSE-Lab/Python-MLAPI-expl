#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd #importing all the necessary libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


# In[ ]:


cancer = load_breast_cancer() #loading our data


# In[ ]:


cancer.keys()


# In[ ]:


df = pd.DataFrame(cancer['data'],columns = cancer['feature_names']) #putting our data in a Dataframe


# In[ ]:


df.head() #checking the head of the data


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum() #checking for any sort of null value in our data


# In[ ]:


sns.heatmap(df.isnull()) #looking for null values with help of heat map


# In[ ]:


from sklearn.model_selection import train_test_split #to split our data into training and testing set


# In[ ]:


x = df
y = cancer['target']


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 101) #splitting our data


# In[ ]:


from sklearn.svm import SVC #importing the svm model


# In[ ]:


svc  =SVC()


# In[ ]:


svc.fit(x_train,y_train) #fitting the data to our model


# In[ ]:


pred = svc.predict(x_test) #predicting the result


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred)) #we print our results and its quite decent but it can be improved by using GridSearch which would help us find better hyperparameters for our problem


# In[ ]:


from sklearn.model_selection import GridSearchCV #importing Gridsearch


# In[ ]:


param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,verbose = 3)


# In[ ]:


grid.fit(x_train,y_train)


# In[ ]:


grid.best_estimator_


# In[ ]:


grid.best_params_


# In[ ]:


grid_pred = grid.predict(x_test)


# In[ ]:


print(classification_report(y_test,grid_pred))
print('\n')
print(confusion_matrix(y_test,grid_pred))


# We were able to get slightly better results upon using the GridSearch. This problem can be tackled using many other algorithms but i had never worked on it using SVM and i am pleased with the result

# In[ ]:




