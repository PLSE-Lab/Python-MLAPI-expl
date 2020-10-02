#!/usr/bin/env python
# coding: utf-8

# # Load Libraries

# In[ ]:


import pandas as pd
import numpy as numpy
from sklearn.neural_network import MLPClassifier


# # Load Dataset 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


data.info()


# In[ ]:


#Get Target data 
y = data['Outcome']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['Outcome'], axis = 1)


# In[ ]:


X.head()


# # Check X Variables

# In[ ]:


#Check size of data
X.shape


# In[ ]:


X.isnull().sum()
#We do not have any missing values


# # Build Model

# In[ ]:


nnModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter = 1000)


# In[ ]:


nnModel.fit(X,y)


# # Check Accuracy

# In[ ]:


print (f'Accuracy - : {nnModel.score(X,y):.3f}')


# # END
