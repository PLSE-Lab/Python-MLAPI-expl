#!/usr/bin/env python
# coding: utf-8

# # Video and Codes for Basic Logistic Regression
# * Code: https://www.kaggle.com/funxexcel/p1-sklearn-logistic-regression
# * Video: https://www.youtube.com/watch?v=tI_Pco7snZw

# # Load Libraries

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# # Load Dataset 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')


# In[ ]:


#Get Target data 
y = data['diagnosis']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['id','diagnosis','Unnamed: 32'], axis = 1)


# # Check X Variables

# In[ ]:


X.isnull().sum()
#We do not have any missing values


# In[ ]:


X.head()


# In[ ]:


#Check size of data
X.shape


# # Build Logistic Regression with Hyperparameter

# In[ ]:


logModel = LogisticRegression()


# In[ ]:


param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]


# ### More on Hyperparameters 
# * Solver: https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451
# * L1 and L2 Regularisation: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
# * Slearn Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# ## Import Grid Search

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


clf = GridSearchCV(logModel, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)


# In[ ]:


best_clf = clf.fit(X,y)


# In[ ]:


best_clf.best_estimator_


# # Check Accuracy

# In[ ]:


print (f'Accuracy - : {best_clf.score(X,y):.3f}')


# # END
