#!/usr/bin/env python
# coding: utf-8

# # **Loading Data**

# In[ ]:


import pandas as pd
data = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


data.head()


# # **Installing PyCaret**

# In[ ]:


get_ipython().system('pip install pycaret')


# # **Initializing Setup**

# In[ ]:


from pycaret.classification import *
clf1 = setup(data, target = 'Survived', ignore_features = ['Ticket', 'Name', 'PassengerId'], silent = True, session_id = 786) 

#silent is True to perform unattended run when kernel is executed.


# # **Compare Models**

# In[ ]:


compare_models()


# # **Create Model**

# In[ ]:


catboost = create_model('catboost')


# # **Tuning Catboost**

# In[ ]:


tuned_catboost = tune_model('catboost', optimize = 'AUC', n_iter = 100)


# # **Loading Testset**

# In[ ]:


test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()


# # **Generate Predictions**

# In[ ]:


predictions = predict_model(tuned_catboost, data = test)


# In[ ]:


predictions.head()


# # THANK YOU
