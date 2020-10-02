#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


import pandas as pd
from pandas import Series,DataFrame

data_df = pd.read_csv('/kaggle/input/titanic/train.csv')
data_df.head()


# In[ ]:


from pycaret.classification import *
clf1 = setup(data_df, target = 'Survived', ignore_features = ['Name', 'Ticket', 'PassengerId'])


# In[ ]:


compare_models()


# In[ ]:


tuned_lightgbm = tune_model('lightgbm', optimize = 'AUC')


# In[ ]:


evaluate_model(tuned_lightgbm)


# In[ ]:


final_lightgbm = finalize_model(tuned_lightgbm)


# In[ ]:


print(final_lightgbm)


# In[ ]:


test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_df.head()


# In[ ]:


predictions = predict_model(final_lightgbm, data = test_df)


# In[ ]:


predictions.head()


# In[ ]:




