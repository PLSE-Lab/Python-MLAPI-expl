#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[ ]:


# initialize list of lists 
data = [[2600, 550000], [3000, 565000], [3200, 610000],[3200, 550000],[3600, 680000],[4000, 550000],[4400, 725000]] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['area', 'price']) 
  
# print dataframe. 
df


# In[ ]:


#df = pd.read_csv("homeprices.csv")
df.head()


# In[ ]:


model = linear_model.LinearRegression()
model.fit(df[['area']],df.price)


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


model.predict([[5000]])


# ###### Save Model To a File Using Python Pickle

# In[ ]:


import pickle


# In[ ]:


with open('model_pickle','wb') as file:
    pickle.dump(model,file)


# #### Load the saved model

# In[ ]:


with open('model_pickle','rb') as file:
    mp = pickle.load(file)


# In[ ]:


mp.coef_


# In[ ]:


mp.intercept_


# In[ ]:


mp.predict([[5000]])


# #### Save Trained Model Using joblib

# In[ ]:


#!pip install joblib
get_ipython().system('pip install sklearn.externals')


# In[ ]:


import sklearn.external.joblib as extjoblib
import joblib

from sklearn.externals import joblib


# In[ ]:


joblib.dump(model, 'model_joblib')


# #### Load Saved Model

# In[ ]:


mj = joblib.load('model_joblib')


# In[ ]:


mj.coef_


# In[ ]:


mj.intercept_


# In[ ]:


mj.predict([[5000]])


# In[ ]:




