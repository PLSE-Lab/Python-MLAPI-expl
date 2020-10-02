#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imported Libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


# data = pd.read_csv("/content/drive/My Drive/creditcard.csv")
# data.head(10)


# In[ ]:


data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head(10)


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


sns.countplot(data['Class'])


# In[ ]:


noFraud = len(data[data.Class == 0.000000])
Fraud = len(data[data.Class == 1.000000])
print("Fair trasactions: {:.2f}%".format((noFraud / (len(data.Class))*100)))
print("Fraud trasactions: {:.2f}%".format((Fraud / (len(data.Class))*100)))


# In[ ]:


# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = data, target = 'Class')


# In[ ]:


compare_models(fold=5)


# In[ ]:


# creating logistic regression model
model = create_model('xgboost')


# In[ ]:


model


# In[ ]:


model=tune_model('xgboost')


# In[ ]:




