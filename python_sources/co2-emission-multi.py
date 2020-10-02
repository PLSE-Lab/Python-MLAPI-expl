#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


FuelConsumptionCo2 = "/kaggle/input/FuelConsumptionCo2.csv"
fuel = pd.read_csv(FuelConsumptionCo2)
fuel.head()


# In[ ]:


x = np.asarray(data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])

