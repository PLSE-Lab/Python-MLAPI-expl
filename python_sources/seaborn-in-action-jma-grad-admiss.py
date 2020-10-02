#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import pandas as pd

dat = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
dat.head()


# In[ ]:


sns.lmplot(x="GRE Score", y="TOEFL Score", hue="Research", data=dat)


# In[ ]:


sns.lmplot(x="GRE Score", y="TOEFL Score", hue="University Rating", data=dat)


# In[ ]:


dat1 = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
dat1.head()


# In[ ]:


sns.lmplot(x="GRE Score", y="TOEFL Score", hue="Research", data=dat1)


# In[ ]:


sns.lmplot(x="GRE Score", y="TOEFL Score", hue="University Rating", data=dat1)

