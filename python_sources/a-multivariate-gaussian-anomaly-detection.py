#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This is a simple text book implementation of anomaly detection that I learnt in a course in Coursera for fraud detection. 
# What I enjoyed was how 1. splitting the visualizing the data , 2. looking at relevance of variables , 3. cleanup , or 4. making the curve normal can help improving the initial proto model built.
# 
# Feel free to contact me if you want the notebook for visualization also.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score, confusion_matrix,classification_report
import math


# # Load_Data

# In[ ]:


df = pd.read_csv('../input/creditcard.csv')


# In[ ]:


df.shape


# # Drop some columns that dont seem to differentiate
# You may want to experiment with this set of columns and decide - I chose this based on a simple visualization of the ok and fraud distribution plots

# In[ ]:


col_to_drop = ['V8','V13','V15','V20','V21','V22','V23','V24','V26','V27','V28','Time']
a = df.drop(col_to_drop, axis=1)


# # Create normal curves and Mean normalize amount 
# Play around with this as it may help you to see the impact of having normally distributed variables

# In[ ]:



a['Amount']=a['Amount'].apply(lambda x: math.log(x**.5+1)) # Getting to a normal curve is done based on visualization and iteration to get the shape
a['V1']=a['V1'].apply(lambda x: x**.2) # Getting to a normal curve is done based on visualization and iteration to get the shape

mu = a['Amount'].mean()
sigma = a['Amount'].std()
a['Amount'] = (a['Amount']-mu)/sigma


# # Create a train, cross validation and test split
# I built the protoype without the train test split and later refined it with this * ideal * split to get teh CV and Test (60:20:20)

# In[ ]:


a_good = a[a['Class']==0]
a_fraud = a[a['Class']==1]


# In[ ]:


gr = len(a_good)
fr = len(a_fraud)


# In[ ]:


g_tr = a_good[:gr*60//100]
g_cv = a_good[(gr*60//100)+1:(gr*80//100)]
g_t = a_good[(gr*80//100)+1:]


# In[ ]:


fr_cv = a_fraud[:fr*50//100]
fr_t = a_fraud[(fr*50//100)+1:]


# In[ ]:


a_cv = pd.concat([g_cv,fr_cv])
a_t = pd.concat([g_t,fr_t])


# In[ ]:


g_tr.drop('Class',inplace=True,axis=1)


# # Train and create the probability distributions for the multivariate gaussian
# *** This is done only on the good values to get a more appropriate distribution

# In[ ]:


p = multivariate_normal(mean=np.mean(g_tr,axis=0), cov=np.cov(g_tr.T))


# # Get epsilon from the CV set

# In[ ]:


a_cv_X = a_cv.drop('Class',axis=1)
x = p.pdf(a_cv_X)


# In[ ]:


epsilons = [1e-60,1e-65,1e-70,1e-75,1e-80,1e-85,1e-90,1e-95,1e-100,1e-105]
pred = (x<epsilons[2])
f = f1_score(a_cv['Class'],pred,average='binary')
print(f)


# In[ ]:


f_max = 0
e_final=0

for e in epsilons:
    pred = (x<e)
    f = f1_score(a_cv['Class'],pred,average='binary')
    print(f,e)
    if f>f_max:
        f_max=f
        e_final=e


# In[ ]:


print (e_final,f_max)


# # Check the model with the final test set

# In[ ]:


a_t_X = a_t.drop('Class',axis=1)
x = p.pdf(a_t_X)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


pred = (x<e_final)
confusion_matrix(a_t['Class'],pred)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(a_t['Class'],pred))


# This is a text book implementation of Anamoly detection - the primary aim was to see if the "normalization" of the curves to a bell shape, reducing parameters to those relevant have any improvements. It can be useful for anyone who has done the Coursera Andrew Ng course as a practice using python.

# In[ ]:




