#!/usr/bin/env python
# coding: utf-8

# 

# **Lets import libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import seaborn as sns


# **Read our csv file**

# In[ ]:


Data = pd.read_csv("../input/SolarPrediction.csv")
Data.head(10)


# In[ ]:


Data.info()


# In[ ]:





# In[ ]:


X = Data.iloc[:, 4:8]    
Y = Data['Radiation']
X.head()


# **Make cross-validation dataset**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape )
print(y_train.shape,y_test.shape)


# **Create correlation matrix**

# In[ ]:


corr = Data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.show()


# **Lets see the exact values of correlations**

# In[ ]:


X1 = Data['Temperature'] 
X2 = Data['Pressure']


# **Lets see the exact values of correlations**

# In[ ]:


print ('Temperature and Radiation',np.corrcoef(X1,Y),'\n')
print ('Pressure and Radiation',np.corrcoef(X2,Y),'\n')


# **Train and test the model** 

# In[ ]:


from sklearn import linear_model
reg = linear_model.BayesianRidge()
reg.fit(X_train ,y_train)
reg.score(X_test, y_test)

