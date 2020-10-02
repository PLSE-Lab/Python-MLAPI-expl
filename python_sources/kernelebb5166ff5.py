#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


pip install sckit-learn


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
data=pd.read_csv('zomato.csv')
X=data.iloc[:,0].values.reshape(-1,1)
Y=data.iloc[:,1].values.reshape(-1,1)
linear_regressor=LinearRegression()
linear_regressor.fit(X,Y)
Y_pred=linear_regressor.predict(X)
plt.scatter(x,Y)
plt.plot(X,Y_pred,color='red')

