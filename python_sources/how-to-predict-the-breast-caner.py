#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression


# In[ ]:


data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv").dropna(how="all").drop(labels="Unnamed: 32", axis=1)


# In[ ]:


X = data.drop( labels="diagnosis", axis=1 )
y = data["diagnosis"].to_list()
for i in range(len(y)):
    if( y[i]=="B" ):
        y[i]=1
    else:
        y[i]=0
y = pd.Series(y)

print( y )


# In[ ]:


print( f_regression(X, y) )


# In[ ]:


reg_model = LinearRegression()
reg_model.fit(X, y)


# In[ ]:


print( reg_model.score(X, y) )


# In[ ]:


print( len(X.columns) )


# As we see, for predicting value of parameter 'diagnosis' we can use linear model, based in all other parameters, and this model will be good.

# In[ ]:


#print of coeffitients of linear model
print( reg_model.coef_ )

