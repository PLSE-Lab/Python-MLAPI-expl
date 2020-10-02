#!/usr/bin/env python
# coding: utf-8

# ****Linear Regression - Using first principle****
# 

# > ***This notebook shows the model to solve linear regression using first principle also plotting of same model using matplotlib library, I will share next time how we can create model using  sklearn****
# >***Linear Regression shown below is used for bivariant type of data***
# > ***We use linear regression when corelation is not equal to zero ***
# 

# In[ ]:


import pandas as pd
import numpy as np
def funcy(a1,a2,a1m,a2m):
    m=(sum(a1*a2)-len(a1)*a1m*a2m)/(sum(a1**2)-len(a1)*(a1m**2))
    c=a2m-(m*a1m)
    return(m,c)
def errorid(a2,slop,inter):
    ycap=(slop*a1)+inter
    s=np.sqrt((sum((a2-ycap)**2))/len(a2))
    return(s,ycap)   
a1=np.array([4,9,10,14,4,7,12,22,1,17])
a2=np.array([31,58,65,73,37,44,60,91,21,84])
a1m=np.mean(a1)
a2m=np.mean(a2)
print("---------Eqaution for linear regression---------")
slop,inter=funcy(a1,a2,a1m,a2m)
print(f' y = {slop} x + {inter}')
print("---------Root of Mean Square Error---------")
l,ycap=errorid(a2,slop,inter) #I am using the metric RMSE(Root Mean Square Error) for understanding the model
print(l)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.scatter(a1,a2,color='red')
plt.plot(a1,ycap)


# Blue line is the best fit line for the linear regression model

# In[ ]:




