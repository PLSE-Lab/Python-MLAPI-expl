#!/usr/bin/env python
# coding: utf-8

# L2 Regularization is a method we can use during linear regression by which we punish larger outliers in a dataset.
# This is meant to be a demonstration of the effects of L2 regularization, and here we will examine the relationship
# between total expenditure and capital outlay expenditure, which is money spent on building, busses, etc. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/elsect_summary.csv')
df.dropna(axis=0, subset=['ENROLL'],inplace=True)


# In[ ]:


df.head()


# In[ ]:


X = np.array(df['TOTAL_EXPENDITURE'])
Y = np.array(df['CAPITAL_OUTLAY_EXPENDITURE'])


# In[ ]:


plt.scatter(X,Y)


# In[ ]:


X = np.vstack([np.ones(len(X)),X]).T


# In[ ]:


w_ml = np.linalg.solve(X.T.dot(X),X.T.dot(Y))


# In[ ]:


Yhat_ml = X.dot(w_ml)


# Here is a standard linear regression between our input variable (Total Expenditure) and our output
# variable (Capital outlay expenditure)

# In[ ]:


plt.scatter(X[:,1],Y)
plt.scatter(X[:,1],Yhat_ml)


# Now we will write a for loop to determine a good "gamma" by which we can "punish" those large outliers

# In[ ]:


def r2(Y,yhat):
    d1 = Y-yhat
    d2 = Y-Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)
    return r2


# In[ ]:


r_squared_list = []
for i in range(0,20):
    l2x = 10**i
    w_mapx = np.linalg.solve(l2x*np.eye(2) + X.T.dot(X),X.T.dot(Y))
    Yhat_mapx = X.dot(w_mapx)
    r_squared = (r2(Y,Yhat_mapx))
    r_squared_list.append(r_squared)
plt.plot(r_squared_list)


# We can see a large divergence right around 10**16 where our r-squared value plummets, so lets
# plot the difference between our standard linear regression and our linear regression with l2 regularization

# In[ ]:


l2_16 = 10**16
w_map_16 = np.linalg.solve(l2_16*np.eye(2) + X.T.dot(X),X.T.dot(Y))
Yhat_map_16 = X.dot(w_map_16)

l2_17 = 10**17
w_map_17= np.linalg.solve(l2_17*np.eye(2) + X.T.dot(X),X.T.dot(Y))
Yhat_map_17 = X.dot(w_map_17)


# Here what plot the divergence so you can see the difference. In this case, for suspected outliers, 
# I would adjust my "gamma" value to 10^16 to minimize the effects of outliers on the regression. 10^17 is plotted as well to show the same effects we have shown in the chart above. 

# In[ ]:


plt.scatter(X[:,1],Y,marker=".")
plt.scatter(X[:,1],Yhat_ml,label='l2=0',marker="1")
plt.scatter(X[:,1],Yhat_map_16,label='l2=10**16',marker="2",color='purple')
plt.scatter(X[:,1],Yhat_map_17,label='l2=10**17',marker='3',color = 'red')
plt.legend()
plt.xlabel('Total Expenditure')
plt.ylabel('Capital Outlay Expenditure')
plt.title('Regression Analysis with L2 Regularization')


# In[ ]:


print('The linear equation for gamma = 10**16: y =',round(w_map_16[1],2),'x',round(w_map_16[0]),2)
print('The linear equation for the standard linear regression: y =',round(w_ml[1],2),'x',round(w_ml[0],2))


# In conclusion I highly recommend "The Lazy Programmers" course on Linear Regression at udemy. It far exceeded my expectations and that's where I learned this particular technique from. You'll need a decent understanding of linear algebra coming in though. 

# In[ ]:




