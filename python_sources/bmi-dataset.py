#!/usr/bin/env python
# coding: utf-8

# Importing libraries
# 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Importing datasets

# In[17]:


df=pd.read_csv('../input/500_Person_Gender_Height_Weight_Index.csv')
print(df.sample(frac=0.1)) # this will print only 10% of total data ie: 10%  of df


# Changing Gender columns to binary values (ie: Dichotomous variable -: 0/1)

# In[18]:


df=pd.get_dummies(df)
print(df)


# In[6]:


X=df.iloc[:,[0,1,3,4]].values
Y=df.iloc[:,2].values


# In[20]:


X_nu=df[["Height","Weight","Index"]]

X_nu.corr()

X_nu.hist(bins=50)


# In[24]:


plt.scatter(X_nu.Index,Y,color="g")
plt.grid()


# In[25]:


plt.scatter(X_nu.Weight,Y,color="r")
plt.grid()


# In[26]:


plt.scatter(X_nu.Height,Y,color="teal")
plt.grid()


# Splitting our dataset to train and test sets 

# In[7]:


X_train=X[:400]
X_test=X[400:]

Y_train=Y[:400]
Y_test=Y[400:]


# **Fitting to our model**

# In[8]:




from sklearn.linear_model import LinearRegression
teacher=LinearRegression()
learner=teacher.fit(X_train,Y_train)


# **# Making  Prediction**

# In[9]:



Yp=learner.predict(X_test)
c=learner.intercept_
m=learner.coef_
print("c is {}  \n m is {}  \n Yp is {}".format(c,m,Yp))


# List conversion due to data type

# In[ ]:





# In[10]:



xlist=list(X_train)
ylist=list(Y_train)
yplist=list(Yp)


# In[11]:



mytable=pd.DataFrame({"input":xlist,"out":ylist})
print(mytable)


# In[12]:


from sklearn.metrics import mean_squared_error,accuracy_score
Error=mean_squared_error(Yp,Y_test)
np.sqrt(Error)


# In[13]:



import seaborn as sns
sns.barplot(x=Y_test,y=Yp,data=df)


# In[14]:



y_pred_cls=np.zeros_like(Yp)
y_pred_cls[Yp>2.5]=1

y_test_cls=np.zeros_like(Yp)
y_test_cls[Y_test>2.5]=1


# In[15]:


print(accuracy_score(y_test_cls,y_pred_cls))

