#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# ### * A supervised learning method for classification
#  
# ### * Predicts the probability of an outcome that can have two values

# signmoid function:
# g= 1 / ( 1 + e^(-v))
# 
# Linear output -> Sigmoid function -> 1 / ( 1 + e^(-output))
# 
# range: 0 to 1

# ## 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


# In[ ]:


df=pd.read_csv('/kaggle/input/loanapproval/loan.csv')


# In[ ]:


df.keys()  #shows the column names


# In[ ]:


df #will display the whole dataset


# In[ ]:


df=df.dropna()  #to drop all null values


# In[ ]:


df


# In[ ]:


x=df[["FICO.Score","Loan.Amount"]].values
y=df["Interest.Rate"].values


# In[ ]:


model=LinearRegression()
model.fit(x,y)


# In[ ]:


y_pred=model.predict(x)


# In[ ]:


exp=np.exp(-y_pred)+1
log=1/exp


# In[ ]:


y_con = y<17    #store the condition where interest rate is less than 17


# In[ ]:


df["TF"]=df["Interest.Rate"] <17   #store boolean values in TF where interest rate is less than 17
df


# In[ ]:


df.TF.value_counts()     # no of types of values, 


# In[ ]:


log_reg=LogisticRegression()    #to use logistic regression


# In[ ]:


log_reg.fit(x,y_con)   #y_con is where interest rate i.e. y is < 17


# In[ ]:


log_reg.predict([[10,20000]])


# In[ ]:


dat=log_reg.predict_proba([[1000,20000]])   #this prints the probability :(false,true)
dat


# In[ ]:


dat[0][1]>0.8


# In[ ]:


y_pred = log_reg.predict(x)


# In[ ]:


df["Predict"] = y_pred
df.Predict


# In[ ]:


df.Predict.value_counts()  #predicted data


# In[ ]:


#real data

df.TF.value_counts()


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


cm = confusion_matrix(df.TF.values, df.Predict.values)


# In[ ]:


cm


# In[ ]:


df.shape


# In[ ]:


accuracy_score(df.TF.values, df.Predict.values)

