#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv("../input/creditcardfraud/creditcard.csv")


# In[ ]:


data.head()


# In[ ]:


X=data.drop("Class",axis=1)


# In[ ]:


Y=data["Class"]


# In[ ]:


X.head()


# In[ ]:


## Get the Fraud and the normal dataset 

fraud = data[data['Class']==1]

normal = data[data['Class']==0]


# In[ ]:


print(fraud.shape,normal.shape)


# ### UnderSampling

# In[ ]:


from imblearn.under_sampling import NearMiss


# In[ ]:


nm = NearMiss(sampling_strategy='auto')


# In[ ]:


X_res,Y_res=nm.fit_sample(X,Y)


# In[ ]:


X_res.shape,Y_res.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


LR_model=LogisticRegression()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X_res,Y_res,random_state=1,test_size=0.2)


# In[ ]:


LR_model.fit(X_train,y_train)


# In[ ]:


predict=LR_model.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score,recall_score


# In[ ]:


accuracy=accuracy_score(y_test,predict)


# In[ ]:


print(accuracy)


# In[ ]:


print(recall_score(y_test,predict))


# ### OverSampling

# In[ ]:


print(fraud.shape,normal.shape)


# In[ ]:


from imblearn.combine import SMOTETomek


# In[ ]:


smk=SMOTETomek(random_state=42)


# In[ ]:


X_up,Y_up=smk.fit_sample(X,Y)


# In[ ]:


X_up.shape,Y_up.shape


# In[ ]:


Y_up.value_counts()


# In[ ]:


X1_train,X1_test,Y1_train,Y1_test=train_test_split(X_up,Y_up,random_state=42,test_size=0.2)


# In[ ]:


X1_train.shape


# In[ ]:


LR_model


# In[ ]:


LR_model.fit(X1_train,Y1_train)


# In[ ]:


predict1=LR_model.predict(X1_test)


# In[ ]:


print(accuracy_score(predict1,Y1_test))


# In[ ]:


print(recall_score(predict1,Y1_test))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier()


# In[ ]:


X2_train=X1_train.sample(frac=0.3,random_state=42)
X2_test=X1_test.sample(frac=0.3,random_state=42)
Y2_train=Y1_train.sample(frac=0.3,random_state=42)
Y2_test=Y1_test.sample(frac=0.3,random_state=42)


# In[ ]:


X2_train.shape


# In[ ]:


rf.fit(X2_train,Y2_train)


# In[ ]:


predict2=rf.predict(X2_test)


# In[ ]:


print(accuracy_score(Y2_test,predict2))


# In[ ]:


print(recall_score(Y2_test,predict2))


# ### The Random Forest model is showing high accuracy and recall results.

# In[ ]:




