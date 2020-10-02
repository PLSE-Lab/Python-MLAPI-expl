#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from autofeat import AutoFeatRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


# In[ ]:


df = pd.read_csv("../input/Concrete_Data_Yeh.csv")


# In[ ]:


df.head()


# In[ ]:


X = df[['cement','slag','flyash','water','superplasticizer','coarseaggregate','fineaggregate','age']]
y = df[['csMPa']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Now let's see what results we get with Autofeat ! 

# In[ ]:


model = AutoFeatRegression()
model


# In[ ]:


X_train_feature_creation = model.fit_transform(X_train.to_numpy(), y_train.to_numpy().flatten())


# In[ ]:


X_test_feature_creation = model.transform(X_test.to_numpy())


# In[ ]:


X_test_feature_creation.head()


# We can check our new dataframe ! 

# In[ ]:


X_train_feature_creation.head()


# In[ ]:


X_train_feature_creation.shape[1] - X_train.shape[1]


# **27 features** have been added by autofeat
# 
# We will train 2 logistic regression : 
#     - One with the initial X_train
#     - One with X_train_feature_creation
#     
# 

# In[ ]:


model_1 = LinearRegression().fit(X_train,y_train.to_numpy().flatten())
model_2 = LinearRegression().fit(X_train_feature_creation, y_train.to_numpy().flatten())


# ### Conclusions : 

# In[ ]:


explained_variance_score(y_test, model_1.predict(X_test)), explained_variance_score(y_test, model_2.predict(X_test_feature_creation))


# A big improvement ! A good question would be to know if we could find this kind of improvements with more complex models.
