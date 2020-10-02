#!/usr/bin/env python
# coding: utf-8

# The database shows problem to load here but i have tried it in my computer and gives an accuracy_score of 84% 

# In[ ]:


from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


# In[ ]:


dataset=pd.read_csv("../input/prostate-cancer/datasets_66762_131607_Prostate_Cancer.csv")
dataset.head()


# In[ ]:


dataset=dataset.drop("id",axis=1)
dataset.info()


# In[ ]:


dataset["diagnosis_result"].unique()


# In[ ]:


y=dataset["diagnosis_result"]
x=dataset.drop("diagnosis_result",axis=1)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
y_test.head()


# In[ ]:


reg=linear_model.LogisticRegression(max_iter=1000)
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print(y_predict)


# In[ ]:


accuracy_score(y_predict,y_test)


# In[ ]:


confusion_matrix(y_predict,y_test)


# In[ ]:


dataset["diagnosis_result"].head()
dataset["diagnosis_result"]=pd.get_dummies(dataset["diagnosis_result"])
dataset["diagnosis_result"].head()


# Tried converting the diagonosis result to binary and then running the model and this resulted in decrease accuracy score
# 
