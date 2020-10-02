#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
dataset.head()


# In[ ]:


#Omitting the first column
updated_dataset=dataset.iloc[:,1:9]
updated_dataset.head()


# In[ ]:


print(updated_dataset.shape)
updated_dataset.describe()


# In[ ]:


#Checking for NA values
updated_dataset.isna().sum()


# In[ ]:


updated_dataset.corr(method="pearson")


# As We can See Chance of Admit is highly Correlated with GRE Score,Toefl Score and CGPA
# 

# In[ ]:


plt.subplots(figsize=(20,4))
sns.barplot(x="GRE Score",y="Chance of Admit ",data=dataset)
plt.subplots(figsize=(25,5))
sns.barplot(x="TOEFL Score",y="Chance of Admit ",data=dataset)
plt.subplots(figsize=(20,4))
sns.barplot(x="University Rating",y="Chance of Admit ",data=dataset)
plt.subplots(figsize=(15,5))
sns.barplot(x="SOP",y="Chance of Admit ",data=dataset)
#plt.subplots(figsize=(15,4))
#sns.barplot(x="CGPA",y="Chance of Admit ",data=dataset)
plt.subplots(figsize=(15,5))
sns.barplot(x="Research",y="Chance of Admit ",data=dataset)


# In[ ]:


X=updated_dataset.iloc[:,:7]
y=updated_dataset["Chance of Admit "]


# In[ ]:


print(X.shape)
print(y.shape)
X.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)


# In[ ]:


from sklearn.linear_model import LinearRegression
#Linear Regression
Linear=LinearRegression()
Linear.fit(X_train,y_train)
y_pred=Linear.predict(X_test)
y_pred


# In[ ]:


from sklearn.metrics import mean_absolute_error,r2_score
print("R2 score of the model is ",r2_score(y_pred,y_test))
print("mean_absolute_error  of the model is ",mean_absolute_error(y_pred,y_test))


# Here are My findings.We can also remove Research Paper variable as Chance of Admit doesnt depend upon that variable much.We can also check for multicollinearity and Can find the p-value to see which variables to remove and can make our model more efficient.

# In[ ]:




