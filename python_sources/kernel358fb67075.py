#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


A=data["Category"].unique()
D={}
for i in range(len(A)):
    D[A[i]]=i+1
data["C1"]=data["Category"].map(D).astype(int)


# In[ ]:


data.head()


# In[ ]:


data["Last Updated"]=data["Last Updated"].apply(lambda x:pd.to_datetime(x))
data.head()


# In[ ]:


data.set_index(data["Last Updated"],inplace=True)


# In[ ]:


data=data.drop(["App","Price","Genres","Android Ver"],axis=1)


# In[ ]:


data.head()


# In[ ]:


data=data.drop(["Current Ver","Category"],axis=1)


# In[ ]:


data=data.drop(["Content Rating"],axis=1)


# In[ ]:


B={"Free":1, "Paid":0}
data["Type"]=data["Type"].map(B).astype(int)


# In[ ]:





# In[ ]:


def correct_size(x):
    if "M" in x:
        x=x.strip('M')
        x=float(x)*1000
    elif "K" in x:
        x=x.strip('K')
        x=float(x)*100
    else:
        x=0
    return x
data["S"]=data["Size"].apply(correct_size)
    


# In[ ]:


import re
def correct_install(x):
    x=re.sub("[^0-9]","",x)
    return x
data["INS"]=data["Installs"].apply(correct_install)

    


# In[ ]:


data.head()


# In[ ]:


data["Reviews"]=data["Reviews"].astype(int)


# In[ ]:


data["Rating"]=data["Rating"].astype(float)


# In[ ]:


data=data.drop(["Size","Installs","Last Updated"],axis=1)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data["INS"]=data["INS"].astype(int)


# In[ ]:


data.info()


# In[ ]:


# vizzzzz
C=data.corr()
sns.heatmap(C,annot=True,fmt=".2f")


# In[ ]:


# training
y=data.pop("Rating").values
X=data.values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
print(X)
print(y)

    


# In[ ]:


S=[]
for j in y:
    S.append(j)
S1=[S.count(i) for i in S]

plt.bar(S,S1)


# In[ ]:


from sklearn.model_selection import train_test_split,GridSearchCV
X_train,x_test,Y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(x_test.shape)


# In[ ]:


from sklearn.linear_model import SGDRegressor
reg=SGDRegressor()
reg.fit(X_train,Y_train)


# In[ ]:


pred=reg.predict(x_test)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,pred)


# In[ ]:


mse


# In[ ]:


pred=pred.round(1)


# In[ ]:


plt.scatter([i for i in range(len(y_test))],y_test,color="red")
plt.plot([j for j in range(len(y_test))],pred)


# In[ ]:


y_test


# In[ ]:




