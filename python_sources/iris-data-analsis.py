#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

data=pd.read_csv('../input/Iris.csv')


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data[['SepalLengthCm','SepalWidthCm']].describe()


# In[ ]:


data['Species'].unique()


# In[ ]:


len(data.iloc[:,5:])


# In[ ]:


plt.scatter(data['SepalLengthCm'],data['SepalWidthCm'],color='red')
plt.scatter(data['PetalLengthCm'],data['PetalWidthCm'],color='blue')
plt.show()


# In[ ]:


petalLXpetalW=data.iloc[:,3:4].values*data.iloc[:,4:5].values
petalLXpetalW
data['PetalLXPetalW']=petalLXpetalW
sepalLXsepalW=data['SepalLengthCm'].values*data['SepalWidthCm'].values
data['SepalLXSepalW']=sepalLXsepalW


# In[ ]:


data['SepalLengthCm']=np.log(data['SepalLengthCm'].values)
data['SepalWidthCm']=np.log(data['SepalWidthCm'].values)
data['PetalLengthCm']=np.log(data['PetalLengthCm'].values)
data['PetalWidthCm']=np.log(data['SepalWidthCm'].values)


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(data['SepalLengthCm'],data['SepalWidthCm'],color='r')
plt.scatter(data['PetalLengthCm'],data['PetalWidthCm'],color='b')
plt.show()


# In[ ]:


data1=data.iloc[:,1:5]
data2=data.iloc[:,5:6]
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data2=le.fit_transform(data2)
data3=data.iloc[:,6:]
new_data=pd.concat([data1],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(new_data,data2,test_size=0.33,random_state=0)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)


# In[ ]:


y_pred=knn.predict(x_test)


# In[ ]:


knn.score(x_test,y_test)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
new_data=sc.fit_transform(new_data)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(new_data,data2,test_size=0.33,random_state=0)


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)


# In[ ]:


y_pred=knn.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rnd=RandomForestClassifier()
rnd.fit(x_train,y_train)


# In[ ]:


y_pred=rnd.predict(x_test)


# In[ ]:


rnd.score(x_test,y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
cm


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)


# In[ ]:


dtc.score(x_test,y_test)


# In[ ]:


cm=confusion_matrix(y_pred,y_test)
cm

