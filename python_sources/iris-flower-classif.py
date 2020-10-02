#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")


# In[ ]:


data


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.sample(5)


# In[ ]:


data.head(10)


# In[ ]:


x=data.describe()
x=x.iloc[1:,:]
x.style.background_gradient(cmap="Wistia")


# In[ ]:


data.isnull().sum().sum()


# In[ ]:


data.dtypes


# In[ ]:


data.info()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sb


# In[ ]:


sb.barplot(data.species,data.sepal_length)


# In[ ]:


sb.lineplot(data.species,data.sepal_length)


# In[ ]:


sb.scatterplot(data.sepal_width,data.sepal_length)


# In[ ]:


data.iloc[:,1:].plot(kind="line")


# In[ ]:


get_ipython().system('pip install dabl')


# In[ ]:


import dabl as db


# In[ ]:


db.plot(data,target_col='species')


# In[ ]:


email
adhar
pan
mobile
health
bmi
age
salary


# In[ ]:


data.profile_report()


# In[ ]:


plt.rcParams["figure.figsize"]=(18,6)
plt.style.use("classic")
sb.stripplot(data.species,data.petal_length)
plt.title("species and petal length comp",fontsize=20) 
plt.grid()
plt.show()


# In[ ]:


plt.style.available


# In[ ]:


sb.distplot(data.sepal_length)


# In[ ]:


sb.countplot(data.species)


# In[ ]:


data["species"].value_counts().plot(kind='pie',colors=['pink','black','blue'])
plt.axis("off")


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


features=data.iloc[:,:4]
print(features)


# In[ ]:


target=data.iloc[:,4]
print(target)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=0)
print(features.shape)
print(target.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


x_test.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression 
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
cr=classification_report(y_test,y_pred)
print(cr)


# In[ ]:


cm=confusion_matrix(y_test,y_pred)
sb.heatmap(cm,annot=True)


# In[ ]:


print(y_pred)


# In[ ]:




