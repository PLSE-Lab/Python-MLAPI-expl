#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


iris=pd.read_csv('../input/iris/Iris.csv')


# In[ ]:


df=iris[['SepalLengthCm','PetalLengthCm','Species']]


# In[ ]:


df.rename(columns={'SepalLengthCm':'SL','PetalLengthCm':'PL'},inplace=True)


# In[ ]:


df['Species'].replace('Iris-virginica','2',inplace=True)
df['Species'].replace('Iris-setosa','0',inplace=True)
df['Species'].replace('Iris-versicolor','1',inplace=True)


# In[ ]:


df


# In[ ]:


X=df.iloc[:,0:2].values
y=df.iloc[:,-1].values


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[ ]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[ ]:


a=np.arange(start=X_train[:,0].min()-1, stop=X_train[:,0].max()+1, step=0.01)
b=np.arange(start=X_train[:,1].min()-1, stop=X_train[:,1].max()+1, step=0.01)


# In[ ]:


print(a.shape)
print(b.shape)


# # creating a meshgrid

# In[ ]:


XX,YY=np.meshgrid(a,b)


# In[ ]:


print(XX.shape)
print(YY.shape)


# In[ ]:


525*643


# ### Classifying every point on the meshgrid

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()


# In[ ]:


classifier.fit(X_train,y_train)


# In[ ]:


input_array=np.array([XX.ravel(),YY.ravel()]).T

label=classifier.predict(input_array)


# In[ ]:


label


# In[ ]:


label.shape


# # Plotting the array as an image

# In[ ]:


plt.contourf(XX,YY,label.reshape(XX.shape))


# In[ ]:





# # Plotting all the training data on the plot

# In[ ]:


y_train


# In[ ]:


my_dict = { '0' : 'violet' , '1' : 'green', '2' : 'yellow'}
y_train = [my_dict[zi] for zi in y_train]


# # Decision boundary

# In[ ]:


plt.figure(figsize=(10, 6), dpi=80)
plt.contourf(XX,YY,label.reshape(XX.shape), alpha=0.50)
plt.scatter(x=X_train[:,0],y=X_train[:,1], c=y_train, marker='*')
plt.xlabel("Sepal Length (Cm)")
plt.ylabel("Petal Length (Cm)")
plt.title('Decision Boundary')
plt.show()


# In[ ]:





# In[ ]:




