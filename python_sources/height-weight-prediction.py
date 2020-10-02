#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv('../input/weight-height/weight-height.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


sns.scatterplot('Height','Weight',data=df,hue='Gender')


# In[ ]:


sns.countplot('Gender',data=df)


# We have balanced dataset

# In[ ]:


sns.jointplot(x='Height',y='Weight',data=df,kind='reg')


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# Height and Weight are highly correlated.Gender and weight are slightly correlated.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
df['Gender']=lab.fit_transform(df['Gender'])


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)


# In[ ]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(xtrain,ytrain)


# In[ ]:


ypred=reg.predict(xtest)


# In[ ]:


print('Accuracy is {}'.format(reg.score(xtest,ytest)*100))


# In[ ]:


trainypred=reg.predict(xtrain)
testypred=reg.predict(xtest)


# In[ ]:


from sklearn.metrics import r2_score
print('Training set score {}'.format(r2_score(ytrain,trainypred)))
print('Testing set score {}'.format(r2_score(ytest,testypred)))


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D 


# In[ ]:


ytest=np.array(ytest)


# In[ ]:


xtest.shape


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xtrain['Gender'],xtrain['Height'],ytrain,c='red', marker='o', alpha=0.5)
ax.plot_surface(xtest['Gender'],xtest['Height'],ytest, color='blue', alpha=0.3)
ax.set_xlabel('Price')
ax.set_ylabel('AdSpends')
ax.set_zlabel('Sales')
plt.show()


# In[ ]:




