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


import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sns


# In[ ]:


data = pd.read_csv("/kaggle/input/social-network-ads/Social_Network_Ads.csv")


# In[ ]:


data.head()


# In[ ]:


# Here User ID is not suitable to predict the results,so we are ignore this coloumn.

data = data[['Gender','Age','EstimatedSalary','Purchased']]


# In[ ]:


print(data.head())


# In[ ]:


data.isnull().sum()


# In[ ]:


data.plot()


# In[ ]:


import seaborn as sns
sns.set(style="ticks")

sns.pairplot(data, hue="Gender")


# In[ ]:


# Here We Check The Total No. Who Purchased or Not Purchased
sns.countplot(x="Purchased",data=data)


# In[ ]:


# As We See here mostly female's like to buy product then male's
sns.countplot(x="Purchased",hue="Gender",data=data)


# In[ ]:


# Now Lets Convert The Variables into dummy variabels for our ML model.
# If the Value of 1 in Male Then i.e male if value is 1 in Female then i.e Female 
pd.get_dummies(data['Gender'])


# In[ ]:


sex = pd.get_dummies(data['Gender'],drop_first=True)
sex.head()


# In[ ]:


data_p = pd.concat([data,sex],axis=1)


# In[ ]:


data_p.head()


# In[ ]:


# Now There is a Gender Column which we do neet further because we have converted into dummies and concat the male column in data set
data_p = data_p.drop(['Gender'],axis=1)


# In[ ]:


data_p.head()


# # Now Lets Split The Data

# In[ ]:


X = data_p[['Age','EstimatedSalary','Male']].values
y = data_p['Purchased'].values


# In[ ]:


# Now Train our Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)


# In[ ]:


# Preprocessing
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


log = LogisticRegression()
log.fit(X_train,y_train)
predict = log.predict(X_test)


# # Now Lets See The Accuracy Of our model

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predict))


# In[ ]:


print(classification_report(y_test,predict))


# In[ ]:


plt.plot(y_test,predict)
plt.show()


# In[ ]:




