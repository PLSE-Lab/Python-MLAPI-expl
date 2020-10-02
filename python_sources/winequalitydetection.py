#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the 
input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


import pandas as pd
#top50 = pd.read_csv("../input/top50.csv")
dataset = pd.read_csv("../input/winequality.csv")


# In[ ]:


dataset.shape


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset.describe()


# In[ ]:



plt.figure(figsize=(25,12))
sns.barplot(x=dataset['alcohol'],y=dataset['sulphates'])
plt.xlabel('Alcohol')
plt.ylabel('sulphates')


# In[ ]:



alcohol_mean=dataset['alcohol'].mean()
print(alcohol_mean)
alcohol_median=dataset['alcohol'].median()
print(alcohol_median)


# In[ ]:



X=dataset.drop('quality',axis=1)
Y=dataset['quality']


# In[ ]:



sns.lineplot(data=dataset['total sulfur dioxide'],label='Sulplhur di-oxide')
sns.lineplot(data=dataset['free sulfur dioxide'],label='free sulfur dioxide')


# In[ ]:


sns.swarmplot(x=dataset['fixed acidity'],y=dataset['volatile acidity'])


# In[ ]:


sns.distplot(dataset['density'],kde=True,bins=10)


# In[ ]:


sns.distplot(dataset['pH'],kde=True,color='green',bins=10)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)


# In[ ]:


model=LinearRegression()
model.fit(x_train,y_train)


# In[ ]:


y_predict=model.predict(x_test)


# In[ ]:



df = pd.DataFrame({'  Actual Quality ': y_test, '   Predicted Quality': y_predict})
df1 = df.head(25)
print(df1)


# In[ ]:



df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:



r2_score(y_test,y_predict)


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))


# In[ ]:



from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


model1 = LogisticRegression(solver='lbfgs', multi_class='auto')
model1 = LogisticRegression()
model1.fit(x_train, y_train)


# In[ ]:


logistic_regression_pred=model1.predict(x_test)


# In[ ]:


df = pd.DataFrame({'  Actual Quality ': y_test, '   Predicted Quality': logistic_regression_pred})
df1 = df.head(25)
print(df1)


# In[ ]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:




