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


# importing data file

# In[ ]:


data=pd.read_csv('../input/winequalityred/winequality-red.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# Relation between variables or correlation in visual

# In[ ]:


import seaborn as sm
sm.heatmap(data.corr(),center=0)


# machine learning by keeping quality as points(1 to 10)

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor


# In[ ]:


x=data.drop("quality",axis=1)
y=data.quality


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25,random_state=3)
x_train.shape,y_train.shape


# In[ ]:


reg= LinearRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test,y_predict.round(),normalize=True)


# In[ ]:


print(acc)


# In[ ]:


model=SGDRegressor()
model.fit(x_train,y_train)
predict=model.predict(x_test)


# In[ ]:


acc=accuracy_score(y_test,y_predict.round())
acc


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)


# In[ ]:


Y_predict=model.predict(x_test)


# In[ ]:


acc=accuracy_score(y_test,y_predict.round())
acc


# By keeping quality in points the accuracy is too low.

# In[ ]:


sm.barplot(x=data['quality'],y=data['fixed acidity'])


# In[ ]:


sm.countplot(data['quality'])


# **So change the quality in two groups '1' (this means good quality. 3,4,5 rated wines are in this group) '0' (this means low quality wine.6,7,8 rated wines are in this group) **

# In[ ]:


data[['quality']]=data['quality'].apply(lambda x: 0 if int(x)<6 else 1)


# In[ ]:


data.quality.value_counts()


# In[ ]:


y=data['quality']
x=data.drop("quality",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=20,random_state=2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model1=LogisticRegression(max_iter=10000)
model1.fit(x_train,y_train)
model1=model1.score(x_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier()
model2.fit(x_train,y_train)
model2=model2.score(x_test,y_test)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model3.fit(x_train,y_train)
model3=model3.score(x_test,y_test)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
model4 = GaussianNB()
model4.fit(x_train,y_train)
model4=model4.score(x_test,y_test)


# In[ ]:


from  sklearn.neighbors import KNeighborsClassifier
model5=KNeighborsClassifier()
model5.fit(x_train,y_train)
model5=model5.score(x_test,y_test)


# In[ ]:


final_results=pd.DataFrame({'models':['LogisticRegression','RandomForset','Dessiontree','GaussianNB','KNeighborsClassifier'],'accuracy_score':[model1,model2,model3,model4,model5]})


# In[ ]:


sm.barplot(x=final_results['models'],y=final_results['accuracy_score'])


# CONCLUSION

# RANDOM FOREST has the highest accurancy with 85%
