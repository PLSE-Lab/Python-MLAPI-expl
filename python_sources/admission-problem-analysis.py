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


df=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()


# In[ ]:


df.corr(method='pearson')


# In[ ]:


df.columns = df.columns.str.replace('\s+', '_') # in case there are multiple white spaces


# In[ ]:


df.head()


# In[ ]:


X=df.drop(['Serial_No.','Chance_of_Admit_'],axis=1)


# In[ ]:


y=df['Chance_of_Admit_']


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scalerX=MinMaxScaler(feature_range=(0,1))
X_train[X_train.columns]=scalerX.fit_transform(X_train[X_train.columns])
X_test[X_test.columns]=scalerX.transform(X_test[X_test.columns])


# In[ ]:


X.head()
X.isnull().sum()


# In[ ]:


X['GRE_Score'].hist(bins=50)


# In[ ]:


X['TOEFL_Score'].hist(bins=50)


# In[ ]:


X.boxplot(column='GRE_Score',by='TOEFL_Score')


# In[ ]:


X['GRE_Score'].plot('density',color='Red')


# In[ ]:


X['TOEFL_Score'].plot('density',color='Black')


# In[ ]:


X['SOP'].hist(bins=50)


# In[ ]:


X.boxplot(column='SOP',by='TOEFL_Score')


# In[ ]:


X['SOP'].plot('density',color='Green')


# In[ ]:


X.boxplot(column='SOP',by='GRE_Score')


# In[ ]:


X['LOR_'].hist(bins=50)


# In[ ]:


X.boxplot(column='LOR_',by='TOEFL_Score')


# In[ ]:


X['LOR_'].plot('density',color='Pink')


# In[ ]:


X.boxplot(column='LOR_',by='GRE_Score')


# In[ ]:


X['CGPA'].hist(bins=50)


# In[ ]:


X.boxplot(column='CGPA',by='GRE_Score')


# In[ ]:


X['CGPA'].plot('density')


# In[ ]:


X['Research'].hist(bins=50)


# In[ ]:


X.boxplot(column='Research',by='CGPA')


# In[ ]:


X['Research'].plot('density',color='Blue')


# In[ ]:


X=pd.get_dummies(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
q=DecisionTreeRegressor()
q.fit(X_train,y_train)
q.score(X_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
r=RandomForestRegressor()
r.fit(X_train,y_train)
r.score(X_test,y_test)


# In[ ]:


from sklearn.linear_model import LinearRegression
o=LinearRegression()
o.fit(X_train,y_train)
o.score(X_test,y_test)


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
e=ExtraTreesRegressor()
e.fit(X_train,y_train)
e.score(X_test,y_test)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
n=KNeighborsRegressor()
n.fit(X_train,y_train)
n.score(X_test,y_test)


# In[ ]:


# save model
import pickle
file_name='Admission.sav'
tuples=(o,X)
pickle.dump(tuples,open(file_name,'wb'))


# In[ ]:


print(o.coef_)


# In[ ]:


print(o.intercept_)


# In[ ]:


c=print(o.predict(X_test))
c


# In[ ]:


print(y_test,' ',c)


# In[ ]:




