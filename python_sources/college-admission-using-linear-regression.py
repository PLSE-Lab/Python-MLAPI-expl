#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/Admission_Predict_Ver1.1.csv',sep=r'\s*,\s*')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:





# In[ ]:


df.describe()


# In[ ]:


sns.scatterplot(x='Chance of Admit',y='GRE Score',data=df)


# In[ ]:


sns.scatterplot(x='Chance of Admit',y='TOEFL Score',data=df)


# In[ ]:


plt.hist(df['GRE Score'])
plt.xlabel('GRE SCORE')


# In[ ]:


plt.hist(df['TOEFL Score'])
plt.xlabel('TOEFL SCORE')


# In[ ]:


plt.scatter(df['University Rating'],df['GRE Score'],)


# In[ ]:


sns.countplot(df['Research'])


# In[ ]:


sns.scatterplot(y='Chance of Admit',x='Research',data=df)


# In[ ]:


sns.jointplot(x='CGPA',y='Chance of Admit',data=df)


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.jointplot(x='GRE Score',y='TOEFL Score',data=df)


# In[ ]:


df.drop('Serial No.',axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


x=df.drop('Chance of Admit',axis=1)
y=df['Chance of Admit']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lr=LinearRegression()


# In[ ]:


lr.fit(x_train,y_train)


# In[ ]:


predictions=lr.predict(x_test)


# In[ ]:


print("intercept=",lr.intercept_)
print("coefficient=",lr.coef_)


# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:




