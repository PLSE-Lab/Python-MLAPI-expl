#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


data.head()


# In[ ]:


X=data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]


# In[ ]:


y=data['Outcome']


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


X[X.isnull()].count()


# In[ ]:


y[y.isnull()].count()


# In[ ]:


df=data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]


# In[ ]:


l=[]
for i in y:
    if i==0:
        l.append('Non-Diabetic')
    else:
        l.append('Diabetic')


# In[ ]:


df['Outcome']=l


# In[ ]:


df.head()


# In[ ]:


X.plot(kind="box",figsize=(16,6),subplots=True)
plt.show()


# In[ ]:


sns.distplot(df.Age,label='Outcome')
plt.show()


# In[ ]:


X.BMI.describe()


# In[ ]:


X.BMI.quantile(0.95)


# In[ ]:


X.BloodPressure.describe()


# In[ ]:


X.BloodPressure.quantile(0.95)


# In[ ]:


X.Insulin.describe() #Outlier---------------


# In[ ]:


X[X.Insulin.values>200]


# In[ ]:


fig,ax=plt.subplots(figsize=(10,7))
sns.heatmap(data.corr(),annot=True,cmap="Reds",ax=ax,square=True)


# In[ ]:


sns.pairplot(df,hue='Outcome')

-----------------------------------------Scaling The Values----------------------------------------------
# In[ ]:


X.columns


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


Scaler=MinMaxScaler()


# In[ ]:


Scaler.fit(X.BloodPressure.values.reshape(-1,1))


# In[ ]:


Scaler.data_max_,Scaler.data_min_


# In[ ]:


X['BloodPressure']=Scaler.fit_transform(X.BloodPressure.values.reshape(-1,1))


# In[ ]:


Scaler.fit(X.Insulin.values.reshape(-1,1))


# In[ ]:


X['Insulin']=Scaler.fit_transform(X.Insulin.values.reshape(-1,1))


# In[ ]:


Scaler.fit(X.DiabetesPedigreeFunction.values.reshape(-1,1))


# In[ ]:


X['DiabetesPedigreeFunction']=Scaler.fit_transform(X.DiabetesPedigreeFunction.values.reshape(-1,1))


# In[ ]:


Scaler.fit(X.BMI.values.reshape(-1,1))


# In[ ]:


X['BMI']=Scaler.fit_transform(X.BMI.values.reshape(-1,1))


# In[ ]:


Scaler.fit(X.Glucose.values.reshape(-1,1))


# In[ ]:


X['Glucose']=Scaler.fit_transform(X.Glucose.values.reshape(-1,1))


# In[ ]:


X.head()


# In[ ]:


df_x=X[['Glucose','BloodPressure','Insulin','BMI']]


# In[ ]:


df_x.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(df_x,y,train_size=0.75,random_state=100)


# In[ ]:


y_train.shape,x_train.shape,x_test.shape


# In[ ]:


x_train.reset_index(drop=True,inplace=True)
x_test.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)


# In[ ]:


from sklearn.neighbors import  KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=3)


# In[ ]:


knn.fit(x_train,y_train)


# In[ ]:


y_pred=knn.predict(x_test)


# In[ ]:


knn.score(x_test,y_test)


# In[ ]:




