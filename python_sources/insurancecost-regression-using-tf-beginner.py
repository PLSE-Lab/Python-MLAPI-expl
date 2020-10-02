#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/insurance/insurance.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.countplot(df['sex'])


# In[ ]:


sns.countplot(df['smoker'],hue='sex',data=df)


# In[ ]:


sns.distplot(df['age'],bins=50)


# In[ ]:


sns.scatterplot(df['age'],df['bmi'])


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


df['sex']=pd.get_dummies(df['sex'],drop_first=False)
df['smoker']=pd.get_dummies(df['smoker'],drop_first=False)


# In[ ]:


df.drop('region',axis=1,inplace=True)


# In[ ]:


x=df.drop('charges',axis=1).values
y=df['charges'].values


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler=MinMaxScaler()


# In[ ]:


x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam


# In[ ]:


model = Sequential()
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[ ]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=400)


# In[ ]:


losses=pd.DataFrame(model.history.history)


# In[ ]:


losses.plot()


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score


# In[ ]:


predictions=model.predict(x_test)


# In[ ]:


mean_absolute_error(y_test,predictions)


# In[ ]:


np.sqrt(mean_squared_error(y_test,predictions))


# In[ ]:


explained_variance_score(y_test,predictions)


# In[ ]:


plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')


# In[ ]:




