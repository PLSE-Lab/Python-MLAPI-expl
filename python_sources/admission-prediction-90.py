#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df=pd.read_csv('Admission_Predict_Ver1.1.csv')


# In[ ]:


df.head()


# In[ ]:


X=df.drop(['Chance of Admit ','Serial No.'],axis=1)


# In[ ]:


y=df['Chance of Admit ']


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X.shape


# In[ ]:





# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()


# In[ ]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


model_random=RandomForestRegressor(n_estimators=200)


# In[ ]:


model_random.fit(X_train,y_train)


# In[ ]:


prd_random=model_random.predict(X_test)


# In[ ]:


prd=pd.DataFrame(prd_random)


# In[ ]:


ytest=pd.DataFrame(y_test)


# In[ ]:


ytest=ytest.reset_index()
compare=pd.concat([ytest,prd],axis=1)


# In[ ]:


compare


# In[ ]:


y_test.shape


# In[ ]:


prd_random.shape


# In[ ]:


prd


# In[ ]:


import seaborn as sns


# In[ ]:


ytest=ytest.reset_index()


# In[ ]:


compare=pd.concat([ytest,prd],axis=1)


# In[ ]:


sns.lineplot(x='Chance of Admit ',y=0,data=compare)


# In[ ]:


compare.columns


# In[ ]:


import tensorflow as tf


# In[ ]:


from tensorflow.keras.models import Sequential


# In[ ]:


from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


model=Sequential()
model.add(Dense(7,activation='relu'))
Dropout(.2)
model.add(Dense(7,activation='relu'))
Dropout(.2)
model.add(Dense(7,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(7,activation='relu'))
Dropout(.2)
model.add(Dense(1,activation='relu'))
model.compile(optimizer='adam',loss='mse')


# In[ ]:


early_stop=EarlyStopping(patience=20,verbose=1)


# In[ ]:


model.fit(X_train,y_train.values,epochs=500,validation_data=(X_test,y_test.values))


# In[ ]:


ytrain=pd.DataFrame(y_train)


# In[ ]:


ytrain=ytrain.reset_index()


# In[ ]:


prd_deep=model.predict(X_test)


# In[ ]:


prd_deep=pd.DataFrame(prd_deep)


# In[ ]:


compare=pd.concat([ytest,prd_deep],axis=1)


# In[ ]:


sns.lineplot(x='Chance of Admit ',y=0,data=compare)


# In[ ]:


loses=pd.DataFrame(model.history.history)


# In[ ]:


loses.plot()


# In[ ]:


prd_deep_train=model.predict(X_train)


# In[ ]:


compare=pd.concat([ytrain,prd_deep_train],axis=1)


# In[ ]:




