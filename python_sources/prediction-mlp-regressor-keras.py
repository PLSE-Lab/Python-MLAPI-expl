#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import csv 
from pandas import DataFrame as df
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential 
from keras.layers import Dense , Dropout
from sklearn.cross_validation import train_test_split 
from sklearn.linear_model import LinearRegression


# In[ ]:


data_1 = pd.read_csv('train.csv')


# In[ ]:


data_1.shape


# In[ ]:


data_1.head(5)


# In[ ]:


data_2 = pd.read_csv('test.csv')


# In[ ]:


data_2.shape


# In[ ]:


data_2.head(5)


# In[ ]:


output_target = data_1.target


# In[ ]:


output_target.shape


# In[ ]:


data_1 = data_1.drop('target',axis=1)


# In[ ]:


data_1.shape


# In[ ]:


#frame =[data_1,data_2]
#data = pd.concat(frame)


# In[ ]:


#data.shape


# In[ ]:


le = LabelEncoder()
ID = le.fit_transform(data_1.ID)


# In[ ]:


data_1 = data_1.drop('ID' , axis = 1)
ID = pd.DataFrame(ID, columns=['ID'])


# In[ ]:


data_1 = ID.join(data_1)


# In[ ]:


data_1.shape


# In[ ]:


#data_1 = data_1.fillna(0)


# In[ ]:


scale = MinMaxScaler()
data_1 = scale.fit_transform(data_1)


# In[ ]:


data_1 = pd.DataFrame(data_1)


# In[ ]:


data_1.shape


# In[ ]:


output_target = output_target.reshape(-1,1)
output_target = scale.fit_transform(output_target)


# In[ ]:


output_target = pd.DataFrame(output_target)


# In[ ]:


x_train , x_test , y_train ,y_test = train_test_split(data_1 , output_target , test_size=0.3,random_state=0)


# In[ ]:


regressor = Sequential()
regressor.add(Dense(output_dim=1000,init='uniform',activation='relu',input_dim=4992))
regressor.add(Dropout(0.2))
regressor.add(Dense(output_dim=500,init='uniform',activation='relu',input_dim=1000))
regressor.add(Dropout(0.2))
regressor.add(Dense(output_dim=100,init='uniform',activation='relu',input_dim=500))
regressor.add(Dropout(0.2))
regressor.add(Dense(output_dim=10,init='uniform',activation='relu',input_dim=100))
regressor.add(Dropout(0.2))
regressor.add(Dense(output_dim=1,init='uniform',activation='linear'))
regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])

regressor.summary()


# In[ ]:


regressor.fit(x_train , y_train ,epochs=200,batch_size=64)


# In[ ]:


ID_2 = data_2.ID
ID_2 = le.fit_transform(ID_2)
data_2 = data_2.drop('ID',axis=1)
ID_2 = pd.DataFrame(ID_2,columns=['ID'])
data_2 = ID_2.join(data_2)


# In[ ]:


values = regressor.predict(data_2)
#score_validation = regressor.score(y_test , x_test)
#print(score_validation)
values = abs(values)


# In[ ]:


final_df = le.inverse_transform(data_2.ID)


# In[ ]:


final_df = pd.DataFrame(final_df , columns=['ID'])


# In[ ]:


final_df_2 = pd.DataFrame(values , columns=['Target'])


# In[ ]:


final_test_target = final_df.join(final_df_2)


# In[ ]:


final_test_target.shape


# In[ ]:


final_test_target.head(20)


# In[ ]:


final_test_target.to_csv('final_target.csv' , index=False)


# In[ ]:




