#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


filename="../input/neural-net-regression-data/fake_reg.csv"
df=pd.read_csv(filename)
df


# In[ ]:


sns.pairplot(df)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X=df[['feature1','feature2']].values
y=df['price'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler=MinMaxScaler()


# In[ ]:


scaler.fit(X_train)


# In[ ]:


X_train=scaler.transform(X_train)


# In[ ]:


X_test=scaler.transform(X_test)


# In[ ]:


X_train.max()


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


model=Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')


# In[ ]:


model.fit(x=X_train,y=y_train,epochs=250)


# In[ ]:


loss_df=pd.DataFrame(model.history.history)
loss_df.plot()


# In[ ]:


model.evaluate(x=X_test,y=y_test,verbose=10)


# In[ ]:


model.evaluate(x=X_train,y=y_train,verbose=10)


# In[ ]:


prediction=model.predict(X_test)
prediction


# In[ ]:


test_pred=pd.Series(prediction.reshape(300,))
test_pred


# In[ ]:


pred_df=pd.DataFrame(data=y_test,columns=['True value(y)'])
pred_df


# In[ ]:


pred_df=pd.concat([pred_df,test_pred],axis=1)
pred_df


# In[ ]:


pred_df.columns=['true value(y)','predicted_vals']


# In[ ]:


pred_df


# **Scatterplot shows that the true values and predicted values are extremely close to each other which shows that the deep neural network is working perfectly.**

# In[ ]:


sns.scatterplot(x='true value(y)',y='predicted_vals',data=pred_df)


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[ ]:


ae=mean_absolute_error(pred_df['true value(y)'],pred_df['predicted_vals'])
se=mean_squared_error(pred_df['true value(y)'],pred_df['predicted_vals'])
print('mean absolute error is {}'.format(ae))
print('mean squared error is {}'.format(se))


# In[ ]:


df.describe()


# **load_model is used to save model and to use it in other notebook or script file

# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


model.save('first_neural_net.h5')


# In[ ]:


later_model=load_model('first_neural_net.h5')


# # The End
