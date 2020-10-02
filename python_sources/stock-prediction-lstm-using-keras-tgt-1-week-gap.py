#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# This kernel is a fork from [https://www.kaggle.com/amarpreetsingh/stock-prediction-lstm-using-keras](https://www.kaggle.com/amarpreetsingh/stock-prediction-lstm-using-keras)  
# The target value used to train is gapped 7 days away from the 7 days window input used for training.  
# This is to see how well is the LSTM being able to predict values other than that immediately following the window period

# In[ ]:


data = pd.read_csv('../input/all_stocks_5yr.csv')
cl = data[data['Name']=='MMM'].close


# In[ ]:


cv = cl.values
scl = MinMaxScaler()
#Scale the data
#cl = cl.reshape(cl.shape[0],1)
cv = cv.reshape(cv.shape[0],1)
cv = scl.fit_transform(cv)
cv


# In[ ]:


#Create a function to process the data into 7 day look back slices
#codes is edited here to include the argument gap, i.e. we leave a gap from 7 days window presented to the lstm
#for the target value used for training and prediction
def processData(data,lb,gap):
    X,Y = [],[]
    for i in range(len(data)-lb-gap-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb+gap),0])
    return np.array(X),np.array(Y)
X,y = processData(cv,7,7)
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
print(X_train.shape[0])
print(X_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])


# In[ ]:


#Build the model
model = Sequential()
model.add(LSTM(256,input_shape=(7,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#Fit model with history to check for overfitting
history = model.fit(X_train,y_train,epochs=300,validation_data=(X_test,y_test),shuffle=False)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


# In[ ]:


X_test[0]


# In[ ]:


Xt = model.predict(X_test)
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label='actual')
plt.plot(scl.inverse_transform(Xt), label='predict')
plt.legend()
plt.show()


# In[ ]:


act = []
pred = []
#for i in range(250):
i=X_test.shape[0]-1
Xt = model.predict(X_test[i].reshape(1,7,1))
print('predicted:{0}, actual:{1}'.format(scl.inverse_transform(Xt),scl.inverse_transform(y_test[i].reshape(-1,1))))
pred.append(scl.inverse_transform(Xt))
act.append(scl.inverse_transform(y_test[i].reshape(-1,1)))


# In[ ]:


result_df = pd.DataFrame({'pred':list(np.reshape(pred, (-1))),'act':list(np.reshape(act, (-1)))})


# In[ ]:


#result_df.plot(kind='line')


# In[ ]:


Xt = model.predict(X_test)
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label='actual')
plt.plot(scl.inverse_transform(Xt), label='predict')
plt.legend()
plt.show()


# In[ ]:


X_test[X_test.shape[0]-1]


# In[ ]:


X_test[X_test.shape[0]-2]

